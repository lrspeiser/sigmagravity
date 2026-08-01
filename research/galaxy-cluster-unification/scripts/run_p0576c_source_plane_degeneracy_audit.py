#!/usr/bin/env python3
"""Diagnose mass-sheet/source-collapse behavior in the P0576 power scan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0574_symmetry_gated_arrival_microvariation import quarter_turn_asymmetry  # noqa: E402
from run_p0575_smacs0723_raw_position import (  # noqa: E402
    deflection_from_surface,
    fit_positive_amplitude,
    interpolate_deflection,
    lens_efficiency,
    sha256,
)
from run_p0575b_raw_position_robustness import build_maps  # noqa: E402
from run_p0576_fractional_routed_propagator import fractional_deflection  # noqa: E402


def diagnostic_row(name, theta, scaled, families, calibration_mask):
    amplitude = fit_positive_amplitude(theta, scaled, families, calibration_mask)
    beta = theta - amplitude * scaled
    theta_centered = theta - np.mean(theta, axis=0)
    scaled_centered = scaled - np.mean(scaled, axis=0)
    affine_scale = float(np.sum(theta_centered * scaled_centered) / np.sum(scaled_centered**2))
    affine_residual = theta_centered - affine_scale * scaled_centered
    mass_sheet_r2 = float(1.0 - np.sum(affine_residual**2) / np.sum(theta_centered**2))
    family_means = []
    within = []
    for family in np.unique(families):
        mask = families == family
        mean = np.mean(beta[mask], axis=0)
        family_means.append(mean)
        within.append(beta[mask] - mean)
    family_means = np.asarray(family_means)
    within = np.vstack(within)
    family_dispersion = float(
        np.sqrt(np.mean(np.sum((family_means - np.mean(family_means, axis=0)) ** 2, axis=1)))
    )
    theta_radius = float(np.sqrt(np.mean(np.sum(theta_centered**2, axis=1))))
    beta_centered = beta - np.mean(beta, axis=0)
    beta_radius = float(np.sqrt(np.mean(np.sum(beta_centered**2, axis=1))))
    return {
        "model": name,
        "primary_amplitude": amplitude,
        "global_mass_sheet_R2": mass_sheet_r2,
        "all_family_within_RMS_arcsec": float(np.sqrt(np.mean(np.sum(within**2, axis=1)))),
        "family_mean_source_dispersion_arcsec": family_dispersion,
        "source_radius_arcsec": beta_radius,
        "no_lens_position_radius_arcsec": theta_radius,
        "source_radius_ratio": beta_radius / theta_radius,
    }


def main() -> None:
    protocol_path = ROOT / "configs/p0576c_source_plane_degeneracy_audit_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0576B_before_mass_sheet_diagnostics":
        raise RuntimeError("P0576C protocol is not frozen")
    images = pd.read_csv(ROOT / protocol["inputs"]["p0575_images"], dtype={"family": str})
    data, maps = build_maps(protocol, images)
    local = maps["local_control"]
    gated_map = maps["p0574_symmetry_gated"]
    q90 = quarter_turn_asymmetry(data)
    gate = q90**4 / (q90**4 + 0.05**4)
    map_fraction = 0.8 * gate
    destination = np.maximum((gated_map - (1.0 - map_fraction) * local) / map_fraction, 0.0)
    destination /= np.sum(destination)
    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    calibration_mask = np.isin(families, ["1", "2"])
    efficiency = lens_efficiency(0.39, images.source_redshift.to_numpy(float))
    rows = []
    for name, surface in (
        ("local_control", local),
        ("lenstool_map_reference", maps["lenstool_map_reference"]),
    ):
        ax, ay = deflection_from_surface(surface, 10.0, 3)
        sampled = interpolate_deflection(ax, ay, images, data.axis)
        rows.append(diagnostic_row(name, theta, efficiency[:, None] * sampled, families, calibration_mask))
    for power in map(float, protocol["diagnostic_powers"]):
        ax, ay = fractional_deflection(destination, 10.0, power, 60.0, 3)
        sampled = interpolate_deflection(ax, ay, images, data.axis)
        rows.append(
            diagnostic_row(
                f"fractional_p{power:g}",
                theta,
                efficiency[:, None] * sampled,
                families,
                calibration_mask,
            )
        )
    diagnostics = pd.DataFrame(rows)
    p26 = diagnostics[diagnostics.model.eq("fractional_p2.6")].iloc[0]
    fractional = diagnostics[diagnostics.model.str.startswith("fractional")].copy()
    fractional["power"] = fractional.model.str.replace("fractional_p", "", regex=False).astype(float)
    monotonic = bool(
        np.all(np.diff(fractional.sort_values("power").all_family_within_RMS_arcsec.to_numpy()) < 0.0)
    )
    cfg = protocol["decision_gates"]
    gates = {
        "p2p6_mass_sheet_R2_pass": bool(p26.global_mass_sheet_R2 <= float(cfg["p2p6_global_mass_sheet_R2_max"])),
        "p2p6_source_radius_pass": bool(p26.source_radius_ratio >= float(cfg["p2p6_source_radius_ratio_min"])),
        "nonmonotonic_interior_behavior_pass": bool(not monotonic),
    }
    gates["fractional_source_plane_gain_is_non_degenerate"] = bool(all(gates.values()))
    gates["mass_sheet_resistant_metric_required"] = bool(not gates["fractional_source_plane_gain_is_non_degenerate"])
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output / protocol["outputs"]["diagnostics"], index=False)
    report = {
        "report_version": "P0576C-SOURCE-PLANE-DEGENERACY-AUDIT-RESULTS-0.1.0",
        "status": "complete_source_plane_degeneracy_audit",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "diagnostics": rows,
        "result": {
            "p2p6_global_mass_sheet_R2": float(p26.global_mass_sheet_R2),
            "p2p6_source_radius_ratio": float(p26.source_radius_ratio),
            "p2p6_family_mean_source_dispersion_arcsec": float(p26.family_mean_source_dispersion_arcsec),
            "fractional_within_family_RMS_monotonically_decreases": monotonic,
        },
        "gates": gates,
        "interpretation": "A high-p field that is nearly affine in image position and collapses unrelated family means is exploiting the source-plane/mass-sheet degeneracy rather than uniquely predicting multiple-image roots.",
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0576C source-plane degeneracy audit",
                "",
                f"p=2.6 mass-sheet R2: **{p26.global_mass_sheet_R2:.4f}**.",
                f"p=2.6 inferred-source radius ratio: **{p26.source_radius_ratio:.4f}**.",
                f"Mass-sheet-resistant metric required: **{gates['mass_sheet_resistant_metric_required']}**.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(fractional.power, fractional.all_family_within_RMS_arcsec, marker="o")
    axes[0].set(xlabel="p", ylabel="within-family source RMS")
    axes[1].plot(fractional.power, fractional.global_mass_sheet_R2, marker="o")
    axes[1].axhline(float(cfg["p2p6_global_mass_sheet_R2_max"]), color="black", ls="--")
    axes[1].set(xlabel="p", ylabel="global mass-sheet R2")
    axes[2].plot(fractional.power, fractional.source_radius_ratio, marker="o")
    axes[2].axhline(float(cfg["p2p6_source_radius_ratio_min"]), color="black", ls="--")
    axes[2].set(xlabel="p", ylabel="inferred/no-lens source radius")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
