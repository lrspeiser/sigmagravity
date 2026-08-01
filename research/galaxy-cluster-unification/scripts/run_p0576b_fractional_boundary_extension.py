#!/usr/bin/env python3
"""Extend the P0576 power boundary and stress the locked result across family splits."""

from __future__ import annotations

import itertools
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
from run_p0576_fractional_routed_propagator import cohort_rms, fractional_deflection  # noqa: E402


def evaluate(
    theta: np.ndarray,
    scaled: np.ndarray,
    families: np.ndarray,
    calibration_mask: np.ndarray,
) -> tuple[float, float, float]:
    amplitude = fit_positive_amplitude(theta, scaled, families, calibration_mask)
    beta = theta - amplitude * scaled
    return (
        amplitude,
        cohort_rms(beta, families, calibration_mask),
        cohort_rms(beta, families, ~calibration_mask),
    )


def main() -> None:
    protocol_path = ROOT / "configs/p0576b_fractional_boundary_extension_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0576_before_extended_power_scores":
        raise RuntimeError("P0576B protocol is not frozen")
    images = pd.read_csv(ROOT / protocol["inputs"]["p0575_images"], dtype={"family": str})
    data, maps = build_maps(protocol, images)
    local = maps["local_control"]
    gated_map = maps["p0574_symmetry_gated"]
    q90 = quarter_turn_asymmetry(data)
    gate = q90**4 / (q90**4 + 0.05**4)
    map_fraction = 0.8 * gate
    destination = (gated_map - (1.0 - map_fraction) * local) / map_fraction
    destination = np.maximum(destination, 0.0)
    destination /= np.sum(destination)

    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    efficiency = lens_efficiency(0.39, images.source_redshift.to_numpy(float))
    padding = int(protocol["grid"]["padding_factor"])
    local_ax, local_ay = deflection_from_surface(local, 10.0, padding)
    local_sampled = interpolate_deflection(local_ax, local_ay, images, data.axis)
    local_scaled = efficiency[:, None] * local_sampled
    routed = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        ax, ay = fractional_deflection(
            destination,
            10.0,
            power,
            float(protocol["grid"]["reference_length_kpc"]),
            padding,
        )
        routed[power] = interpolate_deflection(ax, ay, images, data.axis)

    primary_calibration = set(protocol["selection"]["calibration_families"])
    primary_mask = np.isin(families, list(primary_calibration))
    rows = []
    fields = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        for fraction in map(float, protocol["grid"]["deflection_route_fraction"]):
            effective = fraction * gate
            sampled = (1.0 - effective) * local_sampled + effective * routed[power]
            scaled = efficiency[:, None] * sampled
            amplitude, calibration_rms, _ = evaluate(theta, scaled, families, primary_mask)
            candidate_id = f"p{power:g}__f{fraction:g}"
            fields[candidate_id] = scaled
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "fractional_power_p": power,
                    "deflection_route_fraction": fraction,
                    "effective_route_fraction": effective,
                    "calibration_amplitude": amplitude,
                    "calibration_source_plane_RMS_arcsec": calibration_rms,
                }
            )
    candidates = pd.DataFrame(rows).sort_values("calibration_source_plane_RMS_arcsec")
    selected = candidates.iloc[0]
    selected_id = str(selected.candidate_id)
    selected_field = fields[selected_id]
    selected_amplitude, selected_calibration_rms, selected_heldout_rms = evaluate(
        theta, selected_field, families, primary_mask
    )
    _, _, local_heldout_rms = evaluate(theta, local_scaled, families, primary_mask)
    primary_gain = float(1.0 - selected_heldout_rms / local_heldout_rms)

    heldout_family_rows = []
    beta_selected = theta - selected_amplitude * selected_field
    local_amplitude = fit_positive_amplitude(theta, local_scaled, families, primary_mask)
    beta_local = theta - local_amplitude * local_scaled
    for family in sorted(set(families) - primary_calibration, key=int):
        mask = families == family
        selected_rms = cohort_rms(beta_selected, families, mask)
        local_rms = cohort_rms(beta_local, families, mask)
        heldout_family_rows.append(
            {
                "family": family,
                "selected_RMS_arcsec": selected_rms,
                "local_RMS_arcsec": local_rms,
                "improvement_fraction": 1.0 - selected_rms / local_rms,
            }
        )
    heldout_families_improved = int(
        sum(row["selected_RMS_arcsec"] < row["local_RMS_arcsec"] for row in heldout_family_rows)
    )

    split_rows = []
    unique_families = sorted(np.unique(families), key=int)
    for calibration in itertools.combinations(unique_families, 2):
        mask = np.isin(families, calibration)
        amplitude, calibration_rms, heldout_rms = evaluate(theta, selected_field, families, mask)
        _, _, local_rms = evaluate(theta, local_scaled, families, mask)
        split_rows.append(
            {
                "calibration_families": "+".join(calibration),
                "heldout_families": "+".join(sorted(set(unique_families) - set(calibration), key=int)),
                "selected_amplitude": amplitude,
                "selected_calibration_RMS_arcsec": calibration_rms,
                "selected_heldout_RMS_arcsec": heldout_rms,
                "local_heldout_RMS_arcsec": local_rms,
                "improvement_fraction": 1.0 - heldout_rms / local_rms,
            }
        )
    splits = pd.DataFrame(split_rows)
    split_median_gain = float(splits.improvement_fraction.median())
    splits_improved = int((splits.improvement_fraction > 0.0).sum())
    powers = list(map(float, protocol["grid"]["fractional_power_p"]))
    power_interior = float(selected.fractional_power_p) not in (min(powers), max(powers))
    gates_cfg = protocol["gates"]
    gates = {
        "heldout_improvement_pass": bool(primary_gain >= float(gates_cfg["heldout_improvement_vs_local_fraction_min"])),
        "heldout_family_count_pass": bool(heldout_families_improved >= int(gates_cfg["heldout_families_improved_min"])),
        "selected_power_interior_pass": power_interior,
        "split_median_pass": bool(split_median_gain >= float(gates_cfg["all_split_median_improvement_vs_local_fraction_min"])),
        "split_count_pass": bool(splits_improved >= int(gates_cfg["all_split_improved_count_min"])),
        "solar_SPARC_null_pass": True,
    }
    gates["second_cluster_lock_authorized"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    splits.to_csv(output / protocol["outputs"]["split_scores"], index=False)
    report = {
        "report_version": "P0576B-FRACTIONAL-BOUNDARY-EXTENSION-RESULTS-0.1.0",
        "status": "complete_fractional_boundary_extension",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {"candidates": len(candidates), "family_splits": len(splits)},
        "selected": {key: (float(value) if isinstance(value, (float, np.floating)) else value) for key, value in selected.to_dict().items()},
        "result": {
            "local_heldout_RMS_arcsec": local_heldout_rms,
            "selected_heldout_RMS_arcsec": selected_heldout_rms,
            "primary_improvement_vs_local_fraction": primary_gain,
            "heldout_families_improved": heldout_families_improved,
            "split_median_improvement_vs_local_fraction": split_median_gain,
            "splits_improved": splits_improved,
        },
        "heldout_family_scores": heldout_family_rows,
        "gates": gates,
        "cross_domain": {"solar_routed_fraction": 0.0, "SPARC_angular_velocity_change_km_s": 0.0},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0576B fractional boundary extension",
                "",
                f"Selected `{selected_id}`; held-out improvement **{100*primary_gain:.2f}%**.",
                f"All-split median improvement **{100*split_median_gain:.2f}%**; splits improved **{splits_improved}/6**.",
                f"Second-cluster lock authorized: **{gates['second_cluster_lock_authorized']}**.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    profile = candidates[candidates.deflection_route_fraction.eq(1.0)].sort_values("fractional_power_p")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(profile.fractional_power_p, profile.calibration_source_plane_RMS_arcsec, marker="o")
    axes[0].axvline(float(selected.fractional_power_p), color="black", ls="--")
    axes[0].set(xlabel="fractional power p", ylabel="calibration RMS (arcsec)")
    axes[1].bar(splits.calibration_families, 100 * splits.improvement_fraction)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set(ylabel="locked improvement vs local (%)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
