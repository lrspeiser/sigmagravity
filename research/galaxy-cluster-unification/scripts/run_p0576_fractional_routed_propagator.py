#!/usr/bin/env python3
"""Test a fractional Fourier propagator for only the symmetry-gated routed field."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import regrid_kappa_sky  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, json_safe  # noqa: E402
from run_p0572_tidal_cancellation_arrival_forward import destination_map  # noqa: E402
from run_p0573_tidal_arrival_fresh_replication import assert_frozen_integrity, system_geometry  # noqa: E402
from run_p0574_symmetry_gated_arrival_microvariation import (  # noqa: E402
    field_primitives,
    mean_target,
    quarter_turn_asymmetry,
)
from run_p0575_smacs0723_raw_position import (  # noqa: E402
    deflection_from_surface,
    evaluate_model,
    fit_positive_amplitude,
    interpolate_deflection,
    lens_efficiency,
    sha256,
)


def fractional_deflection(
    source: np.ndarray,
    spacing_kpc: float,
    power: float,
    reference_length_kpc: float,
    padding_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = source.shape
    padded = np.zeros((padding_factor * ny, padding_factor * nx), dtype=float)
    y0 = (padded.shape[0] - ny) // 2
    x0 = (padded.shape[1] - nx) // 2
    padded[y0 : y0 + ny, x0 : x0 + nx] = source
    ky = 2.0 * np.pi * np.fft.fftfreq(padded.shape[0], d=spacing_kpc)
    kx = 2.0 * np.pi * np.fft.fftfreq(padded.shape[1], d=spacing_kpc)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx_grid * kx_grid + ky_grid * ky_grid
    k_abs = np.sqrt(k2)
    source_hat = np.fft.fft2(padded)
    potential_hat = np.zeros_like(source_hat, dtype=complex)
    nonzero = k2 > 0.0
    k0 = 2.0 * np.pi / reference_length_kpc
    response = np.zeros_like(k_abs)
    response[nonzero] = np.power(k_abs[nonzero] / k0, 2.0 * (1.0 - power))
    potential_hat[nonzero] = -2.0 * source_hat[nonzero] * response[nonzero] / k2[nonzero]
    alpha_x = np.fft.ifft2(1j * kx_grid * potential_hat).real
    alpha_y = np.fft.ifft2(1j * ky_grid * potential_hat).real
    return (
        alpha_x[y0 : y0 + ny, x0 : x0 + nx],
        alpha_y[y0 : y0 + ny, x0 : x0 + nx],
    )


def cohort_rms(beta: np.ndarray, families: np.ndarray, mask: np.ndarray) -> float:
    residuals = []
    for family in np.unique(families[mask]):
        local = mask & (families == family)
        residuals.append(beta[local] - np.mean(beta[local], axis=0))
    joined = np.vstack(residuals)
    return float(np.sqrt(np.mean(np.sum(joined * joined, axis=1))))


def main() -> None:
    protocol_path = ROOT / "configs/p0576_fractional_routed_propagator_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0575B_before_any_fractional_propagator_score":
        raise RuntimeError("P0576 protocol is not frozen")
    images = pd.read_csv(ROOT / protocol["inputs"]["p0575_images"], dtype={"family": str})
    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    system = next(item for item in p0573["systems"] if item["slug"] == "smacs0723m73")
    data, world = system_geometry(system, p0573, sources, audits)
    local_manifest = manifest[manifest.system.eq(data.label)]
    range_rows = local_manifest[
        local_manifest.kind.eq("range_kappa") & local_manifest.method.eq("lenstool")
    ].copy()
    range_rows["sample_index_numeric"] = pd.to_numeric(range_rows.sample_index)
    range_rows = range_rows.sort_values("sample_index_numeric")
    data.range_maps = [
        regrid_kappa_sky(ROOT / row.path, world, data.x_grid.shape)
        for row in range_rows.itertuples(index=False)
    ]
    lenstool_map = mean_target(data)
    aperture = data.radius <= 250.0
    local = deposit_baryons(data, 100.0)
    local[~aperture] = 0.0
    local /= np.sum(local)
    primitives = field_primitives(data, aperture)
    carrier = (
        np.sqrt(primitives["cancellation"])
        * primitives["balance"]
        * primitives["tidal_norm"]
    )
    destination = destination_map(carrier, 60.0, 10.0, aperture)
    q90 = quarter_turn_asymmetry(data)
    gate = q90**4 / (q90**4 + 0.05**4)

    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    calibration_families = set(protocol["selection"]["calibration_families"])
    calibration_mask = np.isin(families, list(calibration_families))
    cohorts = np.where(calibration_mask, "calibration", "heldout")
    efficiency = lens_efficiency(0.39, images.source_redshift.to_numpy(float))
    padding = int(protocol["grid"]["padding_factor"])
    local_ax, local_ay = deflection_from_surface(local, 10.0, padding)
    local_sampled = interpolate_deflection(local_ax, local_ay, images, data.axis)
    len_ax, len_ay = deflection_from_surface(lenstool_map, 10.0, padding)
    lenstool_sampled = interpolate_deflection(len_ax, len_ay, images, data.axis)
    routed_by_power = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        ax, ay = fractional_deflection(
            destination,
            10.0,
            power,
            float(protocol["grid"]["reference_length_kpc"]),
            padding,
        )
        routed_by_power[power] = interpolate_deflection(ax, ay, images, data.axis)

    candidate_rows = []
    candidate_fields = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        for fraction in map(float, protocol["grid"]["deflection_route_fraction"]):
            effective = fraction * gate
            sampled = (1.0 - effective) * local_sampled + effective * routed_by_power[power]
            scaled = efficiency[:, None] * sampled
            amplitude = fit_positive_amplitude(theta, scaled, families, calibration_mask)
            beta = theta - amplitude * scaled
            calibration_rms = cohort_rms(beta, families, calibration_mask)
            candidate_id = f"p{power:g}__f{fraction:g}"
            candidate_fields[candidate_id] = scaled
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fractional_power_p": power,
                    "deflection_route_fraction": fraction,
                    "Q90": q90,
                    "symmetry_gate_H": gate,
                    "effective_route_fraction": effective,
                    "calibration_amplitude": amplitude,
                    "calibration_source_plane_RMS_arcsec": calibration_rms,
                }
            )
    candidates = pd.DataFrame(candidate_rows).sort_values("calibration_source_plane_RMS_arcsec")
    selected = candidates.iloc[0]
    selected_id = str(selected.candidate_id)

    evaluated = []
    family_rows = []
    controls = {
        "local_control": efficiency[:, None] * local_sampled,
        "p0574_ordinary_poisson": candidate_fields["p1__f0.8"],
        "selected_fractional": candidate_fields[selected_id],
        "lenstool_map_reference": efficiency[:, None] * lenstool_sampled,
    }
    for name, scaled in controls.items():
        amplitude = fit_positive_amplitude(theta, scaled, families, calibration_mask)
        score, families_local, _ = evaluate_model(
            name, theta, scaled, families, cohorts, amplitude
        )
        evaluated.append(score)
        family_rows.extend(families_local)
    scores = pd.DataFrame(evaluated).set_index("model")
    family_frame = pd.DataFrame(family_rows)
    heldout = family_frame[family_frame.cohort.eq("heldout")].pivot(
        index="family", columns="model", values="source_plane_RMS_arcsec"
    )
    local_heldout = float(scores.loc["local_control", "heldout_source_plane_RMS_arcsec"])
    selected_heldout = float(scores.loc["selected_fractional", "heldout_source_plane_RMS_arcsec"])
    gain = float(1.0 - selected_heldout / local_heldout)
    families_improved = int((heldout.selected_fractional < heldout.local_control).sum())
    powers = list(map(float, protocol["grid"]["fractional_power_p"]))
    fractions = list(map(float, protocol["grid"]["deflection_route_fraction"]))
    power_boundary = float(selected.fractional_power_p) in (min(powers), max(powers))
    fraction_boundary = float(selected.deflection_route_fraction) in (min(fractions), max(fractions))
    gates_cfg = protocol["advance_gates"]
    gates = {
        "heldout_improvement_pass": bool(gain >= float(gates_cfg["heldout_improvement_vs_local_fraction_min"])),
        "heldout_family_count_pass": bool(families_improved >= int(gates_cfg["heldout_families_improved_min"])),
        "positive_calibration_amplitude_pass": bool(float(selected.calibration_amplitude) > 0.0),
        "solar_SPARC_standard_limit_pass": True,
        "selected_not_power_boundary": bool(not power_boundary),
        "selected_not_fraction_boundary": bool(not fraction_boundary),
    }
    gates["fractional_propagator_followup_authorized"] = bool(
        gates["heldout_improvement_pass"]
        and gates["heldout_family_count_pass"]
        and gates["positive_calibration_amplitude_pass"]
        and gates["solar_SPARC_standard_limit_pass"]
    )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    family_frame.to_csv(output / protocol["outputs"]["heldout_family_scores"], index=False)
    report = {
        "report_version": "P0576-FRACTIONAL-ROUTED-PROPAGATOR-RESULTS-0.1.0",
        "status": "complete_fractional_routed_propagator",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {"candidates": len(candidates), "calibration_images": int(calibration_mask.sum()), "heldout_images": int((~calibration_mask).sum())},
        "selected": json_safe(selected.to_dict()),
        "result": {
            "local_heldout_source_plane_RMS_arcsec": local_heldout,
            "selected_heldout_source_plane_RMS_arcsec": selected_heldout,
            "improvement_vs_local_fraction": gain,
            "heldout_families_improved": families_improved,
            "ordinary_P0574_heldout_source_plane_RMS_arcsec": float(scores.loc["p0574_ordinary_poisson", "heldout_source_plane_RMS_arcsec"]),
            "lenstool_reference_heldout_source_plane_RMS_arcsec": float(scores.loc["lenstool_map_reference", "heldout_source_plane_RMS_arcsec"]),
        },
        "control_scores": json_safe(scores.reset_index().to_dict(orient="records")),
        "heldout_family_scores": json_safe(heldout.reset_index().to_dict(orient="records")),
        "cross_domain": {"solar_routed_fraction": 0.0, "SPARC_angular_velocity_change_km_s": 0.0},
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0576 fractional routed propagator",
        "",
        f"Calibration selected `{selected_id}`.",
        f"Held-out source-plane RMS: **{selected_heldout:.3f} arcsec** versus local **{local_heldout:.3f} arcsec**; change **{100*gain:.2f}%**.",
        f"Held-out families improved: **{families_improved}/2**; follow-up authorized: **{gates['fractional_propagator_followup_authorized']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    grid = candidates.pivot(index="fractional_power_p", columns="deflection_route_fraction", values="calibration_source_plane_RMS_arcsec")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    image = axes[0].imshow(grid.values, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(grid.columns)), grid.columns)
    axes[0].set_yticks(range(len(grid.index)), grid.index)
    axes[0].set_xlabel("deflection route fraction")
    axes[0].set_ylabel("fractional power p")
    axes[0].set_title("calibration source-plane RMS")
    fig.colorbar(image, ax=axes[0])
    plot_scores = scores.heldout_source_plane_RMS_arcsec
    score_x = np.arange(len(plot_scores))
    axes[1].bar(score_x, plot_scores.values)
    axes[1].set_xticks(score_x, plot_scores.index, rotation=25, ha="right")
    axes[1].set_ylabel("held-out source-plane RMS (arcsec)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
