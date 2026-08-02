#!/usr/bin/env python3
"""Solve and audit the frozen P0685 registered 3D QUMOND field."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.field_solvers import boundary_mask
from voidscreen.metric_lensing_3d import (
    KPC_M,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.spatial_qumond_3d import solve_path_diluted_qumond

G_SI = 6.67430e-11
DEFAULT_CONFIG = ROOT / "configs" / "p0685_locked_path_qumond_3d_field.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_rms(x_values: np.ndarray, y_values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x_values[mask] ** 2 + y_values[mask] ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0685_field_or_deflection_score":
        raise RuntimeError("P0685 protocol is not frozen")
    parent_path = ROOT / protocol["map_parent"]
    parent = read_json(parent_path)
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0670 physical-map parent no longer passes")

    map_path = ROOT / protocol["map_input"]
    with np.load(map_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        density = data["stellar_volume_density_kg_m3"].astype(float) + data[
            "gas_volume_density_kg_m3"
        ].astype(float)
        map_a0 = float(data["a0_m_s2"])
    equation = protocol["equation"]
    a0 = float(equation["a0_m_s2"])
    if not np.isclose(a0, map_a0, rtol=0.0, atol=0.0):
        raise RuntimeError("P0685 a0 no longer matches the frozen physical map")
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M

    print("solving locked path-diluted potential-channel QUMOND", flush=True)
    solution = solve_path_diluted_qumond(
        density,
        spacing_m,
        gravitational_constant=G_SI,
        a0=a0,
        transition_depth=float(equation["chi_t"]),
        transition_power=float(equation["transition_power_n"]),
        extra_spatial_channels=float(equation["extra_spatial_channels"]),
        path_power=float(equation["path_power_q"]),
    )
    print(
        f"newtonian residual={solution.newtonian.normalized_residual_rms:.6g}; "
        f"candidate residual={solution.field.normalized_residual_rms:.6g}",
        flush=True,
    )
    deflection = photon_deflection_zero_slip(
        solution.field.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    candidate_magnitude = np.hypot(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
    )

    scalar_path = ROOT / protocol["scalar_comparator"]
    with np.load(scalar_path) as scalar_data:
        scalar_axis = scalar_data["axis_kpc"].astype(float)
        scalar_x = scalar_data["scalar_alpha_x_physical_arcsec"].astype(float)
        scalar_y = scalar_data["scalar_alpha_y_physical_arcsec"].astype(float)
    if not np.array_equal(axis_kpc, scalar_axis):
        raise RuntimeError("P0671 scalar comparator is not on the registered P0670 grid")

    x, y = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    radius = np.hypot(x, y)
    lower, upper = (float(value) for value in protocol["audit_region"]["strong_lens_radius_kpc"])
    annulus = (radius >= lower) & (radius <= upper)
    candidate_rms = vector_rms(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        annulus,
    )
    scalar_rms = vector_rms(scalar_x, scalar_y, annulus)
    candidate_scalar_ratio = candidate_rms / max(scalar_rms, np.finfo(float).tiny)
    candidate_median = float(np.median(candidate_magnitude[annulus]))
    candidate_curl = normalized_deflection_curl(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    edge = boundary_mask(density.shape)
    boundary_scale = max(
        float(np.max(np.abs(solution.boundary_potential[edge]))),
        np.finfo(float).tiny,
    )
    boundary_mismatch = float(
        np.max(np.abs(solution.field.potential[edge] - solution.boundary_potential[edge]))
        / boundary_scale
    )
    exponent_min = float(np.min(solution.channel_exponent))
    exponent_max = float(np.max(solution.channel_exponent))
    finite = bool(
        np.all(np.isfinite(solution.newtonian.potential))
        and np.all(np.isfinite(solution.field.potential))
        and all(np.all(np.isfinite(item)) for item in solution.newtonian.acceleration)
        and all(np.all(np.isfinite(item)) for item in solution.field.acceleration)
        and np.all(np.isfinite(solution.potential_depth))
        and np.all(np.isfinite(solution.potential_path_ratio))
        and np.all(np.isfinite(solution.path_survival))
        and np.all(np.isfinite(solution.channel_exponent))
        and np.all(np.isfinite(candidate_magnitude))
    )

    gates = protocol["predeclared_progression_gates"]
    gate_results = {
        "P0670_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0670_all_progression_gates_pass"]),
        "newtonian_residual": solution.newtonian.normalized_residual_rms
        <= float(gates["newtonian_normalized_residual_RMS_max"]),
        "candidate_residual": solution.field.normalized_residual_rms
        <= float(gates["candidate_normalized_residual_RMS_max"]),
        "candidate_convergence": solution.field.converged
        is bool(gates["candidate_solver_converged"]),
        "boundary": boundary_mismatch <= float(gates["boundary_maximum_relative_mismatch_max"]),
        "finite_fields": finite
        is bool(gates["all_potentials_accelerations_geometry_and_deflections_finite"]),
        "exponent_lower": exponent_min >= float(gates["channel_exponent_min"]),
        "exponent_upper": exponent_max <= float(gates["channel_exponent_max"]),
        "candidate_deflection_present": candidate_median
        >= float(gates["candidate_strong_lens_median_physical_deflection_arcsec_min"]),
        "intended_amplitude_lower": candidate_scalar_ratio
        >= float(gates["candidate_to_scalar_AQUAL_strong_lens_deflection_RMS_ratio_min"]),
        "intended_amplitude_upper": candidate_scalar_ratio
        <= float(gates["candidate_to_scalar_AQUAL_strong_lens_deflection_RMS_ratio_max"]),
        "candidate_curl": candidate_curl
        <= float(gates["candidate_normalized_deflection_curl_RMS_max"]),
        "no_fitted_gravity": int(equation["gravity_parameters_fit_to_RXJ2129"])
        == int(gates["gravity_parameters_fit_to_RXJ2129"]),
        "no_fitted_photon_amplitude": int(equation["photon_amplitudes_fit_to_RXJ2129"])
        == int(gates["photon_amplitudes_fit_to_RXJ2129"]),
        "no_raw_lens_score": not bool(gates["raw_lens_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "newtonian_normalized_residual_RMS": solution.newtonian.normalized_residual_rms,
        "candidate_normalized_residual_RMS": solution.field.normalized_residual_rms,
        "boundary_maximum_relative_mismatch": boundary_mismatch,
        "baryonic_center_of_mass_m": list(solution.center_of_mass),
        "potential_depth_min": float(np.min(solution.potential_depth)),
        "potential_depth_max": float(np.max(solution.potential_depth)),
        "potential_path_ratio_min": float(np.min(solution.potential_path_ratio)),
        "potential_path_ratio_median": float(np.median(solution.potential_path_ratio)),
        "potential_path_ratio_max": float(np.max(solution.potential_path_ratio)),
        "channel_exponent_min": exponent_min,
        "channel_exponent_median": float(np.median(solution.channel_exponent)),
        "channel_exponent_max": exponent_max,
        "candidate_strong_lens_median_physical_deflection_arcsec": candidate_median,
        "candidate_strong_lens_deflection_RMS_arcsec": candidate_rms,
        "scalar_AQUAL_strong_lens_deflection_RMS_arcsec": scalar_rms,
        "candidate_to_scalar_AQUAL_strong_lens_deflection_RMS_ratio": candidate_scalar_ratio,
        "candidate_normalized_deflection_curl_RMS": candidate_curl,
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        field_path,
        axis_kpc=axis_kpc,
        newtonian_potential_m2_s2=solution.newtonian.potential,
        candidate_potential_m2_s2=solution.field.potential,
        candidate_alpha_x_physical_arcsec=deflection.alpha_x_arcsec,
        candidate_alpha_y_physical_arcsec=deflection.alpha_y_arcsec,
        potential_depth=solution.potential_depth,
        potential_path_ratio=solution.potential_path_ratio,
        path_survival=solution.path_survival,
        channel_exponent=solution.channel_exponent,
    )
    report = {
        "report_version": "P0685-LOCKED-PATH-QUMOND-3D-FIELD-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_frozen_P0686_raw_topology": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "solver_source_sha256": sha256(ROOT / "src/voidscreen/spatial_qumond_3d.py"),
        "map_sha256": sha256(map_path),
        "scalar_comparator_sha256": sha256(scalar_path),
        "field_sha256": sha256(field_path),
        "locked_formula": {
            "a0_m_s2": a0,
            "chi_t": float(equation["chi_t"]),
            "transition_power_n": float(equation["transition_power_n"]),
            "extra_spatial_channels": float(equation["extra_spatial_channels"]),
            "path_power_q": float(equation["path_power_q"]),
            "gravity_parameters_fit_to_RXJ2129": int(equation["gravity_parameters_fit_to_RXJ2129"]),
            "photon_amplitudes_fit_to_RXJ2129": int(equation["photon_amplitudes_fit_to_RXJ2129"]),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "raw_lens_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    midplane = density.shape[2] // 2
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = (
        (candidate_magnitude, "physical deflection", "arcsec"),
        (solution.channel_exponent[:, :, midplane], "channel exponent p", "p"),
        (
            np.log10(np.maximum(solution.potential_depth[:, :, midplane], 1e-20)),
            "log10 potential depth",
            "log10 chi",
        ),
        (
            np.log10(np.maximum(solution.potential_path_ratio[:, :, midplane], 1e-20)),
            "log10 path ratio",
            "log10 eta",
        ),
    )
    for axis, (values, title, label) in zip(axes, panels, strict=True):
        image = axis.imshow(values.T, origin="lower", extent=extent, cmap="viridis")
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75, label=label)
    figure.tight_layout()
    figure.savefig(output / "p0685_locked_path_qumond_field.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0685 locked path-QUMOND 3D field

- Status: **{"PASS" if all_pass else "FAIL"}**.
- Newtonian/candidate equation residual: **{solution.newtonian.normalized_residual_rms:.4g} / {solution.field.normalized_residual_rms:.4g}**.
- Channel exponent min/median/max: **{exponent_min:.4g} / {np.median(solution.channel_exponent):.4g} / {exponent_max:.4g}**.
- Candidate strong-lens median/RMS physical deflection: **{candidate_median:.4g} / {candidate_rms:.4g} arcsec**.
- Candidate/scalar-AQUAL deflection RMS ratio: **{candidate_scalar_ratio:.4g}**.
- Normalized deflection curl: **{candidate_curl:.3g}**.
- Failed frozen gates: **{", ".join(failed) if failed else "none"}**.
- Raw lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
