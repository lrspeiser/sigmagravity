#!/usr/bin/env python3
"""Solve the frozen dual-transverse-survival RX J2129 field."""

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
from run_p0671_spent_rxj2129_absolute_3d_field_solve import (
    G_SI,
    relative_vector_rms,
    vector_rms,
)

from voidscreen.compound_activation_3d import exact_compound_path_activation_3d
from voidscreen.field_solvers import boundary_mask
from voidscreen.metric_lensing_3d import (
    KPC_M,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)
from voidscreen.transverse_confinement_3d import (
    solve_transverse_confinement_aqual_3d,
)

DEFAULT_CONFIG = (
    ROOT / "configs" / "p0677_spent_rxj2129_dual_transverse_survival_field.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0677_field_or_deflection_score":
        raise RuntimeError("P0677 protocol is not frozen")
    failure_parent = read_json(ROOT / protocol["failure_parent"])
    activation_parent = read_json(ROOT / protocol["activation_parent"])
    scalar_reference = read_json(ROOT / protocol["scalar_reference_result"])
    scalar_field_path = ROOT / protocol["scalar_field_input"]
    map_path = ROOT / protocol["map_input"]

    with np.load(map_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        stars = data["stellar_volume_density_kg_m3"].astype(float)
        gas = data["gas_volume_density_kg_m3"].astype(float)
        boundary = data["simple_mond_boundary_m2_s2"].astype(float)
        map_a0 = float(data["a0_m_s2"])
    with np.load(scalar_field_path) as data:
        if not np.array_equal(axis_kpc, data["axis_kpc"].astype(float)):
            raise RuntimeError("P0670 and P0674 axes differ")
        scalar_potential = data["scalar_potential_m2_s2"].astype(float)
        scalar_alpha_x = data["scalar_alpha_x_physical_arcsec"].astype(float)
        scalar_alpha_y = data["scalar_alpha_y_physical_arcsec"].astype(float)

    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M
    candidate = protocol["candidate"]
    activation = exact_compound_path_activation_3d(
        stars,
        gas,
        spacing_m,
        a0=map_a0,
        coherence_length=float(candidate["coherence_length_kpc"]) * KPC_M,
        coherence_power=float(candidate["coherence_power"]),
        mu_floor=float(candidate["mu_floor"]),
    )
    sigma = activation.sigma
    transverse_dimensions = int(candidate["transverse_dimensions"])
    effective_sigma = 1.0 - np.power(1.0 - sigma, transverse_dimensions)
    directions = activation.local.transport_direction
    total_density = stars + gas
    recomputed_sigma = float(np.sum(sigma * total_density) / np.sum(total_density))
    expected_sigma = float(
        activation_parent["metrics"]["spent_RXJ2129_mass_weighted_sigma"]
    )

    source = 4.0 * np.pi * G_SI * total_density
    settings = protocol["solver"]
    print("solving dual-transverse-survival AQUAL", flush=True)
    dual = solve_transverse_confinement_aqual_3d(
        source,
        spacing_m,
        boundary,
        effective_sigma,
        *directions,
        a0=map_a0,
        residual_tolerance=float(settings["nonlinear_residual_tolerance"]),
        maximum_nonlinear_iterations=int(settings["maximum_nonlinear_iterations"]),
        maximum_linear_iterations=int(settings["maximum_linear_iterations"]),
        linear_relative_tolerance=float(settings["linear_relative_tolerance"]),
        damping=float(settings["picard_damping"]),
        mu_floor=float(settings["mu_floor"]),
    )
    print(
        f"dual: converged={dual.converged} residual={dual.normalized_residual_rms:.6g} "
        f"iterations={dual.nonlinear_iterations}",
        flush=True,
    )
    deflection = photon_deflection_zero_slip(
        dual.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    x_values, y_values = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    radius = np.hypot(x_values, y_values)
    lower, upper = (
        float(value) for value in protocol["audit_region"]["strong_lens_radius_kpc"]
    )
    annulus = (radius >= lower) & (radius <= upper)
    scalar_magnitude = np.hypot(scalar_alpha_x, scalar_alpha_y)
    dual_magnitude = np.hypot(deflection.alpha_x_arcsec, deflection.alpha_y_arcsec)
    scalar_rms = vector_rms(scalar_alpha_x, scalar_alpha_y, annulus)
    dual_rms = vector_rms(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        annulus,
    )
    response_ratio = dual_rms / max(scalar_rms, np.finfo(float).tiny)
    relative_difference = relative_vector_rms(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        scalar_alpha_x,
        scalar_alpha_y,
        annulus,
    )
    dual_curl = normalized_deflection_curl(
        deflection.alpha_x_arcsec,
        deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    boundary_cells = boundary_mask(source.shape)
    boundary_scale = max(
        float(np.max(np.abs(boundary[boundary_cells]))),
        np.finfo(float).tiny,
    )
    boundary_mismatch = float(
        np.max(np.abs(dual.potential[boundary_cells] - boundary[boundary_cells]))
    ) / boundary_scale
    finite = bool(
        np.all(np.isfinite(dual.potential))
        and all(np.all(np.isfinite(component)) for component in dual.acceleration)
        and np.all(np.isfinite(dual_magnitude))
    )
    minimum_eigenvalue = float(dual.metadata["minimum_constitutive_eigenvalue"])

    gates = protocol["predeclared_progression_gates"]
    equations = protocol["field_equations"]
    previous_ratio = float(
        failure_parent["metrics"][
            "confinement_to_scalar_strong_lens_deflection_RMS_ratio"
        ]
    )
    gate_results = {
        "P0676_failed": failure_parent["status"] == gates["P0676_status"],
        "P0676_below_strength_gate": previous_ratio
        <= float(gates["P0676_confinement_to_scalar_RMS_ratio_max"]),
        "P0673_parent": bool(activation_parent["all_progression_gates_pass"])
        is bool(gates["P0673_all_progression_gates_pass"]),
        "coefficient_reproduction": abs(recomputed_sigma - expected_sigma)
        <= float(gates["recomputed_compound_mass_weighted_sigma_absolute_tolerance"]),
        "dual_residual": dual.normalized_residual_rms
        <= float(gates["dual_normalized_residual_RMS_max"]),
        "dual_convergence": dual.converged is bool(gates["dual_solver_converged"]),
        "boundary": boundary_mismatch
        <= float(gates["boundary_maximum_relative_mismatch_max"]),
        "finite_fields": finite
        is bool(gates["all_potentials_accelerations_and_deflections_finite"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_dual_constitutive_eigenvalue_strictly_positive"]),
        "response_ratio_lower": response_ratio
        >= float(gates["dual_to_scalar_strong_lens_deflection_RMS_ratio_min"]),
        "response_ratio_upper": response_ratio
        <= float(gates["dual_to_scalar_strong_lens_deflection_RMS_ratio_max"]),
        "dual_change_nonperturbative": relative_difference
        >= float(gates["dual_minus_scalar_strong_lens_relative_RMS_min"]),
        "dual_change_stable": relative_difference
        <= float(gates["dual_minus_scalar_strong_lens_relative_RMS_max"]),
        "dual_curl": dual_curl
        <= float(gates["dual_normalized_deflection_curl_RMS_max"]),
        "two_dimensions": transverse_dimensions == int(gates["transverse_dimensions"]),
        "dimension_not_fitted": bool(candidate["transverse_dimensions_fitted"])
        is bool(gates["transverse_dimensions_fitted"]),
        "zero_slip": float(equations["gravitational_slip"])
        == float(gates["gravitational_slip_exactly_zero"]),
        "no_fitted_photon_amplitude": bool(equations["fitted_photon_amplitude"])
        is bool(gates["fitted_photon_amplitude"]),
        "no_new_constants": int(equations["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(equations["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "no_raw_lens_score": not bool(gates["raw_lens_score_computed"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "recomputed_compound_mass_weighted_sigma": recomputed_sigma,
        "mass_weighted_effective_dual_confinement_sigma": float(
            np.sum(effective_sigma * total_density) / np.sum(total_density)
        ),
        "dual_normalized_residual_RMS": dual.normalized_residual_rms,
        "dual_nonlinear_iterations": dual.nonlinear_iterations,
        "boundary_maximum_relative_mismatch": boundary_mismatch,
        "minimum_dual_constitutive_eigenvalue": minimum_eigenvalue,
        "scalar_strong_lens_median_physical_deflection_arcsec": float(
            np.median(scalar_magnitude[annulus])
        ),
        "dual_strong_lens_median_physical_deflection_arcsec": float(
            np.median(dual_magnitude[annulus])
        ),
        "scalar_strong_lens_deflection_RMS_arcsec": scalar_rms,
        "dual_strong_lens_deflection_RMS_arcsec": dual_rms,
        "dual_to_scalar_strong_lens_deflection_RMS_ratio": response_ratio,
        "dual_minus_scalar_strong_lens_relative_RMS": relative_difference,
        "dual_normalized_deflection_curl_RMS": dual_curl,
        "P0674_scalar_reference_residual_RMS": scalar_reference["metrics"][
            "scalar_normalized_residual_RMS"
        ],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        field_path,
        axis_kpc=axis_kpc,
        scalar_potential_m2_s2=scalar_potential,
        dual_potential_m2_s2=dual.potential,
        scalar_alpha_x_physical_arcsec=scalar_alpha_x,
        scalar_alpha_y_physical_arcsec=scalar_alpha_y,
        dual_alpha_x_physical_arcsec=deflection.alpha_x_arcsec,
        dual_alpha_y_physical_arcsec=deflection.alpha_y_arcsec,
        compound_sigma=sigma,
        effective_dual_confinement_sigma=effective_sigma,
    )
    report = {
        "report_version": "P0677-SPENT-RXJ2129-DUAL-TRANSVERSE-SURVIVAL-FIELD-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_raw_lens_topology_audit": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "solver_source_sha256": sha256(
            ROOT / "src/voidscreen/transverse_confinement_3d.py"
        ),
        "map_sha256": sha256(map_path),
        "scalar_field_source_sha256": sha256(scalar_field_path),
        "field_sha256": sha256(field_path),
        "metrics": metrics,
        "gate_results": gate_results,
        "raw_lens_score_computed": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    difference = np.hypot(
        deflection.alpha_x_arcsec - scalar_alpha_x,
        deflection.alpha_y_arcsec - scalar_alpha_y,
    )
    for axis, values, title in zip(
        axes,
        (scalar_magnitude, dual_magnitude, difference),
        ("scalar", "dual transverse survival", "dual - scalar"),
        strict=True,
    ):
        image = axis.imshow(values.T, origin="lower", extent=extent, cmap="viridis")
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75, label="arcsec")
    figure.tight_layout()
    figure.savefig(output / "p0677_dual_transverse_survival_fields.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0677 spent RX J2129 dual-transverse-survival field

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Dual residual / iterations: **{dual.normalized_residual_rms:.4g} / {dual.nonlinear_iterations}**.
- Scalar/dual strong-lens median physical deflection: **{metrics['scalar_strong_lens_median_physical_deflection_arcsec']:.4g} / {metrics['dual_strong_lens_median_physical_deflection_arcsec']:.4g} arcsec**.
- Dual/scalar deflection RMS ratio: **{response_ratio:.6g}**.
- Dual-minus-scalar relative RMS: **{relative_difference:.6g}**.
- Dual normalized curl: **{dual_curl:.3g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Raw lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
