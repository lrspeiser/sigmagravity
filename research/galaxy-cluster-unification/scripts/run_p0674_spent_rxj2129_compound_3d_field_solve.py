#!/usr/bin/env python3
"""Solve the frozen spent RX J2129 scalar/compound tensor fields."""

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
from voidscreen.tensor_aqual_3d import solve_tensor_aqual_3d

DEFAULT_CONFIG = ROOT / "configs" / "p0674_spent_rxj2129_compound_3d_field_solve.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    expected_status = "frozen_before_any_P0674_field_or_deflection_score"
    if protocol.get("status") != expected_status:
        raise RuntimeError("P0674 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    scalar_reference = read_json(ROOT / protocol["scalar_reference_result"])
    failed_topology = read_json(ROOT / protocol["failed_topology_reference"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0673 parent no longer passes")

    map_path = ROOT / protocol["map_input"]
    with np.load(map_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        stars = data["stellar_volume_density_kg_m3"].astype(float)
        gas = data["gas_volume_density_kg_m3"].astype(float)
        boundary = data["simple_mond_boundary_m2_s2"].astype(float)
        map_a0 = float(data["a0_m_s2"])
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M
    candidate = protocol["candidate"]
    if not np.isclose(map_a0, float(candidate["a0_m_s2"]), rtol=0.0, atol=0.0):
        raise RuntimeError("P0670 and P0674 a0 differ")
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
    directions = activation.local.transport_direction
    total_density = stars + gas
    recomputed_sigma = float(np.sum(sigma * total_density) / np.sum(total_density))
    expected_sigma = float(parent["metrics"]["spent_RXJ2129_mass_weighted_sigma"])

    source = 4.0 * np.pi * G_SI * total_density
    settings = protocol["solver"]
    solver_kwargs = {
        "a0": map_a0,
        "residual_tolerance": float(settings["nonlinear_residual_tolerance"]),
        "maximum_nonlinear_iterations": int(settings["maximum_nonlinear_iterations"]),
        "maximum_linear_iterations": int(settings["maximum_linear_iterations"]),
        "linear_relative_tolerance": float(settings["linear_relative_tolerance"]),
        "damping": float(settings["picard_damping"]),
        "mu_floor": float(settings["mu_floor"]),
    }
    print("solving scalar AQUAL control", flush=True)
    scalar = solve_tensor_aqual_3d(
        source,
        spacing_m,
        boundary,
        np.zeros_like(sigma),
        *directions,
        **solver_kwargs,
    )
    print(
        f"scalar: converged={scalar.converged} residual={scalar.normalized_residual_rms:.6g} "
        f"iterations={scalar.nonlinear_iterations}",
        flush=True,
    )
    print("solving compound-path tensor AQUAL", flush=True)
    compound = solve_tensor_aqual_3d(
        source,
        spacing_m,
        boundary,
        sigma,
        *directions,
        **solver_kwargs,
    )
    print(
        f"compound: converged={compound.converged} "
        f"residual={compound.normalized_residual_rms:.6g} "
        f"iterations={compound.nonlinear_iterations}",
        flush=True,
    )

    scalar_deflection = photon_deflection_zero_slip(
        scalar.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    compound_deflection = photon_deflection_zero_slip(
        compound.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    x_values, y_values = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    radius = np.hypot(x_values, y_values)
    lower, upper = (
        float(value) for value in protocol["audit_region"]["strong_lens_radius_kpc"]
    )
    annulus = (radius >= lower) & (radius <= upper)
    scalar_magnitude = np.hypot(
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
    )
    compound_magnitude = np.hypot(
        compound_deflection.alpha_x_arcsec,
        compound_deflection.alpha_y_arcsec,
    )
    scalar_rms = vector_rms(
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
        annulus,
    )
    compound_rms = vector_rms(
        compound_deflection.alpha_x_arcsec,
        compound_deflection.alpha_y_arcsec,
        annulus,
    )
    response_ratio = compound_rms / max(scalar_rms, np.finfo(float).tiny)
    relative_difference = relative_vector_rms(
        compound_deflection.alpha_x_arcsec,
        compound_deflection.alpha_y_arcsec,
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
        annulus,
    )
    scalar_curl = normalized_deflection_curl(
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    compound_curl = normalized_deflection_curl(
        compound_deflection.alpha_x_arcsec,
        compound_deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    boundary_cells = boundary_mask(source.shape)
    boundary_scale = max(
        float(np.max(np.abs(boundary[boundary_cells]))),
        np.finfo(float).tiny,
    )
    scalar_boundary_mismatch = float(
        np.max(np.abs(scalar.potential[boundary_cells] - boundary[boundary_cells]))
    ) / boundary_scale
    compound_boundary_mismatch = float(
        np.max(np.abs(compound.potential[boundary_cells] - boundary[boundary_cells]))
    ) / boundary_scale
    boundary_mismatch = max(scalar_boundary_mismatch, compound_boundary_mismatch)
    finite = bool(
        np.all(np.isfinite(scalar.potential))
        and np.all(np.isfinite(compound.potential))
        and all(np.all(np.isfinite(component)) for component in scalar.acceleration)
        and all(np.all(np.isfinite(component)) for component in compound.acceleration)
        and np.all(np.isfinite(scalar_magnitude))
        and np.all(np.isfinite(compound_magnitude))
    )
    minimum_eigenvalue = float(compound.metadata["minimum_constitutive_eigenvalue"])
    scalar_median = float(np.median(scalar_magnitude[annulus]))
    compound_median = float(np.median(compound_magnitude[annulus]))

    gates = protocol["predeclared_progression_gates"]
    equations = protocol["field_equations"]
    failed_tensor_topology = failed_topology["topology"]["tensor_absolute_P0669"]
    gate_results = {
        "P0673_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0673_all_progression_gates_pass"]),
        "P0672_missing_multiplicity": int(
            failed_tensor_topology["missing_multiplicity_families"]
        )
        == int(gates["P0672_missing_multiplicity_families"]),
        "P0672_no_critical_curves": int(
            failed_tensor_topology["critical_curve_present_families"]
        )
        == int(gates["P0672_critical_curve_present_families"]),
        "coefficient_reproduction": abs(recomputed_sigma - expected_sigma)
        <= float(gates["recomputed_compound_mass_weighted_sigma_absolute_tolerance"]),
        "scalar_residual": scalar.normalized_residual_rms
        <= float(gates["scalar_normalized_residual_RMS_max"]),
        "compound_residual": compound.normalized_residual_rms
        <= float(gates["compound_normalized_residual_RMS_max"]),
        "scalar_convergence": scalar.converged
        is bool(gates["scalar_solver_converged"]),
        "compound_convergence": compound.converged
        is bool(gates["compound_solver_converged"]),
        "boundary": boundary_mismatch
        <= float(gates["boundary_maximum_relative_mismatch_max"]),
        "finite_fields": finite
        is bool(gates["all_potentials_accelerations_and_deflections_finite"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_compound_constitutive_eigenvalue_strictly_positive"]),
        "scalar_deflection_present": scalar_median
        >= float(gates["scalar_strong_lens_median_physical_deflection_arcsec_min"]),
        "compound_deflection_present": compound_median
        >= float(gates["compound_strong_lens_median_physical_deflection_arcsec_min"]),
        "deflection_ratio_lower": response_ratio
        >= float(gates["compound_to_scalar_strong_lens_deflection_RMS_ratio_min"]),
        "deflection_ratio_upper": response_ratio
        <= float(gates["compound_to_scalar_strong_lens_deflection_RMS_ratio_max"]),
        "compound_change_nonperturbative": relative_difference
        >= float(gates["compound_minus_scalar_strong_lens_relative_RMS_min"]),
        "compound_change_stable": relative_difference
        <= float(gates["compound_minus_scalar_strong_lens_relative_RMS_max"]),
        "scalar_curl": scalar_curl
        <= float(gates["scalar_normalized_deflection_curl_RMS_max"]),
        "compound_curl": compound_curl
        <= float(gates["compound_normalized_deflection_curl_RMS_max"]),
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
        "P0673_spent_compound_mass_weighted_sigma": expected_sigma,
        "scalar_normalized_residual_RMS": scalar.normalized_residual_rms,
        "compound_normalized_residual_RMS": compound.normalized_residual_rms,
        "scalar_nonlinear_iterations": scalar.nonlinear_iterations,
        "compound_nonlinear_iterations": compound.nonlinear_iterations,
        "boundary_maximum_relative_mismatch": boundary_mismatch,
        "minimum_compound_constitutive_eigenvalue": minimum_eigenvalue,
        "scalar_strong_lens_median_physical_deflection_arcsec": scalar_median,
        "compound_strong_lens_median_physical_deflection_arcsec": compound_median,
        "scalar_strong_lens_deflection_RMS_arcsec": scalar_rms,
        "compound_strong_lens_deflection_RMS_arcsec": compound_rms,
        "compound_to_scalar_strong_lens_deflection_RMS_ratio": response_ratio,
        "compound_minus_scalar_strong_lens_relative_RMS": relative_difference,
        "scalar_normalized_deflection_curl_RMS": scalar_curl,
        "compound_normalized_deflection_curl_RMS": compound_curl,
        "P0671_scalar_reference_residual_RMS": scalar_reference["metrics"][
            "scalar_normalized_residual_RMS"
        ],
    }

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        field_path,
        axis_kpc=axis_kpc,
        scalar_potential_m2_s2=scalar.potential,
        compound_potential_m2_s2=compound.potential,
        scalar_alpha_x_physical_arcsec=scalar_deflection.alpha_x_arcsec,
        scalar_alpha_y_physical_arcsec=scalar_deflection.alpha_y_arcsec,
        compound_alpha_x_physical_arcsec=compound_deflection.alpha_x_arcsec,
        compound_alpha_y_physical_arcsec=compound_deflection.alpha_y_arcsec,
        compound_sigma=sigma,
        transport_direction_x=directions[0],
        transport_direction_y=directions[1],
        transport_direction_z=directions[2],
    )
    report = {
        "report_version": "P0674-SPENT-RXJ2129-COMPOUND-3D-FIELD-SOLVE-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_raw_lens_topology_audit": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(
            ROOT / "src/voidscreen/compound_activation_3d.py"
        ),
        "solver_source_sha256": sha256(ROOT / "src/voidscreen/tensor_aqual_3d.py"),
        "map_sha256": sha256(map_path),
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
    images = (
        scalar_magnitude,
        compound_magnitude,
        np.hypot(
            compound_deflection.alpha_x_arcsec - scalar_deflection.alpha_x_arcsec,
            compound_deflection.alpha_y_arcsec - scalar_deflection.alpha_y_arcsec,
        ),
    )
    titles = (
        "scalar physical deflection",
        "compound physical deflection",
        "compound - scalar",
    )
    for axis, values, title in zip(axes, images, titles, strict=True):
        image = axis.imshow(values.T, origin="lower", extent=extent, cmap="viridis")
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75, label="arcsec")
    figure.tight_layout()
    figure.savefig(output / "p0674_compound_deflection_fields.png", dpi=180)
    plt.close(figure)

    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0674 spent RX J2129 compound 3D field solve

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Scalar/compound residual: **{scalar.normalized_residual_rms:.4g} / {compound.normalized_residual_rms:.4g}**.
- Scalar/compound strong-lens median physical deflection: **{scalar_median:.4g} / {compound_median:.4g} arcsec**.
- Compound/scalar deflection RMS ratio: **{response_ratio:.6g}**.
- Compound-minus-scalar relative RMS: **{relative_difference:.6g}**.
- Scalar/compound normalized curl: **{scalar_curl:.3g} / {compound_curl:.3g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Raw lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
