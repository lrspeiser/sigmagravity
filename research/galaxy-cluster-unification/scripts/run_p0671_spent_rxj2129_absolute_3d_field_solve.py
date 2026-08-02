#!/usr/bin/env python3
"""Solve frozen absolute scalar/tensor AQUAL and zero-slip deflection fields."""

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
from voidscreen.tensor_aqual_3d import solve_tensor_aqual_3d

G_SI = 6.67430e-11
DEFAULT_CONFIG = ROOT / "configs" / "p0671_spent_rxj2129_absolute_3d_field_solve.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def vector_rms(x_values: np.ndarray, y_values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x_values[mask] ** 2 + y_values[mask] ** 2)))


def relative_vector_rms(
    first_x: np.ndarray,
    first_y: np.ndarray,
    second_x: np.ndarray,
    second_y: np.ndarray,
    mask: np.ndarray,
) -> float:
    difference = vector_rms(first_x - second_x, first_y - second_y, mask)
    reference = vector_rms(second_x, second_y, mask)
    return difference / max(reference, np.finfo(float).tiny)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0671_field_or_deflection_score":
        raise RuntimeError("P0671 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0670 parent no longer passes")
    map_path = ROOT / protocol["map_input"]
    with np.load(map_path) as data:
        axis_kpc = data["axis_kpc"].astype(float)
        stars = data["stellar_volume_density_kg_m3"].astype(float)
        gas = data["gas_volume_density_kg_m3"].astype(float)
        sigma = data["sigma"].astype(float)
        directions = tuple(
            data[key].astype(float)
            for key in (
                "transport_direction_x",
                "transport_direction_y",
                "transport_direction_z",
            )
        )
        boundary = data["simple_mond_boundary_m2_s2"].astype(float)
        a0 = float(data["a0_m_s2"])
    spacing_m = float(axis_kpc[1] - axis_kpc[0]) * KPC_M
    source = 4.0 * np.pi * G_SI * (stars + gas)
    settings = protocol["solver"]
    solver_kwargs = {
        "a0": a0,
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
    print("solving amplitude-multipole tensor AQUAL", flush=True)
    tensor = solve_tensor_aqual_3d(
        source,
        spacing_m,
        boundary,
        sigma,
        *directions,
        **solver_kwargs,
    )
    print(
        f"tensor: converged={tensor.converged} residual={tensor.normalized_residual_rms:.6g} "
        f"iterations={tensor.nonlinear_iterations}",
        flush=True,
    )
    scalar_deflection = photon_deflection_zero_slip(
        scalar.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    tensor_deflection = photon_deflection_zero_slip(
        tensor.acceleration,
        spacing_m,
        distance_ratio=1.0,
    )
    x, y = np.meshgrid(axis_kpc, axis_kpc, indexing="ij")
    radius = np.hypot(x, y)
    lower, upper = (float(value) for value in protocol["audit_region"]["strong_lens_radius_kpc"])
    annulus = (radius >= lower) & (radius <= upper)
    scalar_magnitude = np.hypot(
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
    )
    tensor_magnitude = np.hypot(
        tensor_deflection.alpha_x_arcsec,
        tensor_deflection.alpha_y_arcsec,
    )
    scalar_rms = vector_rms(
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
        annulus,
    )
    tensor_rms = vector_rms(
        tensor_deflection.alpha_x_arcsec,
        tensor_deflection.alpha_y_arcsec,
        annulus,
    )
    tensor_scalar_ratio = tensor_rms / max(scalar_rms, np.finfo(float).tiny)
    relative_difference = relative_vector_rms(
        tensor_deflection.alpha_x_arcsec,
        tensor_deflection.alpha_y_arcsec,
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
        annulus,
    )
    scalar_curl = normalized_deflection_curl(
        scalar_deflection.alpha_x_arcsec,
        scalar_deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    tensor_curl = normalized_deflection_curl(
        tensor_deflection.alpha_x_arcsec,
        tensor_deflection.alpha_y_arcsec,
        float(axis_kpc[1] - axis_kpc[0]),
    )
    boundary_cells = boundary_mask(source.shape)
    scalar_boundary_mismatch = float(
        np.max(np.abs(scalar.potential[boundary_cells] - boundary[boundary_cells]))
    ) / max(float(np.max(np.abs(boundary[boundary_cells]))), np.finfo(float).tiny)
    tensor_boundary_mismatch = float(
        np.max(np.abs(tensor.potential[boundary_cells] - boundary[boundary_cells]))
    ) / max(float(np.max(np.abs(boundary[boundary_cells]))), np.finfo(float).tiny)
    boundary_mismatch = max(scalar_boundary_mismatch, tensor_boundary_mismatch)
    finite = bool(
        np.all(np.isfinite(scalar.potential))
        and np.all(np.isfinite(tensor.potential))
        and all(np.all(np.isfinite(component)) for component in scalar.acceleration)
        and all(np.all(np.isfinite(component)) for component in tensor.acceleration)
        and np.all(np.isfinite(scalar_magnitude))
        and np.all(np.isfinite(tensor_magnitude))
    )
    minimum_eigenvalue = float(tensor.metadata["minimum_constitutive_eigenvalue"])
    scalar_median = float(np.median(scalar_magnitude[annulus]))
    tensor_median = float(np.median(tensor_magnitude[annulus]))
    gates = protocol["predeclared_progression_gates"]
    equations = protocol["field_equations"]
    gate_results = {
        "P0670_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0670_all_progression_gates_pass"]),
        "scalar_residual": scalar.normalized_residual_rms
        <= gates["scalar_normalized_residual_RMS_max"],
        "tensor_residual": tensor.normalized_residual_rms
        <= gates["tensor_normalized_residual_RMS_max"],
        "scalar_convergence": scalar.converged is bool(gates["scalar_solver_converged"]),
        "tensor_convergence": tensor.converged is bool(gates["tensor_solver_converged"]),
        "boundary": boundary_mismatch
        <= gates["boundary_maximum_relative_mismatch_max"],
        "finite_fields": finite
        is bool(gates["all_potentials_accelerations_and_deflections_finite"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_tensor_constitutive_eigenvalue_strictly_positive"]),
        "scalar_deflection_present": scalar_median
        >= gates["scalar_strong_lens_median_physical_deflection_arcsec_min"],
        "tensor_deflection_present": tensor_median
        >= gates["tensor_strong_lens_median_physical_deflection_arcsec_min"],
        "deflection_ratio_lower": tensor_scalar_ratio
        >= gates["tensor_to_scalar_strong_lens_deflection_RMS_ratio_min"],
        "deflection_ratio_upper": tensor_scalar_ratio
        <= gates["tensor_to_scalar_strong_lens_deflection_RMS_ratio_max"],
        "tensor_change_nonnull": relative_difference
        >= gates["tensor_minus_scalar_strong_lens_relative_RMS_min"],
        "tensor_change_stable": relative_difference
        <= gates["tensor_minus_scalar_strong_lens_relative_RMS_max"],
        "scalar_curl": scalar_curl
        <= gates["scalar_normalized_deflection_curl_RMS_max"],
        "tensor_curl": tensor_curl
        <= gates["tensor_normalized_deflection_curl_RMS_max"],
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
        "scalar_normalized_residual_RMS": scalar.normalized_residual_rms,
        "tensor_normalized_residual_RMS": tensor.normalized_residual_rms,
        "scalar_nonlinear_iterations": scalar.nonlinear_iterations,
        "tensor_nonlinear_iterations": tensor.nonlinear_iterations,
        "boundary_maximum_relative_mismatch": boundary_mismatch,
        "minimum_tensor_constitutive_eigenvalue": minimum_eigenvalue,
        "scalar_strong_lens_median_physical_deflection_arcsec": scalar_median,
        "tensor_strong_lens_median_physical_deflection_arcsec": tensor_median,
        "scalar_strong_lens_deflection_RMS_arcsec": scalar_rms,
        "tensor_strong_lens_deflection_RMS_arcsec": tensor_rms,
        "tensor_to_scalar_strong_lens_deflection_RMS_ratio": tensor_scalar_ratio,
        "tensor_minus_scalar_strong_lens_relative_RMS": relative_difference,
        "scalar_normalized_deflection_curl_RMS": scalar_curl,
        "tensor_normalized_deflection_curl_RMS": tensor_curl,
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    field_path = output / protocol["outputs"]["field"]
    np.savez_compressed(
        field_path,
        axis_kpc=axis_kpc,
        scalar_potential_m2_s2=scalar.potential,
        tensor_potential_m2_s2=tensor.potential,
        scalar_alpha_x_physical_arcsec=scalar_deflection.alpha_x_arcsec,
        scalar_alpha_y_physical_arcsec=scalar_deflection.alpha_y_arcsec,
        tensor_alpha_x_physical_arcsec=tensor_deflection.alpha_x_arcsec,
        tensor_alpha_y_physical_arcsec=tensor_deflection.alpha_y_arcsec,
        tensor_sigma=sigma,
    )
    report = {
        "report_version": "P0671-SPENT-RXJ2129-ABSOLUTE-3D-FIELD-SOLVE-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_raw_lens_topology_audit": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
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
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    extent = [axis_kpc[0], axis_kpc[-1], axis_kpc[0], axis_kpc[-1]]
    images = (
        scalar_magnitude,
        tensor_magnitude,
        np.hypot(
            tensor_deflection.alpha_x_arcsec - scalar_deflection.alpha_x_arcsec,
            tensor_deflection.alpha_y_arcsec - scalar_deflection.alpha_y_arcsec,
        ),
    )
    titles = ("scalar physical deflection", "tensor physical deflection", "tensor - scalar")
    for axis, values, title in zip(axes, images, titles, strict=True):
        image = axis.imshow(values.T, origin="lower", extent=extent, cmap="viridis")
        axis.set(title=title, xlabel="x (kpc)", ylabel="y (kpc)")
        figure.colorbar(image, ax=axis, shrink=0.75, label="arcsec")
    figure.tight_layout()
    figure.savefig(output / "p0671_absolute_deflection_fields.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary = f"""# P0671 spent RX J2129 absolute 3D field solve

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Scalar/tensor residual: **{scalar.normalized_residual_rms:.4g} / {tensor.normalized_residual_rms:.4g}**.
- Scalar/tensor strong-lens median physical deflection: **{scalar_median:.4g} / {tensor_median:.4g} arcsec**.
- Tensor/scalar deflection RMS ratio: **{tensor_scalar_ratio:.6g}**.
- Tensor-minus-scalar relative RMS: **{relative_difference:.6g}**.
- Scalar/tensor normalized curl: **{scalar_curl:.3g} / {tensor_curl:.3g}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Raw lens score computed: **no**; sealed targets opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
