#!/usr/bin/env python3
"""Validate the three-dimensional tensor-AQUAL variational solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import sha256

from voidscreen.field_solvers import surface_density_to_volume
from voidscreen.tensor_aqual_3d import (
    constitutive_eigenvalues_3d,
    solve_tensor_aqual_3d,
    tensor_graph_laplacian_3d,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0665_tensor_aqual_3d_solver.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def constant_mu(values):
    return np.ones_like(values)


def relative_rms(first, second):
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2))) / max(
        float(np.sqrt(np.mean(np.asarray(second) ** 2))),
        np.finfo(float).tiny,
    )


def manufactured(cells: int, sigma_value: float):
    axis = np.linspace(-1.0, 1.0, int(cells))
    spacing = float(axis[1] - axis[0])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    exact = np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) * np.cos(0.5 * np.pi * z)
    wave2 = (0.5 * np.pi) ** 2
    source = -wave2 * (3.0 - float(sigma_value)) * exact
    sigma = np.full_like(exact, float(sigma_value))
    direction_x = np.ones_like(exact)
    direction_y = np.zeros_like(exact)
    direction_z = np.zeros_like(exact)
    solution = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(exact),
        sigma,
        direction_x,
        direction_y,
        direction_z,
        a0=1.0,
        mu_function=constant_mu,
    )
    return {
        "axis": axis,
        "spacing": spacing,
        "exact": exact,
        "source": source,
        "sigma": sigma,
        "directions": (direction_x, direction_y, direction_z),
        "solution": solution,
        "error": relative_rms(solution.potential, exact),
    }


def density_lift_error(protocol):
    analytic = protocol["analytic_tests"]
    cells = int(analytic["surface_density_lift_grid"])
    z_cells = int(analytic["surface_density_lift_z_cells"])
    axis = np.linspace(-2.0, 2.0, cells)
    z = np.linspace(-4.0, 4.0, z_cells)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    surface = np.exp(-0.5 * ((xx / 0.4) ** 2 + (yy / 0.7) ** 2))
    volume = surface_density_to_volume(
        surface,
        z,
        scale_height=float(analytic["surface_density_lift_scale_height"]),
    )
    reconstructed = np.sum(volume, axis=2) * float(z[1] - z[0])
    return relative_rms(reconstructed, surface)


def operator_symmetry_error(reference):
    operator = tensor_graph_laplacian_3d(
        np.ones_like(reference["sigma"]),
        reference["sigma"],
        *reference["directions"],
        reference["spacing"],
    )
    rng = np.random.default_rng(6605)
    first = rng.normal(size=operator.shape[0])
    second = rng.normal(size=operator.shape[0])
    left = float(first @ (operator @ second))
    right = float(second @ (operator @ first))
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0665_solver_score":
        raise RuntimeError("P0665 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0664 parent no longer passes")

    analytic = protocol["analytic_tests"]
    sigma_value = float(analytic["manufactured_sigma"])
    results = {
        int(cells): manufactured(int(cells), sigma_value)
        for cells in analytic["manufactured_grids"]
    }
    convergence = pd.DataFrame(
        [
            {
                "cells": cells,
                "spacing": result["spacing"],
                "relative_RMS_error": result["error"],
                "normalized_residual_RMS": result["solution"].normalized_residual_rms,
            }
            for cells, result in results.items()
        ]
    ).sort_values("cells")
    order = float(
        np.polyfit(
            np.log(convergence.spacing.to_numpy(float)),
            np.log(convergence.relative_RMS_error.to_numpy(float)),
            1,
        )[0]
    )
    reference = results[17]
    source = reference["source"]
    spacing = reference["spacing"]
    directions = reference["directions"]
    sigma = reference["sigma"]
    rotated = solve_tensor_aqual_3d(
        np.swapaxes(source, 0, 1),
        spacing,
        np.zeros_like(source),
        np.swapaxes(sigma, 0, 1),
        np.zeros_like(source),
        np.ones_like(source),
        np.zeros_like(source),
        a0=1.0,
        mu_function=constant_mu,
    )
    rotation_error = relative_rms(
        np.swapaxes(rotated.potential, 0, 1),
        reference["solution"].potential,
    )
    reversed_solution = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(source),
        sigma,
        *(-component for component in directions),
        a0=1.0,
        mu_function=constant_mu,
    )
    reversal_error = relative_rms(
        reversed_solution.potential,
        reference["solution"].potential,
    )
    zero_sigma = np.zeros_like(source)
    scalar_first = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(source),
        zero_sigma,
        *directions,
        a0=0.15,
    )
    scalar_second = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(source),
        zero_sigma,
        np.zeros_like(source),
        np.ones_like(source),
        np.zeros_like(source),
        a0=0.15,
    )
    scalar_difference = relative_rms(scalar_first.potential, scalar_second.potential)

    nonlinear_cells = int(analytic["nonlinear_grid"])
    nonlinear_axis = np.linspace(-2.0, 2.0, nonlinear_cells)
    nonlinear_spacing = float(nonlinear_axis[1] - nonlinear_axis[0])
    x, y, z = np.meshgrid(nonlinear_axis, nonlinear_axis, nonlinear_axis, indexing="ij")
    nonlinear_source = np.exp(
        -0.5 * ((x / 0.55) ** 2 + (y / 0.75) ** 2 + (z / 0.95) ** 2)
    )
    nonlinear_sigma = 0.2 * np.exp(-0.5 * (x * x + y * y + z * z) / 1.2**2)
    nonlinear_direction_x = np.cos(0.3 * y)
    nonlinear_direction_y = np.sin(0.3 * y)
    nonlinear_direction_z = np.zeros_like(x)
    solver = protocol["solver"]
    nonlinear = solve_tensor_aqual_3d(
        nonlinear_source,
        nonlinear_spacing,
        np.zeros_like(nonlinear_source),
        nonlinear_sigma,
        nonlinear_direction_x,
        nonlinear_direction_y,
        nonlinear_direction_z,
        a0=float(analytic["nonlinear_a0"]),
        residual_tolerance=float(solver["nonlinear_residual_tolerance"]),
        maximum_nonlinear_iterations=int(solver["maximum_nonlinear_iterations"]),
        maximum_linear_iterations=int(solver["maximum_linear_iterations"]),
        linear_relative_tolerance=float(solver["linear_relative_tolerance"]),
        damping=float(solver["picard_damping"]),
        mu_floor=float(solver["mu_floor"]),
    )
    minimum_eigenvalue, _, _ = constitutive_eigenvalues_3d(
        nonlinear.coefficient_mu,
        nonlinear.anisotropy_sigma,
    )
    lift_error = density_lift_error(protocol)
    symmetry_error = operator_symmetry_error(reference)
    gates = protocol["predeclared_progression_gates"]
    finest_error = float(convergence.iloc[-1].relative_RMS_error)
    definitions = protocol["definitions"]
    gate_results = {
        "P0664_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0664_all_progression_gates_pass"]),
        "positive_eigenvalue": bool(np.min(minimum_eigenvalue) > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "manufactured_accuracy": finest_error
        <= gates["manufactured_25_grid_relative_RMS_error_max"],
        "manufactured_order": order >= gates["manufactured_convergence_order_min"],
        "rotation_covariance": rotation_error
        <= gates["rotation_covariance_relative_RMS_error_max"],
        "direction_reversal": reversal_error
        <= gates["direction_reversal_relative_RMS_error_max"],
        "scalar_AQUAL_limit": scalar_difference
        <= gates["sigma_zero_relative_RMS_difference_from_scalar_graph_AQUAL_max"],
        "nonlinear_residual": nonlinear.normalized_residual_rms
        <= gates["nonlinear_normalized_residual_RMS_max"],
        "nonlinear_convergence": nonlinear.converged
        is bool(gates["nonlinear_solver_converged"]),
        "surface_density_lift": lift_error
        <= gates["surface_density_lift_column_mass_relative_error_max"],
        "operator_symmetry": symmetry_error
        <= gates["operator_symmetry_relative_error_max"],
        "no_new_constants": int(definitions["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(definitions["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "spent_lensing_untouched": not bool(gates["spent_lensing_outcomes_opened"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "manufactured_25_grid_relative_RMS_error": finest_error,
        "manufactured_convergence_order": order,
        "rotation_covariance_relative_RMS_error": rotation_error,
        "direction_reversal_relative_RMS_error": reversal_error,
        "sigma_zero_scalar_graph_AQUAL_relative_RMS_difference": scalar_difference,
        "nonlinear_normalized_residual_RMS": nonlinear.normalized_residual_rms,
        "nonlinear_iterations": nonlinear.nonlinear_iterations,
        "minimum_constitutive_eigenvalue": float(np.min(minimum_eigenvalue)),
        "surface_density_lift_column_mass_relative_error": lift_error,
        "operator_symmetry_relative_error": symmetry_error,
    }
    report = {
        "report_version": "P0665-TENSOR-AQUAL-3D-SOLVER-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_zero_slip_photon_deflection": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "solver_source_sha256": sha256(ROOT / "src/voidscreen/tensor_aqual_3d.py"),
        "coverage": {
            "manufactured_grids": len(results),
            "nonlinear_cases": 1,
            "density_lift_cases": 1,
            "new_universal_constants_after_P0659": int(
                definitions["new_universal_constants_after_P0659"]
            ),
            "per_object_gravity_parameters": int(definitions["per_object_gravity_parameters"]),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "spent_RXJ2129_lensing_outcomes_opened": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    convergence.to_csv(output / "manufactured_convergence.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].loglog(convergence.spacing, convergence.relative_RMS_error, "o-")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("grid spacing")
    axes[0].set_ylabel("relative RMS error")
    axes[0].set_title(f"Measured order {order:.3f}")
    center = nonlinear_cells // 2
    image = axes[1].imshow(nonlinear.potential[:, :, center], origin="lower", cmap="viridis")
    axes[1].set_title("Nonlinear central slice")
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.suptitle("P0665 three-dimensional tensor AQUAL")
    figure.tight_layout()
    figure.savefig(output / "p0665_tensor_aqual_3d.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0665 three-dimensional tensor-AQUAL solver

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Finest manufactured relative RMS error: **{finest_error:.3e}**; convergence order: **{order:.4f}**.
- Rotation/reversal errors: **{rotation_error:.3e} / {reversal_error:.3e}**.
- Nonlinear residual: **{nonlinear.normalized_residual_rms:.3e}** in **{nonlinear.nonlinear_iterations}** iterations.
- Surface-to-volume column reconstruction error: **{lift_error:.3e}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Spent and sealed lensing outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
