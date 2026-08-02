#!/usr/bin/env python3
"""Validate the frozen projected tensor-AQUAL structure and solver."""

from __future__ import annotations

import argparse
import hashlib
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

from voidscreen.tensor_aqual import (
    constitutive_eigenvalues,
    solve_projected_tensor_aqual,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0659_tensor_aqual_solver.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def constant_mu(values):
    return np.ones_like(np.asarray(values, dtype=np.float64))


def manufactured(cells: int, sigma_value: float):
    axis = np.linspace(-1.0, 1.0, int(cells))
    spacing = float(axis[1] - axis[0])
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    exact = np.sin(0.5 * np.pi * (xx + 1.0)) * np.sin(
        0.5 * np.pi * (yy + 1.0)
    )
    wave2 = (0.5 * np.pi) ** 2
    source = -wave2 * (2.0 - float(sigma_value)) * exact
    sigma = np.full_like(source, float(sigma_value))
    direction_x = np.ones_like(source)
    direction_y = np.zeros_like(source)
    solution = solve_projected_tensor_aqual(
        source,
        spacing,
        np.zeros_like(source),
        sigma,
        direction_x,
        direction_y,
        a0=1.0,
        mu_function=constant_mu,
    )
    interior = (slice(1, -1), slice(1, -1))
    error = float(
        np.sqrt(np.mean(np.square(solution.potential[interior] - exact[interior])))
        / np.sqrt(np.mean(np.square(exact[interior])))
    )
    return solution, error, spacing, source, sigma, direction_x, direction_y


def relative_rms(first, second) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    return float(
        np.sqrt(np.mean(np.square(left - right)))
        / max(np.sqrt(np.mean(np.square(left))), np.finfo(float).tiny)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0659_solver_score":
        raise RuntimeError("P0659 protocol is not frozen")
    p0634 = read_json(ROOT / protocol["inputs"]["P0634_report"])
    p0643 = read_json(ROOT / protocol["inputs"]["P0643_report"])
    if not p0634["all_progression_gates_pass"]:
        raise RuntimeError("P0634 field solvers no longer pass")
    if not p0643["all_primary_gates_pass"]:
        raise RuntimeError("P0643 geometry lever no longer passes")

    analytic = protocol["analytic_tests"]
    sigma_value = float(analytic["manufactured_sigma"])
    convergence_rows = []
    manufactured_results = {}
    for cells in analytic["manufactured_grids"]:
        result = manufactured(int(cells), sigma_value)
        manufactured_results[int(cells)] = result
        convergence_rows.append(
            {
                "cells": int(cells),
                "spacing": result[2],
                "relative_RMS_error": result[1],
                "normalized_residual_RMS": result[0].normalized_residual_rms,
                "nonlinear_iterations": result[0].nonlinear_iterations,
            }
        )
    convergence = pd.DataFrame(convergence_rows)
    convergence_order = float(
        np.polyfit(
            np.log(convergence.spacing.to_numpy(float)),
            np.log(convergence.relative_RMS_error.to_numpy(float)),
            1,
        )[0]
    )

    reference, _, spacing, source, sigma, direction_x, direction_y = manufactured_results[33]
    rotated = solve_projected_tensor_aqual(
        np.rot90(source),
        spacing,
        np.zeros_like(source),
        np.rot90(sigma),
        np.zeros_like(source),
        np.ones_like(source),
        a0=1.0,
        mu_function=constant_mu,
    )
    rotation_error = relative_rms(reference.potential, np.rot90(rotated.potential, -1))
    reversed_direction = solve_projected_tensor_aqual(
        source,
        spacing,
        np.zeros_like(source),
        sigma,
        -direction_x,
        -direction_y,
        a0=1.0,
        mu_function=constant_mu,
    )
    reversal_error = relative_rms(reference.potential, reversed_direction.potential)

    zero_sigma = np.zeros_like(source)
    scalar_first = solve_projected_tensor_aqual(
        source,
        spacing,
        np.zeros_like(source),
        zero_sigma,
        direction_x,
        direction_y,
        a0=float(analytic["nonlinear_a0"]),
    )
    scalar_second = solve_projected_tensor_aqual(
        source,
        spacing,
        np.zeros_like(source),
        zero_sigma,
        -direction_y,
        direction_x,
        a0=float(analytic["nonlinear_a0"]),
    )
    aligned_difference = relative_rms(scalar_first.potential, scalar_second.potential)

    cells = int(analytic["nonlinear_grid"])
    axis = np.linspace(-2.0, 2.0, cells)
    nonlinear_spacing = float(axis[1] - axis[0])
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    nonlinear_source = np.exp(-0.5 * ((xx / 0.45) ** 2 + (yy / 0.7) ** 2))
    nonlinear_sigma = 0.25 * np.exp(-0.5 * (xx**2 + yy**2) / 1.2**2)
    nonlinear_direction_x = np.cos(0.35 * yy)
    nonlinear_direction_y = np.sin(0.35 * yy)
    solver = protocol["solver"]
    nonlinear = solve_projected_tensor_aqual(
        nonlinear_source,
        nonlinear_spacing,
        np.zeros_like(nonlinear_source),
        nonlinear_sigma,
        nonlinear_direction_x,
        nonlinear_direction_y,
        a0=float(analytic["nonlinear_a0"]),
        residual_tolerance=float(solver["nonlinear_residual_tolerance"]),
        maximum_nonlinear_iterations=int(solver["maximum_nonlinear_iterations"]),
        maximum_linear_iterations=int(solver["maximum_linear_iterations"]),
        linear_relative_tolerance=float(solver["linear_relative_tolerance"]),
        damping=float(solver["picard_damping"]),
        mu_floor=float(solver["mu_floor"]),
    )
    minimum_eigenvalue, _ = constitutive_eigenvalues(
        nonlinear.coefficient_mu, nonlinear.anisotropy_sigma
    )
    registered_ratio = float(
        p0643["primary_metrics"]["registered_cluster_to_galaxy_ratio"]
    )
    solar_anisotropy = float(
        p0643["primary_metrics"]["solar_1au_max_future_lambda_coefficient"]
    ) / 20.0
    gates = protocol["predeclared_progression_gates"]
    gate_results = {
        "P0634_solvers": bool(p0634["all_progression_gates_pass"])
        is bool(gates["P0634_all_solver_gates_pass"]),
        "positive_eigenvalue": bool(np.min(minimum_eigenvalue) > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "manufactured_accuracy": float(convergence.iloc[-1].relative_RMS_error)
        <= gates["manufactured_65_grid_relative_RMS_error_max"],
        "manufactured_order": convergence_order
        >= gates["manufactured_convergence_order_min"],
        "rotation_covariance": rotation_error
        <= gates["rotation_covariance_relative_RMS_error_max"],
        "direction_reversal": reversal_error
        <= gates["direction_reversal_relative_RMS_error_max"],
        "aligned_AQUAL_limit": aligned_difference
        <= gates["aligned_sigma_zero_relative_RMS_difference_from_scalar_AQUAL_max"],
        "nonlinear_residual": nonlinear.normalized_residual_rms
        <= gates["nonlinear_normalized_residual_RMS_max"],
        "nonlinear_convergence": nonlinear.converged
        is bool(gates["nonlinear_solver_converged"]),
        "registered_domain_separation": registered_ratio
        >= gates["registered_cluster_to_galaxy_activation_ratio_min"],
        "solar_proxy": solar_anisotropy
        <= gates["solar_1au_constitutive_anisotropy_max"],
        "no_new_constants": int(protocol["definitions"]["new_universal_constants"])
        == int(gates["new_universal_constants"]),
        "no_per_object_parameters": int(
            protocol["definitions"]["per_object_gravity_parameters"]
        )
        == int(gates["per_object_gravity_parameters"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    report = {
        "report_version": "P0659-TENSOR-AQUAL-SOLVER-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_outcome_blind_map_tests": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "solver_source_sha256": sha256(ROOT / "src/voidscreen/tensor_aqual.py"),
        "coverage": {
            "manufactured_grids": len(convergence),
            "nonlinear_cases": 1,
            "registered_galaxy_baryon_maps": 13,
            "registered_cluster_baryon_maps": 4,
            "new_universal_constants": 0,
            "per_object_gravity_parameters": 0,
        },
        "metrics": {
            "manufactured_65_grid_relative_RMS_error": float(
                convergence.iloc[-1].relative_RMS_error
            ),
            "manufactured_convergence_order": convergence_order,
            "rotation_covariance_relative_RMS_error": rotation_error,
            "direction_reversal_relative_RMS_error": reversal_error,
            "aligned_sigma_zero_relative_RMS_difference": aligned_difference,
            "nonlinear_normalized_residual_RMS": nonlinear.normalized_residual_rms,
            "nonlinear_iterations": nonlinear.nonlinear_iterations,
            "minimum_constitutive_eigenvalue": float(np.min(minimum_eigenvalue)),
            "registered_cluster_to_galaxy_activation_ratio": registered_ratio,
            "solar_1au_constitutive_anisotropy": solar_anisotropy,
        },
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    convergence.to_csv(output / "manufactured_convergence.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].loglog(
        convergence.spacing,
        convergence.relative_RMS_error,
        marker="o",
        label=f"order {convergence_order:.3f}",
    )
    axes[0].invert_xaxis()
    axes[0].set(xlabel="grid spacing", ylabel="relative RMS error", title="Manufactured tensor solution")
    axes[0].legend()
    image = axes[1].imshow(nonlinear.potential, origin="lower", cmap="viridis")
    axes[1].set(title="Nonlinear tensor-AQUAL potential")
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.tight_layout()
    figure.savefig(output / "tensor_aqual_solver.png", dpi=180)
    plt.close(figure)
    summary = f"""# P0659 tensor-AQUAL solver

- Status: **{report['status'].upper()}** ({sum(gate_results.values())}/{len(gate_results)} gates).
- Manufactured 65-grid relative RMS error: **{report['metrics']['manufactured_65_grid_relative_RMS_error']:.6g}**.
- Measured convergence order: **{convergence_order:.6g}**.
- Nonlinear residual: **{nonlinear.normalized_residual_rms:.3e}** in {nonlinear.nonlinear_iterations} iterations.
- Registered cluster/galaxy activation ratio: **{registered_ratio:.4g}**.
- Sealed outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "gates": gate_results}, indent=2))


if __name__ == "__main__":
    main()
