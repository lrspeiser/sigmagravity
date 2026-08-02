#!/usr/bin/env python3
"""Solve scalar and tensor AQUAL on registered maps with outcomes sealed."""

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

from run_p0660_exact_tensor_activation_audit import gaussian, manifest_sha256, sha256

from voidscreen.geometric_transport import (
    KPC_M,
    aperture_weighted_statistics,
    resample_surface_density,
    thin_sheet_newtonian_field,
)
from voidscreen.registered_tensor_field import (
    constant_mu,
    projected_source_from_newtonian_potential,
    solve_registered_tensor_field_pair,
)
from voidscreen.tensor_aqual import solve_projected_tensor_aqual

DEFAULT_CONFIG = ROOT / "configs" / "p0664_registered_tensor_field_solve.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_rms(first, second):
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2))) / max(
        float(np.sqrt(np.mean(np.asarray(second) ** 2))),
        np.finfo(float).tiny,
    )


def solver_kwargs(protocol):
    solver = protocol["solver"]
    definitions = protocol["definitions"]
    return {
        "a0_m_s2": float(definitions["a0_m_s2"]),
        "coherence_length_kpc": float(definitions["coherence_length_kpc"]),
        "coherence_power": float(definitions["coherence_power"]),
        "border_fraction": float(solver["analysis_border_fraction"]),
        "residual_tolerance": float(solver["nonlinear_residual_tolerance"]),
        "maximum_nonlinear_iterations": int(solver["maximum_nonlinear_iterations"]),
        "maximum_linear_iterations": int(solver["maximum_linear_iterations"]),
        "linear_relative_tolerance": float(solver["linear_relative_tolerance"]),
        "damping": float(solver["picard_damping"]),
        "mu_floor": float(solver["mu_floor"]),
    }


def constant_mu_recovery(protocol):
    axis = np.linspace(-4.0, 4.0, 33)
    cell = float(axis[1] - axis[0])
    stars = gaussian(axis, -0.4, 0.0, 0.45, 3.0e9)
    gas = gaussian(axis, 0.6, 0.0, 0.9, 7.0e9)
    field = thin_sheet_newtonian_field(stars + gas, cell)
    spacing = cell * KPC_M
    source = projected_source_from_newtonian_potential(field.potential_m2_s2, spacing)
    solution = solve_projected_tensor_aqual(
        source,
        spacing,
        field.potential_m2_s2,
        np.zeros_like(stars),
        np.ones_like(stars),
        np.zeros_like(stars),
        a0=float(protocol["definitions"]["a0_m_s2"]),
        mu_function=constant_mu,
        residual_tolerance=float(protocol["solver"]["nonlinear_residual_tolerance"]),
    )
    return relative_rms(solution.potential, field.potential_m2_s2)


def rotation_covariance(protocol):
    axis = np.linspace(-4.0, 4.0, 33)
    cell = float(axis[1] - axis[0])
    stars = gaussian(axis, -0.4, 0.0, 0.45, 3.0e9)
    gas = gaussian(axis, 0.6, 0.0, 0.9, 7.0e9)
    kwargs = solver_kwargs(protocol)
    first = solve_registered_tensor_field_pair(stars, gas, cell, **kwargs)
    rotated = solve_registered_tensor_field_pair(
        np.rot90(stars),
        np.rot90(gas),
        cell,
        **kwargs,
    )
    return abs(
        rotated.tensor_effect_relative_rms
        / max(first.tensor_effect_relative_rms, np.finfo(float).tiny)
        - 1.0
    )


def solve_case(case, domain, stars, gas, cell_kpc, protocol):
    pair = solve_registered_tensor_field_pair(
        stars,
        gas,
        cell_kpc,
        **solver_kwargs(protocol),
    )
    sigma_stats = aperture_weighted_statistics(
        pair.activation.sigma,
        np.asarray(stars) + np.asarray(gas),
        pair.activation.total_field.magnitude_m_s2,
        cell_kpc,
    )
    return {
        "case": case,
        "domain": domain,
        "cells": int(np.asarray(stars).shape[0]),
        "cell_kpc": float(cell_kpc),
        "sigma_weighted_mean": sigma_stats["weighted_mean"],
        "tensor_effect_relative_RMS": pair.tensor_effect_relative_rms,
        "scalar_newtonian_enhancement_RMS": pair.scalar_newtonian_enhancement_rms,
        "scalar_normalized_residual_RMS": pair.scalar.normalized_residual_rms,
        "tensor_normalized_residual_RMS": pair.tensor.normalized_residual_rms,
        "scalar_converged": pair.scalar.converged,
        "tensor_converged": pair.tensor.converged,
        "scalar_iterations": pair.scalar.nonlinear_iterations,
        "tensor_iterations": pair.tensor.nonlinear_iterations,
        "minimum_constitutive_eigenvalue": pair.tensor.metadata[
            "minimum_constitutive_eigenvalue"
        ],
        "tensor_normalized_curl_RMS": pair.tensor_normalized_curl_rms,
    }


def registered_solves(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    paths = []
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
        paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            stars = data["stars"].astype(float) * float(inputs["galaxy_stellar_scale"])
            gas = data["gas"].astype(float)
        rows.append(
            solve_case(
                path.stem,
                "registered_galaxy_baryons_only",
                stars,
                gas,
                float(axis[1] - axis[0]),
                protocol,
            )
        )
    target = int(inputs["cluster_cells"])
    for path in sorted((ROOT / inputs["clusters"]).glob("*.npz")):
        paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            stars = data[inputs["cluster_stellar_map"]].astype(float)
            gas = data[inputs["cluster_gas_map"]].astype(float)
        rows.append(
            solve_case(
                path.stem.replace("_baryons", ""),
                "registered_cluster_baryons_only",
                resample_surface_density(stars, target),
                resample_surface_density(gas, target),
                float((axis[-1] - axis[0]) / (target - 1)),
                protocol,
            )
        )
    return pd.DataFrame(rows), paths


def evaluate(protocol, parent, scores, recovery_error, rotation_error):
    gates = protocol["predeclared_progression_gates"]
    galaxy = scores[scores.domain.eq("registered_galaxy_baryons_only")]
    cluster = scores[scores.domain.eq("registered_cluster_baryons_only")]
    galaxy_median = float(galaxy.tensor_effect_relative_RMS.median())
    cluster_median = float(cluster.tensor_effect_relative_RMS.median())
    effect_ratio = cluster_median / max(galaxy_median, np.finfo(float).tiny)
    maximum_residual = float(
        scores[["scalar_normalized_residual_RMS", "tensor_normalized_residual_RMS"]]
        .to_numpy(float)
        .max()
    )
    minimum_eigenvalue = float(scores.minimum_constitutive_eigenvalue.min())
    maximum_curl = float(scores.tensor_normalized_curl_RMS.max())
    definitions = protocol["definitions"]
    gate_results = {
        "P0663_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0663_all_progression_gates_pass"]),
        "galaxy_coverage": int(galaxy.case.nunique()) == int(gates["registered_galaxy_count"]),
        "cluster_coverage": int(cluster.case.nunique()) == int(gates["registered_cluster_count"]),
        "constant_mu_recovery": recovery_error
        <= gates["constant_mu_sigma_zero_newtonian_recovery_relative_RMS_max"],
        "rotation_covariance": rotation_error
        <= gates["rotation_covariance_tensor_effect_relative_error_max"],
        "all_solvers_converged": bool(
            scores.scalar_converged.all() and scores.tensor_converged.all()
        )
        is bool(gates["all_scalar_and_tensor_solvers_converged"]),
        "nonlinear_residual": maximum_residual
        <= gates["maximum_nonlinear_normalized_residual_RMS"],
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "conservative_acceleration": maximum_curl
        <= gates["maximum_normalized_acceleration_curl_RMS"],
        "galaxy_median_preserved": galaxy_median
        <= gates["registered_galaxy_median_tensor_effect_max"],
        "galaxy_maximum_preserved": float(galaxy.tensor_effect_relative_RMS.max())
        <= gates["registered_galaxy_maximum_tensor_effect_max"],
        "cluster_median_response": cluster_median
        >= gates["registered_cluster_median_tensor_effect_min"],
        "cluster_minimum_response": float(cluster.tensor_effect_relative_RMS.min())
        >= gates["registered_cluster_minimum_tensor_effect_min"],
        "domain_effect_ratio": effect_ratio
        >= gates["cluster_to_galaxy_median_tensor_effect_ratio_min"],
        "no_new_constants": int(definitions["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(definitions["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    metrics = {
        "constant_mu_sigma_zero_newtonian_recovery_relative_RMS": recovery_error,
        "rotation_covariance_tensor_effect_relative_error": rotation_error,
        "maximum_nonlinear_normalized_residual_RMS": maximum_residual,
        "minimum_constitutive_eigenvalue": minimum_eigenvalue,
        "maximum_normalized_acceleration_curl_RMS": maximum_curl,
        "registered_galaxy_median_tensor_effect": galaxy_median,
        "registered_galaxy_maximum_tensor_effect": float(
            galaxy.tensor_effect_relative_RMS.max()
        ),
        "registered_cluster_median_tensor_effect": cluster_median,
        "registered_cluster_minimum_tensor_effect": float(
            cluster.tensor_effect_relative_RMS.min()
        ),
        "cluster_to_galaxy_median_tensor_effect_ratio": effect_ratio,
        "registered_galaxy_median_scalar_newtonian_enhancement": float(
            galaxy.scalar_newtonian_enhancement_RMS.median()
        ),
        "registered_cluster_median_scalar_newtonian_enhancement": float(
            cluster.scalar_newtonian_enhancement_RMS.median()
        ),
    }
    return gate_results, metrics


def make_figure(scores, output):
    frame = scores.sort_values(["domain", "tensor_effect_relative_RMS"]).copy()
    frame["map_type"] = frame.domain.str.contains("cluster").map(
        {False: "galaxy", True: "cluster"}
    )
    colors = frame.map_type.map({"galaxy": "#3274a1", "cluster": "#d95f02"})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(frame.case, frame.tensor_effect_relative_RMS, color=colors)
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_ylabel("tensor vs scalar acceleration RMS")
    axes[0].set_title("Domain-selective field effect")
    axes[1].bar(frame.case, frame.scalar_newtonian_enhancement_RMS, color=colors)
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    axes[1].set_ylabel("scalar AQUAL / Newtonian RMS")
    axes[1].set_title("Scalar AQUAL baseline")
    axes[2].scatter(
        frame.sigma_weighted_mean,
        frame.tensor_effect_relative_RMS,
        c=colors,
    )
    axes[2].set_xlabel("weighted tensor sigma")
    axes[2].set_ylabel("field effect")
    axes[2].set_title("Coefficient to solved response")
    figure.suptitle("P0664 registered tensor field solves")
    figure.tight_layout()
    figure.savefig(output / "p0664_registered_tensor_fields.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0664_field_score":
        raise RuntimeError("P0664 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0663 parent no longer passes")
    recovery_error = constant_mu_recovery(protocol)
    rotation_error = rotation_covariance(protocol)
    scores, map_paths = registered_solves(protocol)
    gate_results, metrics = evaluate(
        protocol,
        parent,
        scores,
        recovery_error,
        rotation_error,
    )
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0664-REGISTERED-TENSOR-FIELD-SOLVE-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_lensing_topology": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "field_source_sha256": sha256(ROOT / "src/voidscreen/registered_tensor_field.py"),
        "registered_map_manifest_sha256": manifest_sha256(map_paths),
        "coverage": {
            "registered_galaxies": int(
                scores[scores.domain.eq("registered_galaxy_baryons_only")].case.nunique()
            ),
            "registered_clusters": int(
                scores[scores.domain.eq("registered_cluster_baryons_only")].case.nunique()
            ),
            "new_universal_constants_after_P0659": int(
                protocol["definitions"]["new_universal_constants_after_P0659"]
            ),
            "per_object_gravity_parameters": int(
                protocol["definitions"]["per_object_gravity_parameters"]
            ),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    scores.to_csv(output / "registered_field_scores.csv", index=False)
    make_figure(scores, output)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0664 registered tensor field solve

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Galaxy median/max tensor field effect: **{metrics['registered_galaxy_median_tensor_effect']:.3%} / {metrics['registered_galaxy_maximum_tensor_effect']:.3%}**.
- Cluster median/min tensor field effect: **{metrics['registered_cluster_median_tensor_effect']:.3%} / {metrics['registered_cluster_minimum_tensor_effect']:.3%}**.
- Cluster/galaxy median field-effect ratio: **{metrics['cluster_to_galaxy_median_tensor_effect_ratio']:.4g}x**.
- Maximum nonlinear residual: **{metrics['maximum_nonlinear_normalized_residual_RMS']:.3e}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633 velocities and P0640 lensing constraints opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
