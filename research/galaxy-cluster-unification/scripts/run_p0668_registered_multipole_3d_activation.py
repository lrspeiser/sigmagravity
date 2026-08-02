#!/usr/bin/env python3
"""Evaluate the multipole-gated 3D coefficient on all registered maps."""

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

from run_p0660_exact_tensor_activation_audit import manifest_sha256, sha256

from voidscreen.metric_lensing_3d import (
    KPC_M,
    M_SUN_KG,
    lift_surface_density_msun_kpc2_to_si_volume,
)
from voidscreen.multipole_activation_3d import exact_multipole_gated_activation_3d
from voidscreen.observational_resampling import common_resolution_surface_density

DEFAULT_CONFIG = ROOT / "configs" / "p0668_registered_multipole_3d_activation.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def common_surface(surface, target_cells):
    values = np.asarray(surface, dtype=float)
    if values.shape[0] == int(target_cells):
        return values.copy()
    return common_resolution_surface_density(values, int(target_cells)).coarse


def mass_weighted(values, density):
    return float(np.sum(np.asarray(values) * density) / np.sum(density))


def score_case(case, domain, scenario, stars, gas, axis, protocol):
    cells = int(protocol["map_inputs"]["common_grid_cells"])
    star_surface = common_surface(stars, cells)
    gas_surface = common_surface(gas, cells)
    cell_kpc = float((axis[-1] - axis[0]) / (cells - 1))
    z_kpc = np.linspace(float(axis[0]), float(axis[-1]), cells)
    star_volume, star_scale = lift_surface_density_msun_kpc2_to_si_volume(
        star_surface,
        z_kpc,
        cell_kpc=cell_kpc,
    )
    gas_volume, gas_scale = lift_surface_density_msun_kpc2_to_si_volume(
        gas_surface,
        z_kpc,
        cell_kpc=cell_kpc,
    )
    dz_m = float(z_kpc[1] - z_kpc[0]) * KPC_M
    star_expected = star_surface * M_SUN_KG / KPC_M**2
    gas_expected = gas_surface * M_SUN_KG / KPC_M**2
    star_reconstructed = np.sum(star_volume, axis=2) * dz_m
    gas_reconstructed = np.sum(gas_volume, axis=2) * dz_m

    def relative_rms(first, second):
        return float(np.sqrt(np.mean((first - second) ** 2))) / max(
            float(np.sqrt(np.mean(second**2))),
            np.finfo(float).tiny,
        )

    mass_error = max(
        relative_rms(star_reconstructed, star_expected),
        relative_rms(gas_reconstructed, gas_expected),
    )
    candidate = protocol["candidate"]
    activation = exact_multipole_gated_activation_3d(
        star_volume,
        gas_volume,
        cell_kpc * KPC_M,
        a0=float(candidate["a0_m_s2"]),
        coherence_length=float(candidate["coherence_length_kpc"]) * KPC_M,
        coherence_power=float(candidate["coherence_power"]),
    )
    total = star_volume + gas_volume
    return {
        "case": case,
        "domain": domain,
        "scenario": scenario,
        "cells": cells,
        "cell_kpc": cell_kpc,
        "stellar_scale_height_kpc": star_scale,
        "gas_scale_height_kpc": gas_scale,
        "multipole_gate": activation.multipole.gate,
        "dipole_squared": activation.multipole.dipole_squared,
        "quadrupole_squared": activation.multipole.quadrupole_squared,
        "local_sigma_mass_weighted_mean": mass_weighted(activation.local.sigma, total),
        "final_sigma_mass_weighted_mean": mass_weighted(activation.sigma, total),
        "transverse_mismatch_mass_weighted_mean": mass_weighted(
            activation.local.transverse_mismatch,
            total,
        ),
        "survival_mass_weighted_mean": mass_weighted(activation.local.survival, total),
        "trace_length_mass_weighted_kpc": mass_weighted(
            activation.local.trace_length,
            total,
        )
        / KPC_M,
        "minimum_constitutive_eigenvalue_proxy": float(
            np.min(activation.minimum_eigenvalue_proxy)
        ),
        "sigma_global_minimum": float(np.min(activation.sigma)),
        "sigma_global_maximum": float(np.max(activation.sigma)),
        "component_mass_relative_error": mass_error,
        "all_coefficients_finite": bool(
            np.all(np.isfinite(activation.sigma))
            and np.all(np.isfinite(activation.minimum_eigenvalue_proxy))
        ),
    }


def registered_scores(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    paths = []
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
        paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            nominal_stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        for scenario, scale in inputs["galaxy_stellar_scale_sensitivity"].items():
            rows.append(
                score_case(
                    path.stem,
                    "registered_galaxy_baryons_only",
                    scenario,
                    nominal_stars * float(scale),
                    gas,
                    axis,
                    protocol,
                )
            )
    for path in sorted((ROOT / inputs["clusters"]).glob("*.npz")):
        paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            maps = {
                scenario: (data[keys[0]].astype(float), data[keys[1]].astype(float))
                for scenario, keys in inputs["cluster_sensitivity_maps"].items()
            }
        for scenario, (stars, gas) in maps.items():
            rows.append(
                score_case(
                    path.stem.replace("_baryons", ""),
                    "registered_cluster_baryons_only",
                    scenario,
                    stars,
                    gas,
                    axis,
                    protocol,
                )
            )
    return pd.DataFrame(rows), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0668_map_score":
        raise RuntimeError("P0668 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0667 parent no longer passes")
    scores, map_paths = registered_scores(protocol)
    summary = (
        scores.groupby(["domain", "scenario"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_sigma=("final_sigma_mass_weighted_mean", "median"),
            minimum_sigma=("final_sigma_mass_weighted_mean", "min"),
            maximum_sigma=("final_sigma_mass_weighted_mean", "max"),
            median_multipole_gate=("multipole_gate", "median"),
            median_trace_length_kpc=("trace_length_mass_weighted_kpc", "median"),
        )
    )
    nominal = summary[summary.scenario.eq("nominal")].set_index("domain")
    galaxy_sigma = float(nominal.loc["registered_galaxy_baryons_only", "median_sigma"])
    cluster_sigma = float(nominal.loc["registered_cluster_baryons_only", "median_sigma"])
    sigma_ratio = cluster_sigma / max(galaxy_sigma, np.finfo(float).tiny)
    galaxy_gate = float(
        nominal.loc["registered_galaxy_baryons_only", "median_multipole_gate"]
    )
    cluster_gate = float(
        nominal.loc["registered_cluster_baryons_only", "median_multipole_gate"]
    )
    gate_ratio = cluster_gate / max(galaxy_gate, np.finfo(float).tiny)
    sensitivity_ratios = {}
    for scenario in protocol["map_inputs"]["galaxy_stellar_scale_sensitivity"]:
        block = summary[summary.scenario.eq(scenario)].set_index("domain")
        sensitivity_ratios[scenario] = float(
            block.loc["registered_cluster_baryons_only", "median_sigma"]
            / max(
                float(block.loc["registered_galaxy_baryons_only", "median_sigma"]),
                np.finfo(float).tiny,
            )
        )
    galaxy_count = int(
        scores[scores.domain.eq("registered_galaxy_baryons_only")].case.nunique()
    )
    cluster_count = int(
        scores[scores.domain.eq("registered_cluster_baryons_only")].case.nunique()
    )
    maximum_mass_error = float(scores.component_mass_relative_error.max())
    scale_heights_valid = bool(
        np.all(np.isfinite(scores.stellar_scale_height_kpc))
        and np.all(np.isfinite(scores.gas_scale_height_kpc))
        and np.all(scores.stellar_scale_height_kpc > 0.0)
        and np.all(scores.gas_scale_height_kpc > 0.0)
    )
    bounded = bool(
        scores.all_coefficients_finite.all()
        and scores.sigma_global_minimum.min() >= 0.0
        and scores.sigma_global_maximum.max() <= 1.0
    )
    minimum_eigenvalue = float(scores.minimum_constitutive_eigenvalue_proxy.min())
    gates = protocol["predeclared_progression_gates"]
    candidate = protocol["candidate"]
    gate_results = {
        "P0667_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0667_all_progression_gates_pass"]),
        "galaxy_coverage": galaxy_count == int(gates["registered_galaxy_count"]),
        "cluster_coverage": cluster_count == int(gates["registered_cluster_count"]),
        "surface_to_volume_mass": maximum_mass_error
        <= gates["surface_to_volume_maximum_component_mass_relative_error_max"],
        "scale_heights": scale_heights_valid
        is bool(gates["all_scale_heights_finite_positive"]),
        "bounded_sigma": bounded is bool(gates["sigma_finite_and_in_closed_unit_interval"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_constitutive_eigenvalue_strictly_positive"]),
        "galaxy_channel_small": galaxy_sigma
        <= gates["registered_galaxy_nominal_median_sigma_max"],
        "cluster_channel_present": cluster_sigma
        >= gates["registered_cluster_nominal_median_sigma_min"],
        "sigma_domain_separation": sigma_ratio
        >= gates["registered_cluster_to_galaxy_nominal_median_sigma_ratio_min"],
        "mass_sensitivity": min(sensitivity_ratios.values())
        >= gates["cluster_to_galaxy_sigma_ratio_min_in_all_mass_sensitivities"],
        "multipole_domain_separation": gate_ratio
        >= gates["registered_cluster_to_galaxy_nominal_median_multipole_gate_ratio_min"],
        "no_new_constants": int(candidate["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(candidate["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "spent_lensing_untouched": not bool(gates["spent_lensing_outcomes_opened"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics = {
        "registered_galaxy_nominal_median_sigma": galaxy_sigma,
        "registered_cluster_nominal_median_sigma": cluster_sigma,
        "registered_cluster_to_galaxy_nominal_median_sigma_ratio": sigma_ratio,
        "mass_sensitivity_cluster_to_galaxy_sigma_ratios": sensitivity_ratios,
        "registered_galaxy_nominal_median_multipole_gate": galaxy_gate,
        "registered_cluster_nominal_median_multipole_gate": cluster_gate,
        "registered_cluster_to_galaxy_nominal_median_multipole_gate_ratio": gate_ratio,
        "maximum_component_mass_relative_error": maximum_mass_error,
        "minimum_constitutive_eigenvalue_proxy": minimum_eigenvalue,
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0668-REGISTERED-MULTIPOLE-3D-ACTIVATION-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_RXJ2129_3D_map_build": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(ROOT / "src/voidscreen/multipole_activation_3d.py"),
        "registered_map_manifest_sha256": manifest_sha256(map_paths),
        "coverage": {
            "registered_galaxies": galaxy_count,
            "registered_clusters": cluster_count,
            "mass_scenarios": int(scores.scenario.nunique()),
            "new_universal_constants_after_P0659": int(
                candidate["new_universal_constants_after_P0659"]
            ),
            "per_object_gravity_parameters": int(candidate["per_object_gravity_parameters"]),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "domain_summary": summary.to_dict(orient="records"),
        "spent_RXJ2129_lensing_outcomes_opened": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    scores.to_csv(output / "registered_multipole_3d_scores.csv", index=False)
    summary.to_csv(output / "domain_summary.csv", index=False)
    nominal_scores = scores[scores.scenario.eq("nominal")].sort_values(
        ["domain", "final_sigma_mass_weighted_mean"]
    )
    colors = nominal_scores.domain.str.contains("cluster").map(
        {False: "#3274a1", True: "#d95f02"}
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(nominal_scores.case, nominal_scores.multipole_gate, color=colors)
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_ylabel("multipole gate")
    axes[0].set_title("Registered baryonic multipoles")
    axes[1].bar(
        nominal_scores.case,
        nominal_scores.final_sigma_mass_weighted_mean,
        color=colors,
    )
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    axes[1].set_ylabel("mass-weighted final sigma")
    axes[1].set_title("Multipole-gated 3D activation")
    figure.suptitle("P0668 registered multipole 3D activation")
    figure.tight_layout()
    figure.savefig(output / "p0668_registered_multipole_3d.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0668 registered multipole 3D activation

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Nominal galaxy/cluster median sigma: **{galaxy_sigma:.6g} / {cluster_sigma:.6g}**.
- Nominal cluster/galaxy sigma ratio: **{sigma_ratio:.4g}x**.
- Nominal galaxy/cluster multipole gate: **{galaxy_gate:.6g} / {cluster_gate:.6g}**.
- Weakest mass-sensitivity sigma ratio: **{min(sensitivity_ratios.values()):.4g}x**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Spent and sealed lensing outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
