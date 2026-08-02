#!/usr/bin/env python3
"""Retest P0662 resolution stability with a conservative anti-alias operator."""

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
from run_p0662_physical_tidal_length_tensor_audit import (
    activation_row,
    failed_gates,
    registered_map_audits,
)

from voidscreen.geometric_transport import resample_surface_density
from voidscreen.observational_resampling import common_resolution_surface_density

DEFAULT_CONFIG = ROOT / "configs" / "p0663_common_resolution_tensor_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def common_pair_rows(
    case,
    domain,
    scenario,
    stars,
    gas,
    native_cell_kpc,
    target_cells,
    protocol,
):
    stellar_pair = common_resolution_surface_density(stars, target_cells)
    gas_pair = common_resolution_surface_density(gas, target_cells)
    ratio = stellar_pair.downsampling_ratio
    rows = [
        activation_row(
            case,
            domain,
            scenario,
            "common_native",
            stellar_pair.filtered_native,
            gas_pair.filtered_native,
            native_cell_kpc,
            protocol,
        ),
        activation_row(
            case,
            domain,
            scenario,
            "common_coarse",
            stellar_pair.coarse,
            gas_pair.coarse,
            native_cell_kpc * ratio,
            protocol,
        ),
    ]
    audit = {
        "case": case,
        "domain": domain,
        "scenario": scenario,
        "downsampling_ratio": ratio,
        "added_native_gaussian_sigma_pixels": stellar_pair.added_native_gaussian_sigma_pixels,
        "stellar_filtered_mass_relative_error": stellar_pair.filtered_mass_relative_error,
        "stellar_coarse_mass_relative_error": stellar_pair.coarse_mass_relative_error,
        "gas_filtered_mass_relative_error": gas_pair.filtered_mass_relative_error,
        "gas_coarse_mass_relative_error": gas_pair.coarse_mass_relative_error,
    }
    return rows, audit


def common_resolution_audits(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    audits = []
    map_paths = []
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
        map_paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            nominal_stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        native_cells = int(inputs["galaxy_native_cells"])
        target = int(inputs["galaxy_resolution_check_cells"])
        cell = float((axis[-1] - axis[0]) / (native_cells - 1))
        for scenario, scale in inputs["galaxy_stellar_scale_sensitivity"].items():
            pair_rows, audit = common_pair_rows(
                path.stem,
                "registered_galaxy_baryons_only",
                scenario,
                nominal_stars * float(scale),
                gas,
                cell,
                target,
                protocol,
            )
            rows.extend(pair_rows)
            audits.append(audit)
    for path in sorted((ROOT / inputs["clusters"]).glob("*.npz")):
        map_paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            maps = {
                scenario: (data[keys[0]].astype(float), data[keys[1]].astype(float))
                for scenario, keys in inputs["cluster_sensitivity_maps"].items()
            }
        native_cells = int(inputs["cluster_primary_cells"])
        target = int(inputs["cluster_resolution_check_cells"])
        cell = float((axis[-1] - axis[0]) / (native_cells - 1))
        for scenario, (stars, gas) in maps.items():
            pair_rows, audit = common_pair_rows(
                path.stem.replace("_baryons", ""),
                "registered_cluster_baryons_only",
                scenario,
                resample_surface_density(stars, native_cells),
                resample_surface_density(gas, native_cells),
                cell,
                target,
                protocol,
            )
            rows.extend(pair_rows)
            audits.append(audit)
    return pd.DataFrame(rows), pd.DataFrame(audits), map_paths


def resolution_comparison(scores):
    comparison = scores.pivot_table(
        index=["case", "domain", "scenario"],
        columns="resolution",
        values="sigma_weighted_mean",
    ).reset_index()
    comparison["absolute_fractional_change"] = np.abs(
        comparison.common_coarse
        / np.maximum(comparison.common_native, np.finfo(float).tiny)
        - 1.0
    )
    summary = (
        comparison.groupby("domain", as_index=False)
        .absolute_fractional_change.median()
        .rename(columns={"absolute_fractional_change": "median_absolute_fractional_change"})
    )
    return comparison, summary


def primary_scores_unchanged(protocol, predecessor):
    recomputed, _ = registered_map_audits(protocol)
    recomputed = recomputed[recomputed.resolution.eq("primary")].sort_values(
        ["case", "domain", "scenario"]
    )
    predecessor_scores = pd.read_csv(
        ROOT
        / "results/p0662_physical_tidal_length_tensor_audit/registered_map_activation_scores.csv",
        float_precision="round_trip",
    )
    predecessor_scores = predecessor_scores[
        predecessor_scores.resolution.eq("primary")
    ].sort_values(["case", "domain", "scenario"])
    identifiers_equal = recomputed[["case", "domain", "scenario"]].reset_index(
        drop=True
    ).equals(
        predecessor_scores[["case", "domain", "scenario"]].reset_index(drop=True)
    )
    score_equal = np.array_equal(
        recomputed.sigma_weighted_mean.to_numpy(float),
        predecessor_scores.sigma_weighted_mean.to_numpy(float),
    )
    metrics_equal = np.isclose(
        float(
            recomputed[
                recomputed.domain.eq("registered_galaxy_baryons_only")
                & recomputed.scenario.eq("nominal")
            ].sigma_weighted_mean.median()
        ),
        float(predecessor["metrics"]["registered_galaxy_nominal_median_sigma"]),
        rtol=0.0,
        atol=0.0,
    )
    return bool(identifiers_equal and score_equal and metrics_equal)


def make_figure(predecessor, summary, comparison, output):
    metrics = predecessor["metrics"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    axes[0].bar(
        ["galaxy", "cluster"],
        [
            metrics["registered_galaxy_nominal_median_sigma"],
            metrics["registered_cluster_nominal_median_sigma"],
        ],
        color=["#3274a1", "#d95f02"],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("unchanged primary sigma")
    axes[0].set_title("P0662 primary scores")
    plot_summary = summary.copy()
    plot_summary["label"] = plot_summary.domain.str.contains("cluster").map(
        {False: "galaxy", True: "cluster"}
    )
    axes[1].bar(
        plot_summary.label,
        plot_summary.median_absolute_fractional_change,
        color="#55a868",
    )
    axes[1].axhline(0.35, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("median common-resolution change")
    axes[1].set_title("Anti-aliased grid comparison")
    nominal = comparison[comparison.scenario.eq("nominal")].copy()
    nominal["map_type"] = nominal.domain.str.contains("cluster").map(
        {False: "galaxy", True: "cluster"}
    )
    axes[2].scatter(
        nominal.common_native,
        nominal.common_coarse,
        c=nominal.map_type.map({"galaxy": "#3274a1", "cluster": "#d95f02"}),
    )
    limits = [
        max(float(nominal[["common_native", "common_coarse"]].min().min()) * 0.7, 1e-8),
        float(nominal[["common_native", "common_coarse"]].max().max()) * 1.3,
    ]
    axes[2].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel("filtered native sigma")
    axes[2].set_ylabel("coarse sigma")
    axes[2].set_title("Per-system convergence")
    figure.suptitle("P0663 common-resolution tensor audit")
    figure.tight_layout()
    figure.savefig(output / "p0663_common_resolution_tensor.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0663_resolution_score":
        raise RuntimeError("P0663 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    predecessor = read_json(ROOT / protocol["diagnostic_predecessor"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0659 parent no longer passes")
    if failed_gates(predecessor) != ["resolution_stability"]:
        raise RuntimeError("P0662 diagnostic state changed")

    scores, mass_audits, map_paths = common_resolution_audits(protocol)
    comparison, resolution_summary = resolution_comparison(scores)
    primary_unchanged = primary_scores_unchanged(protocol, predecessor)
    mass_columns = [column for column in mass_audits if column.endswith("mass_relative_error")]
    maximum_mass_error = float(mass_audits[mass_columns].to_numpy(float).max())
    maximum_domain_median_change = float(
        resolution_summary.median_absolute_fractional_change.max()
    )
    metrics = dict(predecessor["metrics"])
    sensitivity_minimum = float(
        min(metrics["mass_sensitivity_cluster_to_galaxy_ratios"].values())
    )
    gates = protocol["predeclared_progression_gates"]
    definitions = protocol["definitions"]
    galaxy_count = int(
        scores[scores.domain.eq("registered_galaxy_baryons_only")].case.nunique()
    )
    cluster_count = int(
        scores[scores.domain.eq("registered_cluster_baryons_only")].case.nunique()
    )
    gate_results = {
        "P0659_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0659_all_progression_gates_pass"]),
        "P0662_diagnostic_state": failed_gates(predecessor) == ["resolution_stability"],
        "mass_conservation": maximum_mass_error
        <= gates["common_resolution_mass_conservation_relative_error_max"],
        "primary_scores_unchanged": primary_unchanged
        is bool(gates["primary_candidate_scores_bitwise_unchanged_from_P0662"]),
        "galaxy_coverage": galaxy_count == int(gates["registered_galaxy_count"]),
        "cluster_coverage": cluster_count == int(gates["registered_cluster_count"]),
        "registered_domain_separation": metrics[
            "registered_cluster_to_galaxy_nominal_median_sigma_ratio"
        ]
        >= gates["registered_cluster_to_galaxy_nominal_median_sigma_ratio_min"],
        "galaxy_channel_small": metrics["registered_galaxy_nominal_median_sigma"]
        <= gates["registered_galaxy_nominal_median_sigma_max"],
        "cluster_channel_present": metrics["registered_cluster_nominal_median_sigma"]
        >= gates["registered_cluster_nominal_median_sigma_min"],
        "mass_sensitivity": sensitivity_minimum
        >= gates["cluster_to_galaxy_sigma_ratio_min_in_all_mass_sensitivities"],
        "common_resolution_stability": maximum_domain_median_change
        <= gates["common_resolution_median_change_fraction_max"],
        "solar_proxy": metrics["solar_1au_constitutive_anisotropy"]
        <= gates["solar_1au_constitutive_anisotropy_max"],
        "no_new_constants": int(definitions["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(definitions["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    metrics.update(
        {
            "maximum_common_resolution_mass_relative_error": maximum_mass_error,
            "maximum_domain_median_common_resolution_change_fraction": maximum_domain_median_change,
            "minimum_mass_sensitivity_cluster_to_galaxy_ratio": sensitivity_minimum,
        }
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0663-COMMON-RESOLUTION-TENSOR-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_outcome_blind_real_map_field_solves": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "resampling_source_sha256": sha256(
            ROOT / "src/voidscreen/observational_resampling.py"
        ),
        "registered_map_manifest_sha256": manifest_sha256(map_paths),
        "coverage": {
            "registered_galaxies": galaxy_count,
            "registered_clusters": cluster_count,
            "mass_scenarios": int(scores.scenario.nunique()),
            "common_resolutions": int(scores.resolution.nunique()),
            "new_universal_constants_after_P0659": int(
                definitions["new_universal_constants_after_P0659"]
            ),
            "per_object_gravity_parameters": int(definitions["per_object_gravity_parameters"]),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "resolution_summary": resolution_summary.to_dict(orient="records"),
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    scores.to_csv(output / "common_resolution_activation_scores.csv", index=False)
    mass_audits.to_csv(output / "mass_conservation_audit.csv", index=False)
    comparison.to_csv(output / "common_resolution_comparison.csv", index=False)
    make_figure(predecessor, resolution_summary, comparison, output)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0663 common-resolution tensor audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- P0662 primary scores reproduced bitwise: **{'yes' if primary_unchanged else 'no'}**.
- Maximum mass-conservation error: **{maximum_mass_error:.3e}**.
- Maximum domain-median common-resolution change: **{maximum_domain_median_change:.3%}**.
- Primary cluster/galaxy activation remains **{metrics['registered_cluster_to_galaxy_nominal_median_sigma_ratio']:.4g}x**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633 velocities and P0640 lensing constraints opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
