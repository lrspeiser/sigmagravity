#!/usr/bin/env python3
"""Evaluate frozen amplitude-level multipole activation on registered maps."""

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

DEFAULT_CONFIG = ROOT / "configs" / "p0669_amplitude_multipole_3d_activation.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def transform_sufficient_statistics(parent_scores: pd.DataFrame) -> pd.DataFrame:
    """Transform exactly because each map's multipole gate is spatially global."""
    scores = parent_scores.copy()
    gate = np.asarray(scores.multipole_gate, dtype=float)
    if np.any(gate <= 0.0) or np.any(gate > 1.0):
        raise RuntimeError("parent multipole gates are outside the transform domain")
    amplitude = np.sqrt(gate)
    scores["intensity_gate"] = gate
    scores["amplitude_gate"] = amplitude
    scores["final_sigma_mass_weighted_mean"] = (
        scores.local_sigma_mass_weighted_mean * amplitude
    )
    scores["sigma_global_minimum"] = scores.sigma_global_minimum / amplitude
    scores["sigma_global_maximum"] = scores.sigma_global_maximum / amplitude
    scores["minimum_constitutive_eigenvalue_conservative_bound"] = (
        1e-6 * (1.0 - scores.sigma_global_maximum)
    )
    return scores


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["domain", "scenario"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_sigma=("final_sigma_mass_weighted_mean", "median"),
            minimum_sigma=("final_sigma_mass_weighted_mean", "min"),
            maximum_sigma=("final_sigma_mass_weighted_mean", "max"),
            median_amplitude_gate=("amplitude_gate", "median"),
            median_intensity_gate=("intensity_gate", "median"),
            median_trace_length_kpc=("trace_length_mass_weighted_kpc", "median"),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0669_registered_map_score":
        raise RuntimeError("P0669 protocol is not frozen")
    parent_path = ROOT / protocol["development_parent"]
    synthetic_path = ROOT / protocol["synthetic_parent"]
    parent = read_json(parent_path)
    synthetic = read_json(synthetic_path)
    parent_failed = [
        name for name, passed in parent["gate_results"].items() if not passed
    ]
    if parent_failed != ["cluster_channel_present"]:
        raise RuntimeError("P0668 failure mode changed")
    if not synthetic["all_progression_gates_pass"]:
        raise RuntimeError("P0667 synthetic parent no longer passes")
    parent_scores_path = parent_path.parent / "registered_multipole_3d_scores.csv"
    scores = transform_sufficient_statistics(pd.read_csv(parent_scores_path))
    summary = summarize(scores)
    nominal = summary[summary.scenario.eq("nominal")].set_index("domain")
    galaxy_domain = "registered_galaxy_baryons_only"
    cluster_domain = "registered_cluster_baryons_only"
    galaxy_sigma = float(nominal.loc[galaxy_domain, "median_sigma"])
    cluster_sigma = float(nominal.loc[cluster_domain, "median_sigma"])
    sigma_ratio = cluster_sigma / max(galaxy_sigma, np.finfo(float).tiny)
    galaxy_gate = float(nominal.loc[galaxy_domain, "median_amplitude_gate"])
    cluster_gate = float(nominal.loc[cluster_domain, "median_amplitude_gate"])
    gate_ratio = cluster_gate / max(galaxy_gate, np.finfo(float).tiny)
    sensitivities = {}
    for scenario in protocol["map_inputs"]["galaxy_stellar_scale_sensitivity"]:
        block = summary[summary.scenario.eq(scenario)].set_index("domain")
        sensitivities[scenario] = float(
            block.loc[cluster_domain, "median_sigma"]
            / max(float(block.loc[galaxy_domain, "median_sigma"]), np.finfo(float).tiny)
        )
    galaxy_count = int(scores[scores.domain.eq(galaxy_domain)].case.nunique())
    cluster_count = int(scores[scores.domain.eq(cluster_domain)].case.nunique())
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
    minimum_eigenvalue = float(
        scores.minimum_constitutive_eigenvalue_conservative_bound.min()
    )
    gates = protocol["predeclared_progression_gates"]
    candidate = protocol["candidate"]
    gate_results = {
        "P0667_synthetic_parent": bool(synthetic["all_progression_gates_pass"])
        is bool(gates["P0667_synthetic_all_progression_gates_pass"]),
        "P0668_failure_mode": (parent_failed == ["cluster_channel_present"])
        is bool(gates["P0668_failed_only_absolute_cluster_channel"]),
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
        "mass_sensitivity": min(sensitivities.values())
        >= gates["cluster_to_galaxy_sigma_ratio_min_in_all_mass_sensitivities"],
        "amplitude_gate_separation": gate_ratio
        >= gates[
            "registered_cluster_to_galaxy_nominal_median_amplitude_gate_ratio_min"
        ],
        "no_new_constants": int(candidate["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(candidate["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "spent_lensing_untouched": not bool(gates["spent_lensing_outcomes_opened"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    all_pass = bool(all(gate_results.values()))
    map_inputs = protocol["map_inputs"]
    map_paths = sorted((ROOT / map_inputs["galaxies"]).glob("*.npz")) + sorted(
        (ROOT / map_inputs["clusters"]).glob("*.npz")
    )
    metrics = {
        "registered_galaxy_nominal_median_sigma": galaxy_sigma,
        "registered_cluster_nominal_median_sigma": cluster_sigma,
        "registered_cluster_to_galaxy_nominal_median_sigma_ratio": sigma_ratio,
        "mass_sensitivity_cluster_to_galaxy_sigma_ratios": sensitivities,
        "registered_galaxy_nominal_median_amplitude_gate": galaxy_gate,
        "registered_cluster_nominal_median_amplitude_gate": cluster_gate,
        "registered_cluster_to_galaxy_nominal_median_amplitude_gate_ratio": gate_ratio,
        "maximum_component_mass_relative_error": maximum_mass_error,
        "minimum_constitutive_eigenvalue_conservative_bound": minimum_eigenvalue,
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0669-AMPLITUDE-MULTIPOLE-3D-ACTIVATION-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_spent_RXJ2129_3D_map_build": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(
            ROOT / "src/voidscreen/amplitude_activation_3d.py"
        ),
        "parent_scores_sha256": sha256(parent_scores_path),
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
        "sufficient_statistic_identity": "mean(sqrt(M)*local_sigma)=sqrt(M)*mean(local_sigma) because M is global per registered map",
        "spent_RXJ2129_lensing_outcomes_opened": False,
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    scores.to_csv(output / "registered_amplitude_3d_scores.csv", index=False)
    summary.to_csv(output / "domain_summary.csv", index=False)
    nominal_scores = scores[scores.scenario.eq("nominal")].sort_values(
        ["domain", "final_sigma_mass_weighted_mean"]
    )
    colors = nominal_scores.domain.str.contains("cluster").map(
        {False: "#3274a1", True: "#d95f02"}
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(nominal_scores.case, nominal_scores.amplitude_gate, color=colors)
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_ylabel("multipole amplitude gate")
    axes[0].set_title("Registered baryonic field amplitudes")
    axes[1].bar(
        nominal_scores.case,
        nominal_scores.final_sigma_mass_weighted_mean,
        color=colors,
    )
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    axes[1].set_ylabel("mass-weighted final sigma")
    axes[1].set_title("Amplitude-gated 3D activation")
    figure.suptitle("P0669 registered amplitude-multipole activation")
    figure.tight_layout()
    figure.savefig(output / "p0669_amplitude_multipole_3d.png", dpi=180)
    plt.close(figure)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0669 amplitude-multipole 3D activation

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Nominal galaxy/cluster median sigma: **{galaxy_sigma:.6g} / {cluster_sigma:.6g}**.
- Nominal cluster/galaxy sigma ratio: **{sigma_ratio:.4g}x**.
- Nominal galaxy/cluster amplitude gate: **{galaxy_gate:.6g} / {cluster_gate:.6g}**.
- Weakest mass-sensitivity sigma ratio: **{min(sensitivities.values()):.4g}x**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Spent and sealed target outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
