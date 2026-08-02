#!/usr/bin/env python3
"""Test quadratic coherent-vector accumulation with all outcomes sealed."""

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

from run_p0660_exact_tensor_activation_audit import (
    evaluate as evaluate_registered,
)
from run_p0660_exact_tensor_activation_audit import (
    manifest_sha256,
    registered_map_audits,
    sha256,
    synthetic_audits,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0661_quadratic_coherence_tensor_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predecessor_failed_only_domain_separation(predecessor: dict) -> bool:
    failed = [name for name, passed in predecessor["gate_results"].items() if not passed]
    return predecessor["status"] == "fail" and failed == ["registered_domain_separation"]


def analytic_survival_audit(protocol):
    power = float(protocol["definitions"]["coherence_power"])
    analytic = protocol["analytic_tests"]
    ratios = np.asarray(analytic["short_path_length_ratios"], dtype=float)
    survival = 1.0 - np.exp(-np.power(ratios, power))
    slope = float(np.polyfit(np.log(ratios), np.log(survival), 1)[0])
    long_ratio = float(analytic["long_path_length_ratio"])
    long_survival = float(1.0 - np.exp(-(long_ratio**power)))
    return ratios, survival, slope, long_survival


def make_figure(observed, metrics, ratios, survival, output):
    primary = observed[
        observed.resolution.eq("primary") & observed.scenario.eq("nominal")
    ].copy()
    primary["map_type"] = primary.domain.str.contains("cluster").map(
        {False: "galaxy", True: "cluster"}
    )
    primary = primary.sort_values(["map_type", "sigma_weighted_mean"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = primary.map_type.map({"galaxy": "#3274a1", "cluster": "#d95f02"})
    axes[0].bar(primary.case, primary.sigma_weighted_mean, color=colors)
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_ylabel("weighted quadratic tensor sigma")
    axes[0].set_title("Registered nominal maps")
    sensitivity = metrics["mass_sensitivity_cluster_to_galaxy_ratios"]
    axes[1].bar(sensitivity.keys(), sensitivity.values(), color="#55a868")
    axes[1].axhline(10.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("cluster / galaxy median sigma")
    axes[1].set_title("Mass-map sensitivities")
    dense_ratio = np.linspace(0.0, 3.0, 300)
    dense_survival = 1.0 - np.exp(-dense_ratio**2)
    axes[2].plot(dense_ratio, dense_survival, color="#c44e52")
    axes[2].scatter(ratios, survival, color="black", s=18)
    axes[2].set_xlabel("coherent path / 10 kpc")
    axes[2].set_ylabel("survival")
    axes[2].set_title("Quadratic short-path law")
    figure.suptitle("P0661 quadratic coherence tensor audit")
    figure.tight_layout()
    figure.savefig(output / "p0661_quadratic_coherence_tensor.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0661_activation_score":
        raise RuntimeError("P0661 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    predecessor = read_json(ROOT / protocol["diagnostic_predecessor"])
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0659 parent no longer passes")
    if not predecessor_failed_only_domain_separation(predecessor):
        raise RuntimeError("P0660 diagnostic state changed")

    ratios, survival, short_slope, long_survival = analytic_survival_audit(protocol)
    synthetic, rotation_error, reversal_error = synthetic_audits(protocol)
    observed, map_paths = registered_map_audits(protocol)

    compatible_protocol = json.loads(json.dumps(protocol))
    compatible_gates = compatible_protocol["predeclared_progression_gates"]
    compatible_gates["cluster_to_galaxy_sigma_ratio_above_one_in_all_mass_sensitivities"] = True
    base_gates, metrics, domain_summary, resolution, resolution_summary = evaluate_registered(
        compatible_protocol,
        parent,
        synthetic,
        observed,
        rotation_error,
        reversal_error,
    )
    gates = protocol["predeclared_progression_gates"]
    sensitivity_minimum = float(
        min(metrics["mass_sensitivity_cluster_to_galaxy_ratios"].values())
    )
    gate_results = {
        "P0659_parent": base_gates.pop("P0659_parent"),
        "P0660_diagnostic_state": predecessor_failed_only_domain_separation(predecessor)
        is bool(gates["P0660_failed_only_linear_kernel_domain_separation"]),
        "fixed_quadratic_power": float(protocol["definitions"]["coherence_power"])
        == float(gates["coherence_power_exactly_two"]),
        "short_path_quadratic_slope": gates["short_path_log_slope_min"]
        <= short_slope
        <= gates["short_path_log_slope_max"],
        "long_path_saturation": long_survival >= gates["long_path_survival_min"],
        **base_gates,
    }
    gate_results["mass_sensitivity_sign_stable"] = sensitivity_minimum >= float(
        gates["cluster_to_galaxy_sigma_ratio_min_in_all_mass_sensitivities"]
    )
    all_pass = bool(all(gate_results.values()))
    metrics.update(
        {
            "coherence_power": float(protocol["definitions"]["coherence_power"]),
            "short_path_log_slope": short_slope,
            "long_path_survival": long_survival,
            "minimum_mass_sensitivity_cluster_to_galaxy_ratio": sensitivity_minimum,
        }
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0661-QUADRATIC-COHERENCE-TENSOR-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_real_map_field_solves": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(ROOT / "src/voidscreen/tensor_activation.py"),
        "audit_base_source_sha256": sha256(
            ROOT / "scripts/run_p0660_exact_tensor_activation_audit.py"
        ),
        "registered_map_manifest_sha256": manifest_sha256(map_paths),
        "coverage": {
            "registered_galaxies": int(
                observed[observed.domain.eq("registered_galaxy_baryons_only")].case.nunique()
            ),
            "registered_clusters": int(
                observed[observed.domain.eq("registered_cluster_baryons_only")].case.nunique()
            ),
            "mass_scenarios": int(observed.scenario.nunique()),
            "resolutions": int(observed.resolution.nunique()),
            "new_universal_constants_after_P0659": int(
                protocol["definitions"]["new_universal_constants_after_P0659"]
            ),
            "per_object_gravity_parameters": int(
                protocol["definitions"]["per_object_gravity_parameters"]
            ),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "domain_summary": domain_summary.to_dict(orient="records"),
        "resolution_summary": resolution_summary.to_dict(orient="records"),
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    synthetic.to_csv(output / "synthetic_activation_audits.csv", index=False)
    observed.to_csv(output / "registered_map_activation_scores.csv", index=False)
    domain_summary.to_csv(output / "domain_summary.csv", index=False)
    resolution.to_csv(output / "resolution_audit.csv", index=False)
    make_figure(observed, metrics, ratios, survival, output)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0661 quadratic coherence tensor audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Short-path log slope: **{short_slope:.6f}**; survival at `ell/Lc=2`: **{long_survival:.6f}**.
- Exact nominal weighted sigma: galaxies **{metrics['registered_galaxy_nominal_median_sigma']:.6g}**, clusters **{metrics['registered_cluster_nominal_median_sigma']:.6g}**.
- Exact cluster/galaxy ratio: **{metrics['registered_cluster_to_galaxy_nominal_median_sigma_ratio']:.4g}x**.
- Weakest mass-map ratio: **{sensitivity_minimum:.4g}x**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633 velocities and P0640 lensing constraints opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
