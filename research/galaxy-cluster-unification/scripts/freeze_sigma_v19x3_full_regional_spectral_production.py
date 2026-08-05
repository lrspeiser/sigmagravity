#!/usr/bin/env python3
"""Freeze full 494-region production only after V19X2 passes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x3_full_regional_spectral_production as successor
import sigma_v19x2_unified_response_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_X2_CONFIG = (
    ROOT / "configs" / "sigma_v19x2_unified_spectral_combination_commissioning.json"
)
DEFAULT_X2_RUNNER = (
    ROOT / "scripts" / "run_sigma_v19x2_unified_spectral_combination_commissioning.py"
)
DEFAULT_X2_REPORT = (
    ROOT
    / "results"
    / "sigma_v19x2_unified_spectral_combination_commissioning"
    / "report.json"
)
DEFAULT_V19H_CONFIG = ROOT / "configs" / "sigma_v19h_causal_observable_protocol.json"
DEFAULT_SUCCESSOR = ROOT / "scripts" / "run_sigma_v19x3_full_regional_spectral_production.py"
DEFAULT_OUTPUT = ROOT / "configs" / "sigma_v19x3_full_regional_spectral_production.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_terminal_x2(
    config_path: Path, runner_path: Path, report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (config_path, runner_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"V19X3 freeze parent is absent: {path}")
    config = load_json(config_path)
    report = load_json(report_path)
    if config.get("freeze_state") != "frozen_after_terminal_v19w4_pass":
        raise RuntimeError("V19X2 configuration is not terminally frozen")
    implementation = config.get("implementation", {})
    if implementation.get("runner") != relative(runner_path):
        raise RuntimeError("V19X2 configuration names another runner")
    if implementation.get("runner_sha256") != adapter.sha256(runner_path):
        raise RuntimeError("V19X2 runner changed after freeze")
    if report.get("status") != successor.AUTHORIZED_X2_STATUS:
        raise RuntimeError("V19X2 report did not pass commissioning")
    if report.get("config_sha256") != adapter.sha256(config_path):
        raise RuntimeError("V19X2 report names another config")
    if report.get("runner_sha256") != adapter.sha256(runner_path):
        raise RuntimeError("V19X2 report names another runner")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("V19X2 report contains a failed gate")
    if report.get("full_494_region_combination_and_fit_authorized") is not True:
        raise RuntimeError("V19X2 did not authorize full regional fits")
    if report.get("replacement_cluster_lensing_target_opened") is not False:
        raise RuntimeError("V19X2 opened a prohibited lensing target")
    expected_clusters = set(config["registered_workload"]["clusters"])
    fit_clusters = {
        row["cluster"]
        for row in report.get("integrated_fits", [])
        if row.get("fit_completed") and row.get("gates", {}).get("all_passed")
    }
    if fit_clusters != expected_clusters:
        raise RuntimeError("V19X2 integrated fit inventory did not pass")
    return config, report


def freeze_config(
    x2_config_path: Path,
    x2_runner_path: Path,
    x2_report_path: Path,
    v19h_config_path: Path,
    successor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path in (v19h_config_path, successor_path):
        if not path.is_file():
            raise RuntimeError(f"V19X3 freeze parent is absent: {path}")
    x2_config, x2_report = validate_terminal_x2(
        x2_config_path, x2_runner_path, x2_report_path
    )
    v19h = load_json(v19h_config_path)
    minimum_quality = int(
        v19h["adaptive_thermodynamics"]["fit_gates"][
            "minimum_passing_regions_per_cluster"
        ]
    )
    workload = copy.deepcopy(x2_config["registered_workload"])
    total_regions = sum(
        int(row["total_regions"]) for row in workload["clusters"].values()
    )
    if total_regions != 494 or minimum_quality != 12:
        raise RuntimeError("V19X3 inherited workload or V19H quality gate changed")

    parents = copy.deepcopy(x2_config["parents"])
    parents.update(
        {
            "v19x2_config": relative(x2_config_path),
            "v19x2_config_sha256": adapter.sha256(x2_config_path),
            "v19x2_runner": relative(x2_runner_path),
            "v19x2_runner_sha256": adapter.sha256(x2_runner_path),
            "v19x2_report": relative(x2_report_path),
            "v19x2_report_sha256": adapter.sha256(x2_report_path),
            "v19h_source_protocol": relative(v19h_config_path),
            "v19h_source_protocol_sha256": adapter.sha256(v19h_config_path),
        }
    )
    runtime_x2 = x2_config["runtime_authorization"]
    config: dict[str, Any] = {
        "protocol_version": "SIGMA-V19X3-FULL-REGIONAL-SPECTRAL-PRODUCTION-1.0.0",
        "freeze_state": "frozen_after_terminal_v19x2_pass",
        "status": "frozen mechanically after V19X2 passed both integrated and both target-blind commissioning-region fits, before combining or fitting any other region, constructing a gas source state, opening lensing or halo data, or changing gravity physics",
        "purpose": "Combine and fit every one of the 494 frozen thermodynamic regions with the exact V19X2 response, grouping, plasma-model, abundance and uncertainty rules, retaining every outcome.",
        "parents": parents,
        "runtime_authorization": {
            "required_v19x2_report": relative(x2_report_path),
            "required_v19x2_config_sha256": adapter.sha256(x2_config_path),
            "required_v19x2_runner_sha256": adapter.sha256(x2_runner_path),
            "required_v19w4_report": runtime_x2["required_v19w4_report"],
            "required_unified_cells": int(runtime_x2["required_unified_cells"]),
            "required_completed_cells": int(runtime_x2["required_unified_cells"]),
            "required_unified_products": int(runtime_x2["required_unified_products"]),
            "may_start_before_v19x2_authorization": False,
        },
        "execution": {
            "runner": relative(successor_path),
            "response_archives": copy.deepcopy(
                x2_config["execution"]["response_archives"]
            ),
            "scratch_root": "/home/henry/sigma-v19x3-full-regional-spectra/v100",
            "result_root": "results/sigma_v19x3_full_regional_spectral_production",
            "checkpoint_unit": "one combination checkpoint and one fit checkpoint per cluster/bin",
            "combination_concurrency": 1,
            "failure_rule": "retain the failed stage and every completed checkpoint; do not drop, merge, split, regroup or selectively refit a region",
        },
        "registered_workload": workload,
        "combination": copy.deepcopy(x2_config["combination"]),
        "fit_sequence": copy.deepcopy(x2_config["fit_sequence"]),
        "gates": copy.deepcopy(x2_config["gates"]),
        "regional_gates": {
            "expected_total_regions": total_regions,
            "expected_regions_by_cluster": {
                cluster: int(row["total_regions"])
                for cluster, row in workload["clusters"].items()
            },
            "every_region_attempted": True,
            "every_region_requires_finite_temperature_abundance_and_normalization_best_fit": True,
            "minimum_quality_passes_per_cluster": minimum_quality,
            "quality_pass_definition": "the unchanged V19X2 ordered 68-percent temperature interval, fractional half-width, reduced-statistic and free-parameter-bound gates",
            "retention_rule": "A finite best fit remains in the gas-map posterior even if its individual quality subgate fails; report that uncertainty and never drop the region.",
        },
        "advance": {
            "if_all_gates_pass": "freeze the emission-measure, gas-density, pressure, entropy, front-state and projection posterior construction before opening lensing",
            "if_any_gate_fails": "report the failed region and stage; do not alter the spectral model, grouping, region geometry, response membership or gravity physics",
            "gravity_theory_tested_here": False,
        },
        "authorization": {
            "combine_and_fit_all_494_regions": True,
            "construct_gas_source_state_before_regional_gate_passes": False,
            "open_lensing_or_halo_payload": False,
            "select_source_invariant_or_action": False,
            "change_gravity_formula_or_parameter": False,
        },
        "integrity": {
            "v19x2_report_passed_at_freeze": True,
            "v19x2_integrated_abundances_known_as_measurement_outputs": {
                row["cluster"]: float(row["parameters"]["abundance_solar"])
                for row in x2_report["integrated_fits"]
            },
            "noncommissioning_regional_temperature_known_at_freeze": False,
            "gas_source_state_known_at_freeze": False,
            "lensing_halo_or_gravity_payload_opened_at_freeze": False,
        },
        "claim_boundary": [
            "A pass supplies target-blind regional plasma measurements; it is not evidence for Sigma Gravity.",
            "The two commissioning-region temperatures are already spent implementation evidence and remain part of the complete 494-region output.",
            "Individual quality failures are retained if the best fit is finite; only the frozen cluster-level minimum determines source-map authorization.",
            "No lensing, inferred halo, source-invariant score, action or gravity parameter is read or selected here.",
        ],
        "implementation": {
            "freezer": relative(Path(__file__)),
            "freezer_sha256": adapter.sha256(Path(__file__)),
            "runner": relative(successor_path),
            "runner_sha256": adapter.sha256(successor_path),
            "adapter": x2_config["implementation"]["adapter"],
            "adapter_sha256": x2_config["implementation"]["adapter_sha256"],
        },
    }
    successor.validate_frozen_runner(config)
    successor.validate_frozen_parents(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19x2-config", type=Path, default=DEFAULT_X2_CONFIG)
    parser.add_argument("--v19x2-runner", type=Path, default=DEFAULT_X2_RUNNER)
    parser.add_argument("--v19x2-report", type=Path, default=DEFAULT_X2_REPORT)
    parser.add_argument("--v19h-config", type=Path, default=DEFAULT_V19H_CONFIG)
    parser.add_argument("--successor", type=Path, default=DEFAULT_SUCCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = freeze_config(
        args.v19x2_config.resolve(),
        args.v19x2_runner.resolve(),
        args.v19x2_report.resolve(),
        args.v19h_config.resolve(),
        args.successor.resolve(),
        args.output.resolve(),
    )
    print(args.output.resolve())
    print(config["status"])


if __name__ == "__main__":
    main()
