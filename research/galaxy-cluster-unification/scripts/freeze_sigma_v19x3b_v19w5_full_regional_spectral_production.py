#!/usr/bin/env python3
"""Freeze V19X3B after the V19W5-authorized V19X2 commissioning passes."""

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

import run_sigma_v19x3b_v19w5_full_regional_spectral_production as successor
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
DEFAULT_INHERITED_X3 = (
    ROOT / "scripts" / "run_sigma_v19x3_full_regional_spectral_production.py"
)
DEFAULT_SUCCESSOR = (
    ROOT / "scripts" / "run_sigma_v19x3b_v19w5_full_regional_spectral_production.py"
)
DEFAULT_OUTPUT = (
    ROOT / "configs" / "sigma_v19x3b_v19w5_full_regional_spectral_production.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def validate_terminal_x2(
    config_path: Path, runner_path: Path, report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (config_path, runner_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"V19X3B freeze parent is absent: {path}")
    config = load_json(config_path)
    report = load_json(report_path)
    runtime = config.get("runtime_authorization", {})
    archives = config.get("execution", {}).get("response_archives", {})
    if config.get("freeze_state") != "frozen_after_terminal_v19w5_pass":
        raise RuntimeError("V19X2 configuration is not terminally frozen from V19W5")
    if (
        runtime.get("response_authority") != "V19W5"
        or runtime.get("required_status") != adapter.V19W5_AUTHORIZED_STATUS
        or runtime.get("recovery_archive") != "v19w5_recovery"
        or set(archives) != {"base_v19w", "v19w5_recovery"}
    ):
        raise RuntimeError("V19X2 configuration does not preserve V19W5 authority")
    parents = config.get("parents", {})
    for name in (
        "v19w5_config",
        "v19w5_runner",
        "v19w5_report",
        "v19w5_unified_index",
    ):
        path = ROOT / str(parents.get(name, ""))
        if (
            not path.is_file()
            or adapter.sha256(path) != parents.get(f"{name}_sha256")
        ):
            raise RuntimeError(f"V19X2 changed or omitted V19W5 parent: {name}")
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
    inherited_x3_path: Path,
    successor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path in (v19h_config_path, inherited_x3_path, successor_path):
        if not path.is_file():
            raise RuntimeError(f"V19X3B freeze parent is absent: {path}")
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
        raise RuntimeError("V19X3B inherited workload or V19H quality gate changed")

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
        "protocol_version": (
            "SIGMA-V19X3B-V19W5-FULL-REGIONAL-SPECTRAL-PRODUCTION-1.0.0"
        ),
        "freeze_state": successor.FROZEN_STATE,
        "status": "frozen mechanically after V19W5-authorized V19X2 passed both integrated and both target-blind commissioning-region fits; before combining or fitting any other region, constructing a gas source state, opening lensing or halo data, or changing gravity physics",
        "purpose": "Run all 494 frozen thermodynamic regions with the unchanged preregistered V19X3 engine and the explicit V19W5 response authority inherited from V19X2.",
        "parents": parents,
        "runtime_authorization": {
            "required_v19x2_report": relative(x2_report_path),
            "required_v19x2_config_sha256": adapter.sha256(x2_config_path),
            "required_v19x2_runner_sha256": adapter.sha256(x2_runner_path),
            "response_authority": runtime_x2["response_authority"],
            "required_response_report": runtime_x2["required_response_report"],
            "required_response_status": runtime_x2["required_status"],
            "recovery_archive": runtime_x2["recovery_archive"],
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
            "scratch_root": "/home/henry/sigma-v19x3b-full-regional-spectra/v100",
            "result_root": "results/sigma_v19x3b_v19w5_full_regional_spectral_production",
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
            "quality_pass_definition": "the V19X2 ordered 68-percent temperature interval, fractional half-width, reduced-statistic and free-parameter-bound gates, plus an ordered 68-percent APEC-normalization profile interval required for downstream gas uncertainty",
            "retention_rule": "A finite best fit remains in the gas-map posterior even if its individual quality subgate fails; report that uncertainty and never drop the region.",
        },
        "advance": {
            "if_all_gates_pass": "freeze a separately named V19X4 successor that hashes V19X3B before constructing gas posteriors",
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
            "v19w5_authority_propagated_without_fallback": True,
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
            "V19X3B changes response authority only; it reuses the hash-bound V19X3 regional engine and does not alter scientific fitting rules.",
            "Individual quality failures are retained if the best fit is finite; only the frozen cluster-level minimum determines source-map authorization.",
            "No lensing, inferred halo, source-invariant score, action or gravity parameter is read or selected here.",
        ],
        "implementation": {
            "freezer": relative(Path(__file__)),
            "freezer_sha256": adapter.sha256(Path(__file__)),
            "runner": relative(successor_path),
            "runner_sha256": adapter.sha256(successor_path),
            "inherited_v19x3_runner": relative(inherited_x3_path),
            "inherited_v19x3_runner_sha256": adapter.sha256(inherited_x3_path),
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
    parser.add_argument("--inherited-x3", type=Path, default=DEFAULT_INHERITED_X3)
    parser.add_argument("--successor", type=Path, default=DEFAULT_SUCCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = freeze_config(
        args.v19x2_config.resolve(),
        args.v19x2_runner.resolve(),
        args.v19x2_report.resolve(),
        args.v19h_config.resolve(),
        args.inherited_x3.resolve(),
        args.successor.resolve(),
        args.output.resolve(),
    )
    print(args.output.resolve())
    print(config["status"])


if __name__ == "__main__":
    main()
