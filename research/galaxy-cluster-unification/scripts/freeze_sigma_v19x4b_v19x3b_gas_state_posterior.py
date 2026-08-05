#!/usr/bin/env python3
"""Freeze the V19X4B gas posterior only after terminal V19X3B success."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x3b_v19w5_full_regional_spectral_production as v19x3b
import run_sigma_v19x4b_v19x3b_gas_state_posterior as successor
import sigma_v19x2_unified_response_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_CONFIG = (
    ROOT / "configs" / "sigma_v19x4_gas_state_math_preflight.json"
)
DEFAULT_X3B_CONFIG = (
    ROOT / "configs" / "sigma_v19x3b_v19w5_full_regional_spectral_production.json"
)
DEFAULT_X3B_RUNNER = (
    ROOT / "scripts" / "run_sigma_v19x3b_v19w5_full_regional_spectral_production.py"
)
DEFAULT_X3B_FREEZER = (
    ROOT / "scripts" / "freeze_sigma_v19x3b_v19w5_full_regional_spectral_production.py"
)
DEFAULT_X3B_REPORT = (
    ROOT
    / "results"
    / "sigma_v19x3b_v19w5_full_regional_spectral_production"
    / "report.json"
)
DEFAULT_INHERITED_X4_RUNNER = (
    ROOT / "scripts" / "run_sigma_v19x4_gas_state_posterior.py"
)
DEFAULT_POSTERIOR_MODULE = ROOT / "src" / "voidscreen" / "sigma_gas_posterior.py"
DEFAULT_SUCCESSOR = (
    ROOT / "scripts" / "run_sigma_v19x4b_v19x3b_gas_state_posterior.py"
)
DEFAULT_OUTPUT = (
    ROOT / "configs" / "sigma_v19x4b_v19x3b_gas_state_posterior.json"
)
SCIENCE_SECTIONS = (
    "official_definition",
    "algebra_correction",
    "physical_constants_and_composition",
    "geometry",
    "posterior",
    "common_grid",
    "front_state",
    "future_runtime_gates",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_terminal_x3b(
    config_path: Path,
    runner_path: Path,
    freezer_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (config_path, runner_path, freezer_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"V19X4B freeze parent is absent: {path}")
    config = load_json(config_path)
    report = load_json(report_path)
    implementation = config.get("implementation", {})
    if config.get("freeze_state") != v19x3b.FROZEN_STATE:
        raise RuntimeError("V19X3B configuration is not terminally frozen")
    for name, path in (("runner", runner_path), ("freezer", freezer_path)):
        if implementation.get(name) != relative(path):
            raise RuntimeError(f"V19X3B configuration names another {name}")
        if implementation.get(f"{name}_sha256") != adapter.sha256(path):
            raise RuntimeError(f"V19X3B {name} changed after freeze")
    if report.get("status") != successor.AUTHORIZED_X3B_STATUS:
        raise RuntimeError("V19X3B report did not authorize gas reconstruction")
    if report.get("config_sha256") != adapter.sha256(config_path):
        raise RuntimeError("V19X3B report names another config")
    if report.get("runner_sha256") != adapter.sha256(runner_path):
        raise RuntimeError("V19X3B report names another runner")
    if not report.get("gates") or not all(report["gates"].values()):
        raise RuntimeError("V19X3B report contains a failed gate")
    if report.get("source_map_construction_authorized") is not True:
        raise RuntimeError("V19X3B withheld gas-source authorization")
    if report.get("lensing_or_halo_payload_opened") is not False:
        raise RuntimeError("V19X3B opened a prohibited target")
    regions = report.get("regions", [])
    if len(regions) != 494 or {row.get("cluster") for row in regions} != {
        "BULLET",
        "ABELL2146",
    }:
        raise RuntimeError("V19X3B regional inventory changed")
    return config, report


def freeze_config(
    original_config_path: Path,
    x3b_config_path: Path,
    x3b_runner_path: Path,
    x3b_freezer_path: Path,
    x3b_report_path: Path,
    inherited_x4_runner_path: Path,
    posterior_module_path: Path,
    successor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path in (
        original_config_path,
        inherited_x4_runner_path,
        posterior_module_path,
        successor_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"V19X4B freeze parent is absent: {path}")
    original = load_json(original_config_path)
    x3b_config, _ = validate_terminal_x3b(
        x3b_config_path, x3b_runner_path, x3b_freezer_path, x3b_report_path
    )
    for section in SCIENCE_SECTIONS:
        if section not in original:
            raise RuntimeError(f"V19X4B missing original science section: {section}")

    parents = {
        key: value
        for key, value in original["parents"].items()
        if not key.startswith("v19x3_")
    }
    parents.update(
        {
            "original_v19x4_config": relative(original_config_path),
            "original_v19x4_config_sha256": adapter.sha256(original_config_path),
            "v19x3b_config": relative(x3b_config_path),
            "v19x3b_config_sha256": adapter.sha256(x3b_config_path),
            "v19x3b_runner": relative(x3b_runner_path),
            "v19x3b_runner_sha256": adapter.sha256(x3b_runner_path),
            "v19x3b_freezer": relative(x3b_freezer_path),
            "v19x3b_freezer_sha256": adapter.sha256(x3b_freezer_path),
            "v19x3b_report": relative(x3b_report_path),
            "v19x3b_report_sha256": adapter.sha256(x3b_report_path),
        }
    )
    config: dict[str, Any] = {
        "protocol_version": "SIGMA-V19X4B-V19X3B-GAS-STATE-POSTERIOR-1.0.0",
        "freeze_state": successor.FROZEN_STATE,
        "status": "frozen mechanically after all V19X3B regional spectra passed, before constructing any gas posterior, source invariant, lensing or halo target, gravity parameter or holdout result",
        "purpose": "Run the unchanged V19X4 gas-state and common-grid mathematics on the terminal V19W5-authorized V19X3B regional measurements.",
        "parents": parents,
        "runtime_authorization": {
            "required_v19x3b_config": relative(x3b_config_path),
            "required_v19x3b_report": relative(x3b_report_path),
            "required_v19x3b_config_sha256": adapter.sha256(x3b_config_path),
            "required_v19x3b_runner_sha256": adapter.sha256(x3b_runner_path),
            "required_v19x3b_status": successor.AUTHORIZED_X3B_STATUS,
            "v19w5_authority_inherited": x3b_config["runtime_authorization"][
                "response_authority"
            ]
            == "V19W5",
            "may_start_before_v19x3b_pass": False,
        },
        "exact_science_section_sha256": {
            section: canonical_sha256(original[section]) for section in SCIENCE_SECTIONS
        },
        **{section: copy.deepcopy(original[section]) for section in SCIENCE_SECTIONS},
        "authorization": {
            "construct_observed_gas_map_after_v19x3b_pass": True,
            "open_lensing_or_halo_payload": False,
            "select_source_invariant_or_action": False,
            "fit_gravity_parameter": False,
            "open_holdout": False,
        },
        "outputs": {
            "result_root": "results/sigma_v19x4b_v19x3b_gas_state_posterior",
            "regional_products": "one hash-bound NPZ per cluster and temperature-normalization dependence branch",
            "common_grid_products": "one hash-bound 241 by 241 summary-map NPZ per cluster and dependence branch",
        },
        "integrity": {
            "v19x3b_report_passed_at_freeze": True,
            "v19w5_authority_inherited_without_fallback": True,
            "gas_posterior_known_at_freeze": False,
            "source_invariant_or_target_opened_at_freeze": False,
        },
        "claim_boundary": [
            *original["claim_boundary"],
            "V19X4B changes only the terminal regional-data authority; every gas conversion, uncertainty and grid rule is copied exactly from V19X4.",
            "A pass authorizes a separately named source-score successor and is not evidence for Sigma Gravity.",
        ],
        "implementation": {
            "freezer": relative(Path(__file__)),
            "freezer_sha256": adapter.sha256(Path(__file__)),
            "runner": relative(successor_path),
            "runner_sha256": adapter.sha256(successor_path),
            "inherited_v19x4_runner": relative(inherited_x4_runner_path),
            "inherited_v19x4_runner_sha256": adapter.sha256(
                inherited_x4_runner_path
            ),
            "posterior_module": relative(posterior_module_path),
            "posterior_module_sha256": adapter.sha256(posterior_module_path),
        },
    }
    for section in SCIENCE_SECTIONS:
        if config[section] != original[section]:
            raise RuntimeError(f"V19X4B changed science section: {section}")
    successor.validate_frozen_runner(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-config", type=Path, default=DEFAULT_ORIGINAL_CONFIG)
    parser.add_argument("--v19x3b-config", type=Path, default=DEFAULT_X3B_CONFIG)
    parser.add_argument("--v19x3b-runner", type=Path, default=DEFAULT_X3B_RUNNER)
    parser.add_argument("--v19x3b-freezer", type=Path, default=DEFAULT_X3B_FREEZER)
    parser.add_argument("--v19x3b-report", type=Path, default=DEFAULT_X3B_REPORT)
    parser.add_argument(
        "--inherited-x4-runner", type=Path, default=DEFAULT_INHERITED_X4_RUNNER
    )
    parser.add_argument("--posterior-module", type=Path, default=DEFAULT_POSTERIOR_MODULE)
    parser.add_argument("--successor", type=Path, default=DEFAULT_SUCCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = freeze_config(
        args.original_config.resolve(),
        args.v19x3b_config.resolve(),
        args.v19x3b_runner.resolve(),
        args.v19x3b_freezer.resolve(),
        args.v19x3b_report.resolve(),
        args.inherited_x4_runner.resolve(),
        args.posterior_module.resolve(),
        args.successor.resolve(),
        args.output.resolve(),
    )
    print(args.output.resolve())
    print(config["status"])


if __name__ == "__main__":
    main()
