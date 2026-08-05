#!/usr/bin/env python3
"""Freeze V19X2 only after a terminal, passing V19W5 unified archive exists."""

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

import run_sigma_v19x2_unified_spectral_combination_commissioning as successor
import sigma_v19x2_unified_response_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_CONFIG = (
    ROOT / "configs" / "sigma_v19x_spectral_combination_commissioning.json"
)
DEFAULT_V19W5_CONFIG = ROOT / "configs" / "sigma_v19w5_ccd7_hardened_response_recovery.json"
DEFAULT_V19W5_RUNNER = ROOT / "scripts" / "run_sigma_v19w5_ccd7_hardened_response_recovery.py"
DEFAULT_V19W5_REPORT = (
    ROOT / "results" / "sigma_v19w5_ccd7_hardened_response_recovery" / "report.json"
)
DEFAULT_SUCCESSOR_RUNNER = (
    ROOT / "scripts" / "run_sigma_v19x2_unified_spectral_combination_commissioning.py"
)
DEFAULT_ADAPTER = ROOT / "scripts" / "sigma_v19x2_unified_response_adapter.py"
DEFAULT_OUTPUT = (
    ROOT / "configs" / "sigma_v19x2_unified_spectral_combination_commissioning.json"
)
EXACT_LEGACY_SECTIONS = [
    "registered_workload",
    "combination",
    "fit_sequence",
    "gates",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze_config(
    legacy_config_path: Path,
    v19w5_config_path: Path,
    v19w5_runner_path: Path,
    v19w5_report_path: Path,
    successor_runner_path: Path,
    adapter_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    paths = (
        legacy_config_path,
        v19w5_config_path,
        v19w5_runner_path,
        successor_runner_path,
        adapter_path,
    )
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise RuntimeError(f"V19X2 freeze parent is absent: {missing}")
    v19w5_config_sha = adapter.sha256(v19w5_config_path)
    v19w5_runner_sha = adapter.sha256(v19w5_runner_path)
    terminal, index_path = adapter.authorize_unified_index(
        v19w5_report_path,
        expected_config_sha256=v19w5_config_sha,
        expected_runner_sha256=v19w5_runner_sha,
        expected_cells=5082,
        expected_products=20328,
        root=ROOT,
        expected_status=adapter.V19W5_AUTHORIZED_STATUS,
        authority_label="V19W5",
    )
    legacy = load_json(legacy_config_path)
    parents = copy.deepcopy(legacy["parents"])
    parents.update(
        {
            "legacy_v19x_config": relative(legacy_config_path),
            "legacy_v19x_config_sha256": adapter.sha256(legacy_config_path),
            "v19w5_config": relative(v19w5_config_path),
            "v19w5_config_sha256": v19w5_config_sha,
            "v19w5_runner": relative(v19w5_runner_path),
            "v19w5_runner_sha256": v19w5_runner_sha,
            "v19w5_report": relative(v19w5_report_path),
            "v19w5_report_sha256": adapter.sha256(v19w5_report_path),
            "v19w5_unified_index": relative(index_path),
            "v19w5_unified_index_sha256": adapter.sha256(index_path),
        }
    )
    config: dict[str, Any] = {
        "protocol_version": "SIGMA-V19X2-UNIFIED-SPECTRAL-COMBINATION-COMMISSIONING-1.1.0",
        "freeze_state": "frozen_after_terminal_v19w5_pass",
        "status": "frozen mechanically after V19W5 passed every CCD7-hardened recovery, immutability, unified-index and second full-audit gate; before combining a response, fitting a spectrum or temperature, constructing gas source state, opening lensing or halo data, or changing gravity physics",
        "purpose": "Commission the unchanged V19X integrated and selected-region spectral pipeline against the terminal mixed base/recovery V19W5 archive.",
        "parents": parents,
        "exact_legacy_sections": list(EXACT_LEGACY_SECTIONS),
        "inherited_section_sha256": {
            section: canonical_sha256(legacy[section])
            for section in EXACT_LEGACY_SECTIONS
        },
        "runtime_authorization": {
            "response_authority": "V19W5",
            "required_response_report": relative(v19w5_report_path),
            "required_status": adapter.V19W5_AUTHORIZED_STATUS,
            "recovery_archive": "v19w5_recovery",
            "required_unified_cells": 5082,
            "required_unified_products": 20328,
            "required_unified_index": relative(index_path),
            "required_unified_index_sha256": terminal["unified_product_index"][
                "sha256"
            ],
            "may_start_before_authorization": False,
        },
        "execution": {
            "runner": relative(successor_runner_path),
            "adapter": relative(adapter_path),
            "response_archives": {
                "base_v19w": "/home/henry/sigma-v19w-response-production/v100",
                "v19w5_recovery": "/home/henry/sigma-v19w5-response-recovery/v100",
            },
            "scratch_root": "/home/henry/sigma-v19x2-spectral-combination/v100",
            "result_root": "results/sigma_v19x2_unified_spectral_combination_commissioning",
            "validated_cell_index": "results/sigma_v19x2_unified_spectral_combination_commissioning/validated_cell_index.csv",
            "combination_concurrency": legacy["execution"]["combination_concurrency"],
            "reason_for_serial_combination": legacy["execution"][
                "reason_for_serial_combination"
            ],
        },
        "registered_workload": copy.deepcopy(legacy["registered_workload"]),
        "combination": copy.deepcopy(legacy["combination"]),
        "fit_sequence": copy.deepcopy(legacy["fit_sequence"]),
        "gates": copy.deepcopy(legacy["gates"]),
        "advance": {
            "if_all_gates_pass": "freeze and run all 494 regional combinations and fits with the identical inherited rules",
            "if_any_gate_fails": "report the failed cluster and stage; do not change cell membership, grouping, fit rules, source state, lensing target or gravity physics",
            "gravity_theory_tested_here": False,
        },
        "authorization": {
            "combine_integrated_and_two_selected_regions": True,
            "fit_integrated_abundances_and_two_selected_region_temperatures": True,
            "run_all_494_regional_fits_before_commissioning_passes": False,
            "open_lensing_or_halo_payload": False,
            "change_gravity_formula_or_parameter": False,
        },
        "integrity": {
            "v19w5_terminal_report_existed_at_freeze": True,
            "v19w5_terminal_report_passed_at_freeze": True,
            "unified_index_rows_at_freeze": int(
                terminal["unified_product_index"]["rows"]
            ),
            "unified_product_files_at_freeze": int(terminal["unified_product_files"]),
            "base_archive_modified_by_v19w5": terminal["base_v19w_archive_modified"],
            "obsolete_v19x_authorized": terminal["original_v19x_authorized"],
            "spectrum_or_temperature_known_at_freeze": False,
            "lensing_halo_or_gravity_payload_opened_at_freeze": False,
        },
        "claim_boundary": [
            "A pass commissions spectral combination and fitting; it does not validate Sigma Gravity.",
            "Recovered response cells are engineering replacements selected solely by manifest completeness and independent checkpoint validity.",
            "The exact V19X aperture, grouping, spectral-model, fitting and quality-gate sections are inherited byte-for-structure from the frozen predecessor.",
            "No replacement-cluster lensing, halo target or gravity outcome is read by the freeze step.",
        ],
        "implementation": {
            "freezer": relative(Path(__file__)),
            "freezer_sha256": adapter.sha256(Path(__file__)),
            "runner": relative(successor_runner_path),
            "runner_sha256": adapter.sha256(successor_runner_path),
            "adapter": relative(adapter_path),
            "adapter_sha256": adapter.sha256(adapter_path),
        },
    }
    for section in EXACT_LEGACY_SECTIONS:
        if config[section] != legacy[section]:
            raise RuntimeError(f"V19X2 freeze changed inherited section: {section}")
    successor.validate_frozen_runner(config)
    successor.validate_frozen_parents_and_inheritance(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-config", type=Path, default=DEFAULT_LEGACY_CONFIG)
    parser.add_argument("--v19w5-config", type=Path, default=DEFAULT_V19W5_CONFIG)
    parser.add_argument("--v19w5-runner", type=Path, default=DEFAULT_V19W5_RUNNER)
    parser.add_argument("--v19w5-report", type=Path, default=DEFAULT_V19W5_REPORT)
    parser.add_argument("--successor-runner", type=Path, default=DEFAULT_SUCCESSOR_RUNNER)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = freeze_config(
        args.legacy_config.resolve(),
        args.v19w5_config.resolve(),
        args.v19w5_runner.resolve(),
        args.v19w5_report.resolve(),
        args.successor_runner.resolve(),
        args.adapter.resolve(),
        args.output.resolve(),
    )
    print(args.output.resolve())
    print(config["status"])


if __name__ == "__main__":
    main()
