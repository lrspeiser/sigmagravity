#!/usr/bin/env python3
"""Freeze V19BMB only after terminal V19X4B gas-grid success."""

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

import run_sigma_v19bmb_v19x4b_stellar_morphology_control as successor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_CONFIG = ROOT / "configs" / "sigma_v19bm_stellar_morphology_control.json"
DEFAULT_ORIGINAL_RUNNER = ROOT / "scripts" / "run_sigma_v19bm_stellar_morphology_control.py"
DEFAULT_STELLAR_MODULE = ROOT / "src" / "voidscreen" / "sigma_stellar_control.py"
DEFAULT_X4B_CONFIG = ROOT / "configs" / "sigma_v19x4b_v19x3b_gas_state_posterior.json"
DEFAULT_X4B_RUNNER = ROOT / "scripts" / "run_sigma_v19x4b_v19x3b_gas_state_posterior.py"
DEFAULT_X4B_FREEZER = ROOT / "scripts" / "freeze_sigma_v19x4b_v19x3b_gas_state_posterior.py"
DEFAULT_X4B_REPORT = ROOT / "results" / "sigma_v19x4b_v19x3b_gas_state_posterior" / "report.json"
DEFAULT_SUCCESSOR = ROOT / "scripts" / "run_sigma_v19bmb_v19x4b_stellar_morphology_control.py"
DEFAULT_OUTPUT = ROOT / "configs" / "sigma_v19bmb_v19x4b_stellar_morphology_control.json"
SCIENCE_SECTIONS = ("clusters", "construction", "future_runtime_gates")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def spec(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": successor.sha256(path)}


def validate_terminal_x4b(
    config_path: Path,
    runner_path: Path,
    freezer_path: Path,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (config_path, runner_path, freezer_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"V19BMB freeze parent is absent: {path}")
    config = load_json(config_path)
    report = load_json(report_path)
    if config.get("freeze_state") != "frozen_after_terminal_v19x3b_pass":
        raise RuntimeError("V19X4B configuration is not terminally frozen")
    for name, path in (("runner", runner_path), ("freezer", freezer_path)):
        if config.get("implementation", {}).get(name) != relative(path):
            raise RuntimeError(f"V19X4B configuration names another {name}")
        if config["implementation"].get(f"{name}_sha256") != successor.sha256(path):
            raise RuntimeError(f"V19X4B {name} changed after freeze")
    if (
        report.get("status") != successor.AUTHORIZED_X4B_STATUS
        or report.get("config_sha256") != successor.sha256(config_path)
        or report.get("runner_sha256") != successor.sha256(runner_path)
        or report.get("source_invariant_scoring_authorized") is not True
        or not report.get("gates")
        or not all(report["gates"].values())
        or report.get("lensing_or_halo_payload_opened") is not False
        or len(report.get("products", [])) != 12
    ):
        raise RuntimeError("V19X4B did not authorize the stellar-grid successor")
    return config, report


def freeze_config(
    original_config_path: Path,
    original_runner_path: Path,
    stellar_module_path: Path,
    x4b_config_path: Path,
    x4b_runner_path: Path,
    x4b_freezer_path: Path,
    x4b_report_path: Path,
    successor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path in (original_config_path, original_runner_path, stellar_module_path, successor_path):
        if not path.is_file():
            raise RuntimeError(f"V19BMB freeze parent is absent: {path}")
    original = load_json(original_config_path)
    validate_terminal_x4b(x4b_config_path, x4b_runner_path, x4b_freezer_path, x4b_report_path)
    parents = {
        name: copy.deepcopy(value)
        for name, value in original["parents"].items()
        if name != "v19x4_config"
    }
    parents.update(
        {
            "original_v19bm_config": spec(original_config_path),
            "v19x4b_config": spec(x4b_config_path),
            "v19x4b_runner": spec(x4b_runner_path),
            "v19x4b_freezer": spec(x4b_freezer_path),
            "v19x4b_report": spec(x4b_report_path),
        }
    )
    config: dict[str, Any] = {
        "protocol_version": "SIGMA-V19BMB-V19X4B-STELLAR-MORPHOLOGY-CONTROL-1.0.0",
        "freeze_state": successor.FROZEN_STATE,
        "status": "frozen mechanically after terminal V19X4B passed and before any observed source score, lensing, halo, gravity parameter or holdout result",
        "purpose": "Run the unchanged V19BM filter-invariant stellar-light percentile control on the exact terminal V19X4B common grids.",
        "parents": parents,
        **{section: copy.deepcopy(original[section]) for section in SCIENCE_SECTIONS},
        "outputs": {
            "root": "results/sigma_v19bmb_v19x4b_stellar_morphology_control",
            "terminal_report": "results/sigma_v19bmb_v19x4b_stellar_morphology_control/report.json",
        },
        "authorization": {
            "run_after_terminal_v19x4b": True,
            "compare_cross_filter_amplitudes": False,
            "infer_stellar_mass": False,
            "read_lensing_or_halo_payload": False,
            "select_action_or_gravity_parameter": False,
            "open_holdout": False,
        },
        "claim_boundary": [
            *original["claim_boundary"],
            "V19BMB changes only the common-grid authority from V19X4 to V19X4B; it does not change the stellar calculation.",
            "A pass authorizes the separately named V19BQ source score and is not evidence for modified gravity.",
        ],
        "implementation": {
            "freezer": relative(Path(__file__)),
            "freezer_sha256": successor.sha256(Path(__file__)),
            "runner": relative(successor_path),
            "runner_sha256": successor.sha256(successor_path),
            "inherited_v19bm_runner": relative(original_runner_path),
            "inherited_v19bm_runner_sha256": successor.sha256(original_runner_path),
            "stellar_module": relative(stellar_module_path),
            "stellar_module_sha256": successor.sha256(stellar_module_path),
        },
    }
    for section in SCIENCE_SECTIONS:
        if config[section] != original[section]:
            raise RuntimeError(f"V19BMB changed science section: {section}")
    successor.validate_static(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-config", type=Path, default=DEFAULT_ORIGINAL_CONFIG)
    parser.add_argument("--original-runner", type=Path, default=DEFAULT_ORIGINAL_RUNNER)
    parser.add_argument("--stellar-module", type=Path, default=DEFAULT_STELLAR_MODULE)
    parser.add_argument("--x4b-config", type=Path, default=DEFAULT_X4B_CONFIG)
    parser.add_argument("--x4b-runner", type=Path, default=DEFAULT_X4B_RUNNER)
    parser.add_argument("--x4b-freezer", type=Path, default=DEFAULT_X4B_FREEZER)
    parser.add_argument("--x4b-report", type=Path, default=DEFAULT_X4B_REPORT)
    parser.add_argument("--successor", type=Path, default=DEFAULT_SUCCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frozen = freeze_config(
        args.original_config.resolve(), args.original_runner.resolve(),
        args.stellar_module.resolve(), args.x4b_config.resolve(),
        args.x4b_runner.resolve(), args.x4b_freezer.resolve(),
        args.x4b_report.resolve(), args.successor.resolve(), args.output.resolve(),
    )
    print(args.output.resolve())
    print(frozen["status"])


if __name__ == "__main__":
    main()
