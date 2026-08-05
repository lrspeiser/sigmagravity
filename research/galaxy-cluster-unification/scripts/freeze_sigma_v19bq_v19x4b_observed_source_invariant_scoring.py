#!/usr/bin/env python3
"""Freeze V19BQ after terminal V19X4B and V19BMB success."""

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

import run_sigma_v19bmb_v19x4b_stellar_morphology_control as v19bmb
import run_sigma_v19bq_v19x4b_observed_source_invariant_scoring as successor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_CONFIG = ROOT / "configs" / "sigma_v19bp_observed_source_invariant_scoring.json"
DEFAULT_ORIGINAL_RUNNER = ROOT / "scripts" / "run_sigma_v19bp_observed_source_invariant_scoring.py"
DEFAULT_GAS_MODULE = ROOT / "src" / "voidscreen" / "sigma_gas_source_stream.py"
DEFAULT_SCORE_MODULE = ROOT / "src" / "voidscreen" / "sigma_source_score_engine.py"
DEFAULT_X4B_CONFIG = ROOT / "configs" / "sigma_v19x4b_v19x3b_gas_state_posterior.json"
DEFAULT_X4B_RUNNER = ROOT / "scripts" / "run_sigma_v19x4b_v19x3b_gas_state_posterior.py"
DEFAULT_X4B_REPORT = ROOT / "results" / "sigma_v19x4b_v19x3b_gas_state_posterior" / "report.json"
DEFAULT_BMB_CONFIG = ROOT / "configs" / "sigma_v19bmb_v19x4b_stellar_morphology_control.json"
DEFAULT_BMB_RUNNER = ROOT / "scripts" / "run_sigma_v19bmb_v19x4b_stellar_morphology_control.py"
DEFAULT_BMB_REPORT = ROOT / "results" / "sigma_v19bmb_v19x4b_stellar_morphology_control" / "report.json"
DEFAULT_SUCCESSOR = ROOT / "scripts" / "run_sigma_v19bq_v19x4b_observed_source_invariant_scoring.py"
DEFAULT_OUTPUT = ROOT / "configs" / "sigma_v19bq_v19x4b_observed_source_invariant_scoring.json"
SCIENCE_SECTIONS = (
    "registered_inputs",
    "variants",
    "thresholds",
    "inherited_v19bl_thresholds",
    "decision_rule",
    "execution",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_terminal(
    config_path: Path,
    runner_path: Path,
    report_path: Path,
    *,
    freeze_state: str,
    status: str,
    product_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (config_path, runner_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"V19BQ freeze parent is absent: {path}")
    config = load_json(config_path)
    report = load_json(report_path)
    if config.get("freeze_state") != freeze_state:
        raise RuntimeError(f"V19BQ parent is not frozen: {config_path}")
    if (
        config.get("implementation", {}).get("runner") != relative(runner_path)
        or config["implementation"].get("runner_sha256") != successor.sha256(runner_path)
    ):
        raise RuntimeError(f"V19BQ parent runner changed: {runner_path}")
    if (
        report.get("status") != status
        or report.get("config_sha256") != successor.sha256(config_path)
        or report.get("runner_sha256") != successor.sha256(runner_path)
        or not report.get("gates")
        or not all(report["gates"].values())
        or len(report.get("products", [])) != product_count
    ):
        raise RuntimeError(f"V19BQ terminal parent failed: {report_path}")
    return config, report


def freeze_config(
    original_config_path: Path,
    original_runner_path: Path,
    gas_module_path: Path,
    score_module_path: Path,
    x4b_config_path: Path,
    x4b_runner_path: Path,
    x4b_report_path: Path,
    bmb_config_path: Path,
    bmb_runner_path: Path,
    bmb_report_path: Path,
    successor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path in (
        original_config_path,
        original_runner_path,
        gas_module_path,
        score_module_path,
        successor_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"V19BQ freeze parent is absent: {path}")
    original = load_json(original_config_path)
    _, x4b_report = validate_terminal(
        x4b_config_path,
        x4b_runner_path,
        x4b_report_path,
        freeze_state="frozen_after_terminal_v19x3b_pass",
        status=successor.AUTHORIZED_X4B_STATUS,
        product_count=12,
    )
    if (
        x4b_report.get("source_invariant_scoring_authorized") is not True
        or x4b_report.get("lensing_or_halo_payload_opened") is not False
    ):
        raise RuntimeError("V19X4B did not preserve the source authorization seal")
    _, bmb_report = validate_terminal(
        bmb_config_path,
        bmb_runner_path,
        bmb_report_path,
        freeze_state=v19bmb.FROZEN_STATE,
        status="stellar_morphology_control_passed_invariant_scoring_ready",
        product_count=2,
    )
    if (
        bmb_report.get("invariant_scoring_ready") is not True
        or bmb_report.get("lensing_halo_action_or_gravity_payload_opened") is not False
    ):
        raise RuntimeError("V19BMB did not preserve the target seal")
    parents: dict[str, str] = {}
    for name, value in original["parents"].items():
        if name not in {"v19bm_config", "v19bm_preflight_report", "v19x4_config"}:
            parents[name] = value["path"]
            parents[f"{name}_sha256"] = value["sha256"]
    additions = {
        "original_v19bp_config": original_config_path,
        "v19x4b_config": x4b_config_path,
        "v19x4b_runner": x4b_runner_path,
        "v19x4b_report": x4b_report_path,
        "v19bmb_config": bmb_config_path,
        "v19bmb_runner": bmb_runner_path,
        "v19bmb_report": bmb_report_path,
    }
    for name, path in additions.items():
        parents[name] = relative(path)
        parents[f"{name}_sha256"] = successor.sha256(path)
    config: dict[str, Any] = {
        "protocol_version": "SIGMA-V19BQ-V19X4B-OBSERVED-SOURCE-INVARIANT-SCORING-1.0.0",
        "freeze_state": successor.FROZEN_STATE,
        "status": "frozen mechanically after terminal V19X4B and V19BMB passed, before observed source scoring, lensing, halo, action, gravity parameter or holdout access",
        "purpose": "Apply the unchanged V19BP I4/I5 source-only decision to the terminal V19W5-authorized gas and stellar products.",
        "parents": parents,
        "terminal_authorization": {
            "required_v19x4b_status": successor.AUTHORIZED_X4B_STATUS,
            "required_v19bmb_status": "stellar_morphology_control_passed_invariant_scoring_ready",
            "require_every_report_gate": True,
            "require_every_product_size_and_sha256": True,
            "require_target_sealed_reports": True,
        },
        "exact_science_section_sha256": {
            section: canonical_sha256(original[section]) for section in SCIENCE_SECTIONS
        },
        **{section: copy.deepcopy(original[section]) for section in SCIENCE_SECTIONS},
        "outputs": {
            "root": "results/sigma_v19bq_v19x4b_observed_source_invariant_scoring",
            "terminal_report": "results/sigma_v19bq_v19x4b_observed_source_invariant_scoring/report.json",
            "branch_products": "one hash-bound source-only NPZ per cluster and rank-correlation branch",
        },
        "authorization": {
            "run_after_terminal_v19x4b_and_v19bmb": True,
            "read_lensing_or_halo_payload": False,
            "derive_action_now": False,
            "change_gravity_formula_or_parameter": False,
            "open_holdout": False,
        },
        "claim_boundary": [
            *original["claim_boundary"],
            "V19BQ changes only terminal data authority; all I4/I5 candidates, variants, thresholds and decisions remain identical to V19BP.",
            "The terminal source result can reject or authorize action derivation but cannot establish a gravity or lensing law.",
        ],
        "implementation": {
            "freezer": relative(Path(__file__)),
            "freezer_sha256": successor.sha256(Path(__file__)),
            "runner": relative(successor_path),
            "runner_sha256": successor.sha256(successor_path),
            "inherited_v19bp_runner": relative(original_runner_path),
            "inherited_v19bp_runner_sha256": successor.sha256(original_runner_path),
            "gas_stream_module": relative(gas_module_path),
            "gas_stream_module_sha256": successor.sha256(gas_module_path),
            "score_engine_module": relative(score_module_path),
            "score_engine_module_sha256": successor.sha256(score_module_path),
        },
    }
    for section in SCIENCE_SECTIONS:
        if config[section] != original[section]:
            raise RuntimeError(f"V19BQ changed science section: {section}")
    successor.validate_static(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-config", type=Path, default=DEFAULT_ORIGINAL_CONFIG)
    parser.add_argument("--original-runner", type=Path, default=DEFAULT_ORIGINAL_RUNNER)
    parser.add_argument("--gas-module", type=Path, default=DEFAULT_GAS_MODULE)
    parser.add_argument("--score-module", type=Path, default=DEFAULT_SCORE_MODULE)
    parser.add_argument("--x4b-config", type=Path, default=DEFAULT_X4B_CONFIG)
    parser.add_argument("--x4b-runner", type=Path, default=DEFAULT_X4B_RUNNER)
    parser.add_argument("--x4b-report", type=Path, default=DEFAULT_X4B_REPORT)
    parser.add_argument("--bmb-config", type=Path, default=DEFAULT_BMB_CONFIG)
    parser.add_argument("--bmb-runner", type=Path, default=DEFAULT_BMB_RUNNER)
    parser.add_argument("--bmb-report", type=Path, default=DEFAULT_BMB_REPORT)
    parser.add_argument("--successor", type=Path, default=DEFAULT_SUCCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frozen = freeze_config(
        args.original_config.resolve(), args.original_runner.resolve(),
        args.gas_module.resolve(), args.score_module.resolve(),
        args.x4b_config.resolve(), args.x4b_runner.resolve(), args.x4b_report.resolve(),
        args.bmb_config.resolve(), args.bmb_runner.resolve(), args.bmb_report.resolve(),
        args.successor.resolve(), args.output.resolve(),
    )
    print(args.output.resolve())
    print(frozen["status"])


if __name__ == "__main__":
    main()
