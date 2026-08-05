#!/usr/bin/env python3
"""Audit the V19BR terminal-chain driver before terminal source results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19br_target_sealed_terminal_chain as runner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19br_target_sealed_terminal_chain.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "preflight_report.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def execute(config: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    runner.validate_static(config)
    preflight_reports = [
        load_json(ROOT / config["parents"][name]["path"])
        for name in (
            "v19x3b_preflight_report",
            "v19x4b_preflight_report",
            "v19bmb_preflight_report",
            "v19bq_preflight_report",
        )
    ]
    stage_ids = [stage["id"] for stage in config["stages"]]
    expected_ids = [
        "V19W5_RESPONSE_RECOVERY",
        "FREEZE_V19X2",
        "RUN_V19X2",
        "FREEZE_V19X3B",
        "RUN_V19X3B",
        "FREEZE_V19X4B",
        "RUN_V19X4B",
        "FREEZE_V19BMB",
        "RUN_V19BMB",
        "FREEZE_V19BQ",
        "RUN_V19BQ_SOURCE_DECISION",
    ]
    gates = {
        "all_parent_and_implementation_hashes_exact": True,
        "all_successor_preflights_pass": all(
            report.get("gates") and all(report["gates"].values())
            for report in preflight_reports
        ),
        "eleven_stage_order_is_exact": stage_ids == expected_ids,
        "terminal_products_are_never_retried_after_failure": all(
            stage.get("failure_artifacts")
            for stage in config["stages"]
            if stage["artifact_kind"] == "report"
        ),
        "every_upstream_science_report_requires_all_gates": all(
            stage.get("require_all_gates")
            for stage in config["stages"][:-1]
            if stage["artifact_kind"] == "report"
        ),
        "final_source_stage_accepts_pass_or_falsification_not_execution_failure": (
            config["stages"][-1].get("expected_values")
            == [
                "observed_source_invariant_gates_passed_action_derivation_authorized",
                "observed_source_invariant_gates_failed_no_action_authorized",
            ]
            and not config["stages"][-1].get("require_all_gates", False)
            and config["stages"][-1].get("required_keys")
            == ["aggregate_decision"]
        ),
        "source_result_is_last_and_no_lensing_stage_exists": (
            stage_ids[-1] == "RUN_V19BQ_SOURCE_DECISION"
            and not any(
                token in stage_id.lower()
                for stage_id in stage_ids
                for token in ("lensing", "halo", "gravity_fit", "action")
            )
        ),
        "base_exit_and_target_seals_required": (
            config["authorization"]["execute_only_after_base_process_exits"]
            and not config["authorization"]["open_lensing_or_halo_payload"]
            and not config["authorization"]["derive_action"]
            and not config["authorization"]["change_gravity_formula_or_parameter"]
            and not config["authorization"]["open_holdout"]
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": (
            "v19br_target_sealed_terminal_chain_preflight_passed"
            if all(gates.values())
            else "v19br_target_sealed_terminal_chain_preflight_failed"
        ),
        "config_sha256": runner.sha256(config_path),
        "runner_sha256": runner.sha256(ROOT / config["implementation"]["runner"]["path"]),
        "gates": gates,
        "terminal_gas_stellar_or_source_result_opened": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    report = execute(load_json(config_path), config_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(report["status"])
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
