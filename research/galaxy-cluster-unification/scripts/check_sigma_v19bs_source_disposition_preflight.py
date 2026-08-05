#!/usr/bin/env python3
"""Audit V19BS before any terminal V19BQ source result exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19bs_source_disposition as runner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bs_source_disposition.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19bs_source_disposition" / "preflight_report.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def execute(config: dict[str, Any], config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    runner.validate_static(config)
    v19bj = load_json(ROOT / config["parents"]["v19bj_config"]["path"])
    v19bj_report = load_json(ROOT / config["parents"]["v19bj_report"]["path"])
    bq_preflight = load_json(ROOT / config["parents"]["v19bq_preflight_report"]["path"])
    placement_ids = [row["id"] for row in config["action_placement_classes"]]
    gates = {
        "all_parent_and_implementation_hashes_exact": True,
        "v19bj_target_blind_preselection_passes": (
            v19bj_report.get("decision") == "passed_target_blind_source_preselection_freeze"
            and v19bj_report.get("gate_results")
            and all(v19bj_report["gate_results"].values())
        ),
        "all_three_v19bj_placement_classes_copied": (
            placement_ids == [row["id"] for row in v19bj["action_placement_classes"]]
        ),
        "v19bq_preflight_passes": (
            bq_preflight.get("status") == "v19bq_observed_source_successor_preflight_passed"
            and bq_preflight.get("gates")
            and all(bq_preflight["gates"].values())
        ),
        "scientific_failure_forbids_every_action_class": (
            config["failure_route"]["authorized_action_placement_classes"] == []
            and config["failure_route"]["action_derivation_authorized"] is False
        ),
        "time_even_pass_excludes_unmeasured_dynamic_route": (
            config["pass_route"]["excluded_without_time_odd_evidence"]
            == ["P2_CAUSAL_DYNAMIC_RESPONSE"]
        ),
        "mathematics_not_lensing_selects_compatible_action": (
            "constraint" in config["pass_route"]["selection_rule"].lower()
            and "stability" in config["pass_route"]["selection_rule"].lower()
            and "lensing" not in config["pass_route"]["selection_rule"].lower()
        ),
        "no_result_action_formula_target_or_holdout_opened": not any(
            config["authorization"][name]
            for name in (
                "run_before_terminal_v19bq",
                "select_action_now",
                "write_gravity_formula_now",
                "open_lensing_or_halo_payload",
                "open_galaxy_rotation_payload",
                "open_holdout",
            )
        ),
    }
    return {
        "protocol_version": config["protocol_version"],
        "status": (
            "v19bs_source_disposition_preflight_passed"
            if all(gates.values())
            else "v19bs_source_disposition_preflight_failed"
        ),
        "config_sha256": runner.sha256(config_path),
        "runner_sha256": runner.sha256(ROOT / config["implementation"]["runner"]["path"]),
        "gates": gates,
        "terminal_v19bq_result_opened": False,
        "lensing_halo_galaxy_action_gravity_or_holdout_payload_opened": False,
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
