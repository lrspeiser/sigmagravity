#!/usr/bin/env python3
"""Apply the frozen pass/falsification disposition to terminal V19BQ."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19bs_source_disposition.json"
PASS_STATUS = "observed_source_invariant_gates_passed_action_derivation_authorized"
FAIL_STATUS = "observed_source_invariant_gates_failed_no_action_authorized"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_static(config: dict[str, Any]) -> None:
    if config.get("freeze_state") != "frozen_before_terminal_v19bq_source_result":
        raise RuntimeError("V19BS disposition is not frozen before V19BQ")
    for section in ("parents", "implementation"):
        for name, spec in config[section].items():
            path = ROOT / spec["path"]
            if not path.is_file() or sha256(path) != spec["sha256"]:
                raise RuntimeError(f"V19BS {section[:-1]} changed: {name}")
    if (ROOT / config["implementation"]["runner"]["path"]).resolve() != Path(
        __file__
    ).resolve():
        raise RuntimeError("V19BS configuration names another runner")


def validate_terminal_bq(
    config: dict[str, Any], bq_config_path: Path, bq_report_path: Path
) -> dict[str, Any]:
    if not bq_config_path.is_file() or not bq_report_path.is_file():
        raise RuntimeError("V19BS requires terminal V19BQ config and report")
    bq_config = load_json(bq_config_path)
    report = load_json(bq_report_path)
    expected_runner = config["parents"]["v19bq_runner"]["sha256"]
    if (
        bq_config.get("freeze_state")
        != "frozen_after_terminal_v19x4b_and_v19bmb_pass"
        or bq_config.get("implementation", {}).get("runner_sha256")
        != expected_runner
        or report.get("config_sha256") != sha256(bq_config_path)
        or report.get("runner_sha256") != expected_runner
        or report.get("status") not in {PASS_STATUS, FAIL_STATUS}
        or "aggregate_decision" not in report
        or report.get("lensing_halo_action_or_gravity_payload_opened") is not False
        or report.get("gravity_formula_or_parameter_changed") is not False
    ):
        raise RuntimeError("V19BS received invalid or target-opened V19BQ evidence")
    authorized = report["aggregate_decision"].get("action_derivation_authorized")
    if (report["status"] == PASS_STATUS) != (authorized is True):
        raise RuntimeError("V19BQ status and aggregate decision disagree")
    return report


def disposition_from_decision(
    decision: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    direction = decision.get("I4_direction_pass") is True
    i4_amplitude = decision.get("I4_amplitude_pass") is True
    i5_scalar = decision.get("I5_scalar_pass") is True
    authorized = decision.get("action_derivation_authorized") is True
    logical_authorization = direction and (i4_amplitude or i5_scalar)
    if authorized != logical_authorization:
        raise RuntimeError("V19BS source decision violates the frozen I4/I5 logic")
    if not authorized:
        return {
            "status": "source_gate_falsified_no_action_authorized",
            "source_terms_authorized": [],
            "compatible_action_placement_classes": [],
            "excluded_action_placement_classes": [
                row["id"] for row in config["action_placement_classes"]
            ],
            "next_evidence": config["failure_route"]["required_next_evidence"],
            "action_derivation_authorized": False,
        }
    source_terms = ["I4_THERMODYNAMIC_GRADIENT_STRESS_DIRECTION"]
    if i4_amplitude:
        source_terms.append("I4_THERMODYNAMIC_GRADIENT_STRESS_AMPLITUDE")
    if i5_scalar:
        source_terms.append("I5_BAROCLINICITY_SCALAR")
    compatible = [
        row
        for row in config["action_placement_classes"]
        if row["id"]
        in {"P1_CONSTRAINED_COMPOSITE_RESPONSE", "P3_DEGENERATE_PURE_METRIC_NONLINEAR_VERTEX"}
    ]
    return {
        "status": "source_gate_passed_mathematical_action_comparison_authorized",
        "source_terms_authorized": source_terms,
        "compatible_action_placement_classes": compatible,
        "excluded_action_placement_classes": ["P2_CAUSAL_DYNAMIC_RESPONSE"],
        "exclusion_reason": (
            "I4 and I5 are time-even snapshot sources; V19BQ contains no directly "
            "clocked lag or time-odd current that could authorize a propagating-memory route."
        ),
        "selection_rule": config["pass_route"]["selection_rule"],
        "maximum_distinct_action_derivations": config["pass_route"][
            "maximum_distinct_action_derivations"
        ],
        "action_derivation_authorized": True,
    }


def execute(
    config: dict[str, Any], bq_config_path: Path, bq_report_path: Path
) -> dict[str, Any]:
    validate_static(config)
    report = validate_terminal_bq(config, bq_config_path, bq_report_path)
    result = disposition_from_decision(report["aggregate_decision"], config)
    return {
        **result,
        "v19bq_config_sha256": sha256(bq_config_path),
        "v19bq_report_sha256": sha256(bq_report_path),
        "lensing_halo_galaxy_action_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v19bq-config", type=Path, required=True)
    parser.add_argument("--v19bq-report", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    try:
        result = execute(
            config, args.v19bq_config.resolve(), args.v19bq_report.resolve()
        )
    except Exception as exc:  # noqa: BLE001 - retain terminal disposition failure
        result = {
            "status": "source_disposition_execution_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}",
            "action_derivation_authorized": False,
            "lensing_halo_galaxy_action_or_holdout_payload_opened": False,
            "gravity_formula_or_parameter_changed": False,
        }
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["outputs"]["terminal_report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report["status"])
    if report["status"] == "source_disposition_execution_failed_closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
