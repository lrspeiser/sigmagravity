#!/usr/bin/env python3
"""Repeat V19DO without its invalid source/background BACKSCAL equality check."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19do_observation_resolved_soft_background_audit as v19do

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen(
    config: dict[str, Any], runner: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19DO2 runner changed after freeze")
    for name, item in config["parents"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"V19DO2 parent changed: {name}")
    parent_config_path = ROOT / config["parents"]["v19do_config"]["path"]
    parent_runner = ROOT / config["parents"]["v19do_runner"]["path"]
    parent_config = load_json(parent_config_path)
    v19dl_report, v19dn_report = v19do.validate_frozen(parent_config, parent_runner)
    parent_report = load_json(ROOT / config["parents"]["v19do_report"]["path"])
    if parent_report["status"] != (
        "observation_resolved_soft_background_audit_execution_failed"
    ):
        raise RuntimeError("V19DO no longer retains its registered execution failure")
    expected = "RuntimeError: V19DO BACKSCAL mismatch:"
    if not str(parent_report.get("execution_exception", "")).startswith(expected):
        raise RuntimeError("V19DO failure is not the registered BACKSCAL assertion")
    source = parent_runner.read_text(encoding="utf-8")
    if source.count("abs(") != 1 or "BACKSCAL mismatch" not in source:
        raise RuntimeError("V19DO invalid equality assertion changed")
    return parent_config, v19dl_report, v19dn_report


def execute(
    parent_config: dict[str, Any],
    v19dl_report: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    # V19DO uses abs() exactly once: in the invalid equality assertion. Replacing
    # that call by zero disables only the assertion; the independently correct
    # exposure*BACKSCAL*AREASCAL ratio remains unchanged in every count scale.
    v19do.abs = lambda _: 0.0
    try:
        result = v19do.execute(parent_config, v19dl_report, output)
    finally:
        del v19do.abs
    passed = bool(result["aggregate_pass"])
    result["status"] = (
        "backscal_ratio_observation_soft_background_audit_completed"
        if passed
        else "backscal_ratio_observation_soft_background_audit_gate_failed"
    )
    result["v19do_execution_result_scientifically_discarded"] = True
    result["remediation_scope"] = (
        "Removed only the invalid source/background BACKSCAL equality assertion; "
        "effective scaling still uses the frozen exposure, BACKSCAL and AREASCAL ratio."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = load_json(config_path)
        parent_config, v19dl_report, v19dn_report = validate_frozen(
            config, Path(__file__).resolve()
        )
        result = execute(parent_config, v19dl_report, output)
        result["discarded_v19do_report_sha256"] = sha256(
            ROOT / config["parents"]["v19do_report"]["path"]
        )
        result["v19dn_status"] = v19dn_report["status"]
    except Exception as exc:  # noqa: BLE001 - preserve terminal audit evidence
        result = {
            "status": "backscal_ratio_observation_soft_background_execution_failed",
            "execution_exception": f"{type(exc).__name__}: {exc}",
            "aggregate_pass": False,
            "joint_likelihood_or_full_regional_successor_authorized": False,
        }
    report = {
        "protocol_version": "SIGMA-V19DO2-BACKSCAL-RATIO-1.0.0",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        **result,
        "additional_plasma_component_admitted": False,
        "thermal_stress_or_baroclinicity_constructed": False,
        "lensing_halo_action_gravity_or_holdout_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(f"status: {report['status']}")
    if not report["aggregate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
