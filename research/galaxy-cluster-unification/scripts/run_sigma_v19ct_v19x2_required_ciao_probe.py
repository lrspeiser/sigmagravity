#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import run_sigma_v19cs_v19x2_independent_ciao_probe as v19cs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ct_v19x2_required_ciao_probe.json"


def validate_and_build(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = config["parents"]
    cs_config_path = ROOT / parents["v19cs_config"]["path"]
    cs_runner_path = ROOT / parents["v19cs_runner"]["path"]
    cs_report_path = ROOT / parents["v19cs_failure_report"]["path"]
    cs_log_path = ROOT / parents["v19cs_probe_log"]["path"]
    cs_report = v19cs.load_json(cs_report_path)
    checks = {
        "v19cs_config_exact": v19cs.sha256(cs_config_path) == parents["v19cs_config"]["sha256"],
        "v19cs_runner_exact": v19cs.sha256(cs_runner_path) == parents["v19cs_runner"]["sha256"],
        "v19cs_failure_exact": (
            v19cs.sha256(cs_report_path) == parents["v19cs_failure_report"]["sha256"]
            and cs_report["status"] == parents["v19cs_failure_report"]["required_status"]
            and parents["v19cs_failure_report"]["required_exception_fragment"] in cs_report["exception"]
        ),
        "v19cs_probe_log_exact": (
            v19cs.sha256(cs_log_path) == parents["v19cs_probe_log"]["sha256"]
            and parents["v19cs_probe_log"]["required_text"] in cs_log_path.read_text(encoding="utf-8")
        ),
        "import_closure_exact_and_astropy_absent": all(
            v19cs.sha256(ROOT / spec["path"]) == spec["sha256"]
            and "astropy" not in (ROOT / spec["path"]).read_text(encoding="utf-8").lower()
            for spec in config["source_import_audit"]
        ),
        "authorization_probe_only": (
            config["authorization"]["remove_unused_probe_module"]
            and not config["authorization"]["change_installed_environment"]
            and not config["authorization"]["change_x2_config_runner_scientific_rules_or_data"]
            and not config["authorization"]["run_v19bs_or_derive_action"]
            and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V19CT preflight failed: {checks}")
    base = v19cs.load_json(cs_config_path)
    correction = config["effective_config_correction"]
    if base["environment"]["required_python_modules"] != correction["before"]:
        raise RuntimeError("V19CT base probe-module list changed")
    effective = copy.deepcopy(base)
    effective["environment"]["required_python_modules"] = correction["after"]
    effective["outputs"] = config["effective_outputs"]
    return checks, effective


def execute(config: dict[str, Any]) -> dict[str, Any]:
    wrapper_preflight, effective = validate_and_build(config)
    cs_config_path = ROOT / config["parents"]["v19cs_config"]["path"]
    cs_hash_before = v19cs.sha256(cs_config_path)
    result = v19cs.execute(effective)
    if v19cs.sha256(cs_config_path) != cs_hash_before:
        raise RuntimeError("V19CT mutated the frozen V19CS config")
    result.update({
        "protocol_version": config["protocol_version"],
        "status": "v19x2_required_ciao_probe_completed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19cs.sha256(DEFAULT_CONFIG),
        "base_v19cs_config_sha256_before_and_after": cs_hash_before,
        "wrapper_preflight": wrapper_preflight,
        "effective_config_changes": [{
            "path": config["effective_config_correction"]["json_path"],
            "before": config["effective_config_correction"]["before"],
            "after": config["effective_config_correction"]["after"],
        }],
        "claim_boundary": config["claim_boundary"],
    })
    return result


def main() -> None:
    config = v19cs.load_json(DEFAULT_CONFIG)
    output = ROOT / config["effective_outputs"]["report"]
    try:
        report = execute(config)
    except Exception as exc:
        report = {
            "protocol_version": config["protocol_version"], "status": "v19x2_required_ciao_probe_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(),
            "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False},
            "claim_boundary": config["claim_boundary"],
        }
    v19cs.atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "v19x2_required_ciao_probe_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
