#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cp_v19x2_runtime_alias_remediation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode, "log": path.relative_to(ROOT).as_posix(), "log_sha256": sha256(path), "log_bytes": path.stat().st_size}


def execute() -> dict[str, Any]:
    config = load_json(DEFAULT_CONFIG)
    co_path = ROOT / config["failure_parents"]["v19co_report"]["path"]
    x2_fail_path = ROOT / config["failure_parents"]["v19x2_failure_report"]["path"]
    w5_path = ROOT / config["failure_parents"]["v19w5_report"]["path"]
    co, x2_fail, w5 = map(load_json, (co_path, x2_fail_path, w5_path))
    x2_config_path = ROOT / config["config_correction"]["path"]
    before = load_json(x2_config_path)
    execution = config["unchanged_execution"]
    preflight = {
        "failure_reports_exact": (
            sha256(co_path) == config["failure_parents"]["v19co_report"]["sha256"] and co["exception"] == config["failure_parents"]["v19co_report"]["required_exception"]
            and sha256(x2_fail_path) == config["failure_parents"]["v19x2_failure_report"]["sha256"] and x2_fail["status"] == config["failure_parents"]["v19x2_failure_report"]["required_status"] and x2_fail["execution_exception"] == config["failure_parents"]["v19x2_failure_report"]["required_exception"]
        ),
        "v19w5_pass_exact": sha256(w5_path) == config["failure_parents"]["v19w5_report"]["sha256"] and w5["status"] == config["failure_parents"]["v19w5_report"]["required_status"] and all(w5["gates"].values()),
        "x2_config_boundary_exact_and_alias_absent": sha256(x2_config_path) == config["config_correction"]["sha256_before"] and "required_completed_cells" not in before["runtime_authorization"] and before["runtime_authorization"]["required_unified_cells"] == config["config_correction"]["value"],
        "unchanged_runners_exact": sha256(ROOT / execution["v19x2_runner"]) == execution["v19x2_runner_sha256"] and sha256(ROOT / execution["v19br_config"]) == execution["v19br_config_sha256"] and sha256(ROOT / execution["v19br_runner"]) == execution["v19br_runner_sha256"],
        "authorization_source_only": not config["authorization"]["change_scientific_section_or_value"] and not config["authorization"]["run_v19bs_or_derive_action"] and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"],
    }
    if not all(preflight.values()):
        raise RuntimeError(f"V19CP preflight failed: {preflight}")

    after = json.loads(json.dumps(before))
    after["runtime_authorization"]["required_completed_cells"] = config["config_correction"]["value"]
    atomic_json(x2_config_path, after)
    changed_paths = []
    for key in set(before["runtime_authorization"]) | set(after["runtime_authorization"]):
        if before["runtime_authorization"].get(key) != after["runtime_authorization"].get(key):
            changed_paths.append({"path": f"runtime_authorization.{key}", "before": before["runtime_authorization"].get(key), "after": after["runtime_authorization"].get(key)})
    scientific_exact = all(before[section] == after[section] for section in config["config_correction"]["scientific_sections_unchanged"])

    x2_command = [
        sys.executable, str(ROOT / execution["v19x2_runner"]), "--config", str(x2_config_path),
        "--output", str(ROOT / execution["v19x2_output"]), "--scratch", execution["v19x2_scratch"], "--response-report", str(w5_path),
    ]
    x2_exec = run_logged(x2_command, ROOT / config["outputs"]["v19x2_log"])
    x2_report = load_json(ROOT / execution["v19x2_output"] / "report.json")
    x2_scientific_disposition = x2_report["status"] in {
        "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized",
        "unified_spectral_combination_commissioning_gate_failed",
    }
    br_exec = None
    br_report = None
    if x2_report["status"] == "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized":
        br_command = [sys.executable, str(ROOT / execution["v19br_runner"]), "--config", str(ROOT / execution["v19br_config"]), "--execute"]
        br_exec = run_logged(br_command, ROOT / config["outputs"]["v19br_log"])
        br_report = load_json(ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "report.json")

    gates = {
        "failure_reports_v19w5_pass_and_config_boundary_exact": all(preflight.values()),
        "only_required_completed_cells_alias_added_and_equals_5082": changed_paths == [{"path": "runtime_authorization.required_completed_cells", "before": None, "after": 5082}] and scientific_exact,
        "v19x2_byte_identical_runner_reaches_registered_scientific_disposition": x2_scientific_disposition and x2_report["runner_sha256"] == execution["v19x2_runner_sha256"],
        "if_v19x2_passes_unchanged_v19br_reaches_source_decision": (x2_report["status"] != "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized") or (br_exec is not None and br_exec["returncode"] == 0 and br_report is not None and br_report["status"] == "terminal_chain_complete" and br_report.get("source_decision") is not None),
        "no_lensing_halo_action_gravity_holdout_or_solar_access": not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"] and not x2_report["replacement_cluster_lensing_target_opened"] and not x2_report["gravity_formula_or_parameter_changed"] and (br_report is None or not br_report["lensing_halo_action_gravity_or_holdout_payload_opened"]),
    }
    decision = (
        "run_frozen_v19bs_disposition_next" if br_report is not None and br_report.get("status") == "terminal_chain_complete"
        else "v19x2_valid_scientific_gate_failure_no_full_source_chain" if x2_report["status"] == "unified_spectral_combination_commissioning_gate_failed"
        else "v19x2_or_source_chain_execution_incomplete"
    )
    return {
        "protocol_version": config["protocol_version"], "status": "v19x2_runtime_alias_remediation_completed", "decision": decision,
        "generated_utc": datetime.now(UTC).isoformat(), "config_sha256": sha256(DEFAULT_CONFIG), "preflight": preflight,
        "config_sha256_after": sha256(x2_config_path), "changed_paths": changed_paths, "scientific_sections_unchanged": scientific_exact,
        "v19x2_execution": x2_exec, "v19x2_report": x2_report, "v19br_execution": br_exec,
        "v19br_summary": None if br_report is None else {"status": br_report["status"], "source_decision": br_report.get("source_decision"), "executed_stages": br_report.get("executed_stages"), "already_passing_stages": br_report.get("already_passing_stages")},
        "gate_results": gates, "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False}, "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    try:
        report = execute()
    except Exception as exc:
        report = {"protocol_version": config["protocol_version"], "status": "v19x2_runtime_alias_remediation_failed_closed", "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(), "authorization_boundary": {"action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False}, "claim_boundary": config["claim_boundary"]}
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "v19x2_runtime_alias_remediation_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
