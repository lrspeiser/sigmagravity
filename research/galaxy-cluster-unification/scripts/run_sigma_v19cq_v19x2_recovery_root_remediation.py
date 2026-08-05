#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cq_v19x2_recovery_root_remediation.json"


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
    return {
        "command": command,
        "returncode": completed.returncode,
        "log": path.relative_to(ROOT).as_posix(),
        "log_sha256": sha256(path),
        "log_bytes": path.stat().st_size,
    }


def leaf_changes(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                changes.append({"path": path, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": path, "before": before[key], "after": None})
            else:
                changes.extend(leaf_changes(before[key], after[key], path))
        return changes
    if before != after:
        return [{"path": prefix, "before": before, "after": after}]
    return []


def index_audit(path: Path, corrected_root: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["archive"] for row in rows)
    recovery = [row for row in rows if row["archive"] == "v19w5_recovery"]
    roots = set()
    invalid_directories = []
    for row in recovery:
        directory = PurePosixPath(row["cell_directory"])
        expected = PurePosixPath(corrected_root) / "completed" / row["cell_name"]
        roots.add(str(directory.parent.parent))
        if directory != expected:
            invalid_directories.append(row["cell_directory"])
    return {
        "rows": len(rows),
        "archive_counts": dict(sorted(counts.items())),
        "recovery_rows": len(recovery),
        "recovery_roots": sorted(roots),
        "invalid_recovery_directories": invalid_directories,
    }


def execute(config: dict[str, Any]) -> dict[str, Any]:
    parents = config["failure_parents"]
    cp_path = ROOT / parents["v19cp_report"]["path"]
    x2_fail_path = ROOT / parents["v19x2_failure_report"]["path"]
    w5_path = ROOT / parents["v19w5_report"]["path"]
    index_path = ROOT / parents["v19w5_unified_index"]["path"]
    cp = load_json(cp_path)
    x2_fail = load_json(x2_fail_path)
    w5 = load_json(w5_path)
    correction = config["config_correction"]
    x2_config_path = ROOT / correction["path"]
    before = load_json(x2_config_path)
    execution = config["unchanged_execution"]
    audit = index_audit(index_path, correction["value_after"])

    preflight = {
        "v19cp_failure_exact": (
            sha256(cp_path) == parents["v19cp_report"]["sha256"]
            and cp["status"] == parents["v19cp_report"]["required_status"]
            and cp["decision"] == parents["v19cp_report"]["required_decision"]
        ),
        "v19x2_path_failure_exact": (
            sha256(x2_fail_path) == parents["v19x2_failure_report"]["sha256"]
            and x2_fail["status"] == parents["v19x2_failure_report"]["required_status"]
            and x2_fail["execution_exception"] == parents["v19x2_failure_report"]["required_exception"]
        ),
        "v19w5_pass_exact": (
            sha256(w5_path) == parents["v19w5_report"]["sha256"]
            and w5["status"] == parents["v19w5_report"]["required_status"]
            and all(w5["gates"].values())
        ),
        "unified_index_exact": (
            sha256(index_path) == parents["v19w5_unified_index"]["sha256"]
            and audit["rows"] == parents["v19w5_unified_index"]["required_rows"]
            and audit["archive_counts"] == parents["v19w5_unified_index"]["required_archive_counts"]
        ),
        "recovery_root_independently_exact": (
            audit["recovery_rows"] == parents["v19w5_unified_index"]["required_archive_counts"]["v19w5_recovery"]
            and audit["recovery_roots"] == [correction["value_after"]]
            and not audit["invalid_recovery_directories"]
        ),
        "x2_config_boundary_exact": (
            sha256(x2_config_path) == correction["sha256_before"]
            and before["execution"]["response_archives"]["v19w5_recovery"] == correction["value_before"]
            and before["execution"]["response_archives"]["base_v19w"] == correction["required_base_root_unchanged"]
        ),
        "unchanged_runners_exact": (
            sha256(ROOT / execution["v19x2_runner"]) == execution["v19x2_runner_sha256"]
            and sha256(ROOT / execution["v19br_config"]) == execution["v19br_config_sha256"]
            and sha256(ROOT / execution["v19br_runner"]) == execution["v19br_runner_sha256"]
        ),
        "authorization_source_only": (
            config["authorization"]["change_one_recovery_archive_root"]
            and not config["authorization"]["change_base_archive_root"]
            and not config["authorization"]["change_scientific_section_or_value"]
            and not config["authorization"]["run_v19bs_or_derive_action"]
            and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
        ),
    }
    if not all(preflight.values()):
        raise RuntimeError(f"V19CQ preflight failed: {preflight}")

    after = json.loads(json.dumps(before))
    after["execution"]["response_archives"]["v19w5_recovery"] = correction["value_after"]
    changes = leaf_changes(before, after)
    expected_change = [{
        "path": correction["json_path_changed"],
        "before": correction["value_before"],
        "after": correction["value_after"],
    }]
    scientific_exact = all(before[section] == after[section] for section in correction["scientific_sections_unchanged"])
    if changes != expected_change or not scientific_exact:
        raise RuntimeError(f"V19CQ correction boundary failed: changes={changes}, scientific_exact={scientific_exact}")
    atomic_json(x2_config_path, after)

    x2_command = [
        sys.executable,
        str(ROOT / execution["v19x2_runner"]),
        "--config",
        str(x2_config_path),
        "--output",
        str(ROOT / execution["v19x2_output"]),
        "--scratch",
        execution["v19x2_scratch"],
        "--response-report",
        str(w5_path),
    ]
    x2_exec = run_logged(x2_command, ROOT / config["outputs"]["v19x2_log"])
    x2_report = load_json(ROOT / execution["v19x2_output"] / "report.json")
    passing_status = "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized"
    failing_status = "unified_spectral_combination_commissioning_gate_failed"
    x2_scientific_disposition = x2_report["status"] in {passing_status, failing_status}

    br_exec = None
    br_report = None
    if x2_report["status"] == passing_status:
        br_command = [sys.executable, str(ROOT / execution["v19br_runner"]), "--config", str(ROOT / execution["v19br_config"]), "--execute"]
        br_exec = run_logged(br_command, ROOT / config["outputs"]["v19br_log"])
        br_report = load_json(ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "report.json")

    gates = {
        "failure_reports_and_unified_index_exact": all(preflight.values()),
        "all_384_recovery_rows_share_corrected_root": preflight["recovery_root_independently_exact"],
        "only_recovery_archive_root_changed": changes == expected_change,
        "scientific_sections_unchanged": scientific_exact,
        "v19x2_reaches_registered_scientific_disposition": x2_scientific_disposition and x2_report["runner_sha256"] == execution["v19x2_runner_sha256"],
        "if_v19x2_passes_unchanged_v19br_reaches_source_decision": (
            x2_report["status"] != passing_status
            or (
                br_exec is not None
                and br_exec["returncode"] == 0
                and br_report is not None
                and br_report["status"] == "terminal_chain_complete"
                and br_report.get("source_decision") is not None
            )
        ),
        "no_lensing_halo_action_gravity_holdout_or_solar_access": (
            not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"]
            and not x2_report["replacement_cluster_lensing_target_opened"]
            and not x2_report["gravity_formula_or_parameter_changed"]
            and (br_report is None or not br_report["lensing_halo_action_gravity_or_holdout_payload_opened"])
        ),
    }
    if x2_report["status"] == failing_status:
        decision = "v19x2_valid_scientific_gate_failure_no_full_source_chain"
    elif br_report is not None and br_report.get("status") == "terminal_chain_complete":
        decision = "run_frozen_v19bs_disposition_next"
    else:
        decision = "v19x2_or_source_chain_execution_incomplete"
    if not all(gates.values()):
        raise RuntimeError(f"V19CQ post-execution gates failed: {gates}; decision={decision}; x2_status={x2_report['status']}")

    return {
        "protocol_version": config["protocol_version"],
        "status": "v19x2_recovery_root_remediation_completed",
        "decision": decision,
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG),
        "preflight": preflight,
        "unified_index_audit": audit,
        "config_sha256_after": sha256(x2_config_path),
        "changed_paths": changes,
        "scientific_sections_unchanged": scientific_exact,
        "v19x2_execution": x2_exec,
        "v19x2_report": x2_report,
        "v19br_execution": br_exec,
        "v19br_summary": None if br_report is None else {
            "status": br_report["status"],
            "source_decision": br_report.get("source_decision"),
            "executed_stages": br_report.get("executed_stages"),
            "already_passing_stages": br_report.get("already_passing_stages"),
        },
        "gate_results": gates,
        "authorization_boundary": {
            "v19bs_run": False,
            "action_derived": False,
            "target_or_gravity_opened": False,
            "solar_optimized": False,
        },
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    try:
        report = execute(config)
    except Exception as exc:
        report = {
            "protocol_version": config["protocol_version"],
            "status": "v19x2_recovery_root_remediation_failed_closed",
            "exception": f"{type(exc).__name__}: {exc}",
            "generated_utc": datetime.now(UTC).isoformat(),
            "authorization_boundary": {
                "v19bs_run": False,
                "action_derived": False,
                "target_or_gravity_opened": False,
                "solar_optimized": False,
            },
            "claim_boundary": config["claim_boundary"],
        }
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "v19x2_recovery_root_remediation_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
