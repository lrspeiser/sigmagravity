#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigma_v19w_live_archive as v19w_audit
import run_sigma_v19w2_exact_binmap_response_commissioning as v19w2
import run_sigma_v19w3_full_response_recovery as v19w3
import run_sigma_v19w5_ccd7_hardened_response_recovery as v19w5


DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cn_fine_wmap_final_recovery.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    parent_hashes = {key: sha256(ROOT / spec["path"]) for key, spec in config["parents"].items()}
    hashes_exact = {key: parent_hashes[key] == spec["sha256"] for key, spec in config["parents"].items()}
    cm = load_json(ROOT / config["parents"]["v19cm_report"]["path"])
    boundary = config["boundary"]
    scratch = Path(boundary["scratch"])
    partial = Path(boundary["current_failed_partial"])
    completed = list((scratch / "completed").glob("*/cell_report.json"))
    checks = {
        "all_parent_hashes_and_v19cm_pass_exact": all(hashes_exact.values()) and cm["decision"] == config["parents"]["v19cm_report"]["required_decision"] and all(cm["gate_results"].values()),
        "scratch_initial_boundary_exact": (
            len(completed) == boundary["completed_before"]
            and partial.is_dir()
            and sha256(partial / "logs" / "specextract.log") == boundary["current_failed_log_sha256"]
            and (scratch / "failed_attempts" / "c3432_rmf_attempt1").is_dir()
            and not Path(boundary["failed_attempt2_destination"]).exists()
            and not (scratch / "completed" / boundary["cell"] / "cell_report.json").exists()
        ),
        "no_base_recovery_process": not v19w3.running_base_processes(),
        "authorization_is_source_only": (
            not config["authorization"]["open_lensing_halo_action_gravity_or_holdout"]
            and not config["authorization"]["derive_action_or_change_gravity_constants"]
            and not config["authorization"]["run_v19bs_disposition_here"]
            and not config["authorization"]["perform_detailed_solar_optimization"]
        ),
    }
    return {"checks": checks, "parent_hashes": parent_hashes, "hashes_exact": hashes_exact}


def recover_edge_cell(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["boundary"]
    scratch = Path(boundary["scratch"])
    partial = Path(boundary["current_failed_partial"])
    attempt2 = Path(boundary["failed_attempt2_destination"])
    partial.rename(attempt2)

    w5_config = load_json(ROOT / config["parents"]["v19w5_config"]["path"])
    _, w3_config, _ = v19w5.verify_parents(w5_config)
    base_config = load_json(ROOT / w3_config["parents"]["v19w_config"]["path"])
    manifest = v19w3.v19w.load_manifest(base_config)
    rows = [row for row in manifest if v19w3.v19w.cell_name(row) == boundary["cell"]]
    if len(rows) != 1:
        raise RuntimeError(f"V19CN expected one manifest row, got {len(rows)}")
    row = rows[0]
    contexts = v19w3.v19w.observation_contexts(base_config, rows, scratch)
    cell = v19w2.prepare_mask_cell(
        row, contexts[(row["cluster"], int(row["obsid"]))], scratch
    )

    original_run_step = v19w2.v19w.inherited.run_step
    changes: list[dict[str, Any]] = []

    def det1_run_step(command: list[str], log_path: Path, expected: list[Path], env: dict[str, str]) -> dict[str, Any]:
        modified = list(command)
        if Path(modified[0]).name == "specextract" or modified[0] == "specextract":
            indices = [index for index, value in enumerate(modified) if value == "binwmap=det=8"]
            if len(indices) != 1:
                raise RuntimeError(f"V19CN expected one det=8 argument, got {indices}")
            index = indices[0]
            modified[index] = "binwmap=det=1"
            differences = [(i, before, after) for i, (before, after) in enumerate(zip(command, modified)) if before != after]
            if differences != [(index, "binwmap=det=8", "binwmap=det=1")]:
                raise RuntimeError(f"V19CN command difference changed: {differences}")
            changes.append({"index": index, "before": command[index], "after": modified[index]})
        return original_run_step(modified, log_path, expected, env)

    v19w2.v19w.inherited.run_step = det1_run_step
    try:
        record = v19w2.execute_mask_cell(cell, scratch)
    finally:
        v19w2.v19w.inherited.run_step = original_run_step
    if changes != [{"index": changes[0]["index"], "before": "binwmap=det=8", "after": "binwmap=det=1"}] if changes else True:
        raise RuntimeError(f"V19CN did not apply exactly one command change: {changes}")

    checkpoint = scratch / "completed" / boundary["cell"] / "cell_report.json"
    stored = load_json(checkpoint)
    stored["attempt"] = 3
    stored["edge_wmap_remediation"] = {
        "protocol": config["protocol_version"],
        "diagnosed_failure": "coarse det=8 WMAP nonzero bin centers mapped outside the ACIS chip",
        "only_command_change": changes[0],
        "weight_and_response_reference_unchanged": True,
        "failed_attempts_preserved": [
            str(scratch / "failed_attempts" / "c3432_rmf_attempt1"),
            str(attempt2),
        ],
        "diagnostic_product_admitted": False,
    }
    atomic_json(checkpoint, stored)
    audited = v19w_audit.validate_checkpoint(checkpoint, row)
    return {
        "cell": boundary["cell"],
        "command_changes": changes,
        "original_gates": record["gates"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "independent_audit": audited,
        "failed_attempt2_preserved": attempt2.is_dir(),
        "completed_cells_after": len(list((scratch / "completed").glob("*/cell_report.json"))),
    }


def run_logged(command: list[str], path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode, "log": path.relative_to(ROOT).as_posix(), "log_sha256": sha256(path), "log_bytes": path.stat().st_size}


def execute(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    before = preflight(config)
    if not all(before["checks"].values()):
        raise RuntimeError(f"V19CN preflight failed: {before['checks']}")
    edge = recover_edge_cell(config)

    w5_command = [
        sys.executable, str(ROOT / config["parents"]["v19w5_runner"]["path"]),
        "--config", str(ROOT / config["parents"]["v19w5_config"]["path"]),
        "--output", str(ROOT / config["execution"]["v19w5_output"]),
        "--scratch", config["boundary"]["scratch"],
        "--base-scratch", config["boundary"]["base_scratch"],
    ]
    w5_log = ROOT / config["outputs"]["v19w5_log"]
    w5_execution = run_logged(w5_command, w5_log)
    if w5_execution["returncode"] != 0:
        raise RuntimeError(f"V19CN unchanged V19W5 exited {w5_execution['returncode']}")
    w5_report = load_json(ROOT / config["execution"]["v19w5_report"])
    if w5_report["status"] != "ccd7_hardened_unified_5082_response_archive_passed" or not all(w5_report["gates"].values()):
        raise RuntimeError("V19CN V19W5 report did not pass")

    br_command = [sys.executable, str(ROOT / config["parents"]["v19br_runner"]["path"]), "--config", str(ROOT / config["parents"]["v19br_config"]["path"]), "--execute"]
    br_log = ROOT / config["outputs"]["v19br_log"]
    br_execution = run_logged(br_command, br_log)
    if br_execution["returncode"] != 0:
        raise RuntimeError(f"V19CN unchanged V19BR exited {br_execution['returncode']}")
    br_report = load_json(ROOT / config["execution"]["v19br_report"])
    gates = {
        "all_parent_hashes_and_v19cm_pass_exact": all(before["checks"].values()),
        "scratch_has_383_completed_one_partial_two_failure_records_after_preservation": edge["failed_attempt2_preserved"] and edge["completed_cells_after"] == config["boundary"]["completed_after_edge_recovery"],
        "exactly_one_specextract_argument_changed_to_det1": len(edge["command_changes"]) == 1 and edge["command_changes"][0]["before"] == "binwmap=det=8" and edge["command_changes"][0]["after"] == "binwmap=det=1",
        "edge_cell_passes_every_original_v19w2_gate": all(edge["original_gates"].values()),
        "edge_checkpoint_passes_independent_manifest_audit": bool(edge["independent_audit"]),
        "v19w5_passes_5082_cells_20328_products_and_protected_base_audits": w5_report["unified_cells"] == 5082 and w5_report["unified_product_files"] == 20328 and all(w5_report["gates"].values()),
        "unchanged_v19br_reaches_terminal_source_decision": br_report["status"] == "terminal_chain_complete" and br_report.get("source_decision") is not None,
        "no_lensing_halo_action_gravity_holdout_or_solar_optimization_access": not br_report["lensing_halo_action_gravity_or_holdout_payload_opened"] and not config["authorization"]["perform_detailed_solar_optimization"],
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("V19CN gate schema changed")
    return {
        "protocol_version": config["protocol_version"],
        "status": "fine_wmap_edge_recovery_and_target_sealed_source_chain_complete" if all(gates.values()) else "fine_wmap_final_recovery_failed_closed",
        "decision": "run_frozen_v19bs_disposition_next" if all(gates.values()) else "no_action_or_scientific_progress_authorized",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "preflight": before,
        "edge_recovery": edge,
        "v19w5_execution": w5_execution,
        "v19w5_summary": {key: w5_report[key] for key in ("status", "unified_cells", "unified_product_files", "unified_product_bytes", "gates")},
        "v19br_execution": br_execution,
        "v19br_summary": {"status": br_report["status"], "executed_stages": br_report["executed_stages"], "already_passing_stages": br_report["already_passing_stages"], "source_decision": br_report["source_decision"]},
        "gate_results": gates,
        "authorization_boundary": {"v19bs_run_here": False, "action_derived": False, "lensing_halo_gravity_or_holdout_opened": False, "solar_optimized": False},
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = execute()
    except Exception as exc:
        report = {"protocol_version": config["protocol_version"], "status": "fine_wmap_final_recovery_failed_closed", "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(), "config_sha256": sha256(DEFAULT_CONFIG), "authorization_boundary": {"action_derived": False, "lensing_halo_gravity_or_holdout_opened": False, "solar_optimized": False}, "claim_boundary": config["claim_boundary"]}
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "fine_wmap_edge_recovery_and_target_sealed_source_chain_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
