#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
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

DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19co_checkpoint_attempt_metadata_correction.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def product_paths(checkpoint: Path, report: dict[str, Any]) -> dict[str, Path]:
    return {
        role: checkpoint.parent / "products" / report["products"][role]["name"]
        for role in ("source_pha", "background_pha", "arf", "rmf")
    }


def load_manifest_row(report: dict[str, Any]) -> dict[str, str]:
    v19w_config = load_json(ROOT / "configs" / "sigma_v19w_full_response_production.json")
    manifest_path = ROOT / v19w_config["parents"]["v19u_manifest"]
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [
        row for row in rows
        if (row["cluster"], int(row["bin_id"]), int(row["obsid"]), int(row["ccd_id"]))
        == (report["cluster"], int(report["bin_id"]), int(report["obsid"]), int(report["ccd_id"]))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"V19CO manifest resolution returned {len(matches)} rows")
    return matches[0]


def run_logged(command: list[str], path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode, "log": path.relative_to(ROOT).as_posix(), "log_sha256": sha256(path), "log_bytes": path.stat().st_size}


def execute() -> dict[str, Any]:
    config = load_json(DEFAULT_CONFIG)
    parent_path = ROOT / config["parent_failure"]["path"]
    parent = load_json(parent_path)
    checkpoint = Path(config["checkpoint"]["path"])
    before_bytes = checkpoint.read_bytes()
    before = json.loads(before_bytes)
    products = product_paths(checkpoint, before)
    archive = Path(config["archive"]["scratch"])
    preflight = {
        "parent_failure_exact": sha256(parent_path) == config["parent_failure"]["sha256"] and parent["status"] == config["parent_failure"]["required_status"] and parent["exception"] == config["parent_failure"]["required_exception"],
        "checkpoint_hash_attempt_command_and_gates_exact": sha256(checkpoint) == config["checkpoint"]["sha256_before"] and before["attempt"] == config["checkpoint"]["attempt_before"] and config["checkpoint"]["required_command_argument"] in before["step"]["command"] and all(before["gates"].values()),
        "four_products_exact": {role: sha256(path) for role, path in products.items()} == config["checkpoint"]["product_hashes"],
        "archive_boundary_exact": len(list((archive / "completed").glob("*/cell_report.json"))) == config["archive"]["required_completed"] and {path.name for path in (archive / "failed_attempts").iterdir()} == set(config["archive"]["required_failed_attempt_directories"]),
        "unchanged_chain_hashes_exact": (
            sha256(ROOT / config["unchanged_chain"]["v19w5_config"]) == config["unchanged_chain"]["v19w5_config_sha256"]
            and sha256(ROOT / config["unchanged_chain"]["v19w5_runner"]) == config["unchanged_chain"]["v19w5_runner_sha256"]
            and sha256(ROOT / config["unchanged_chain"]["v19br_config"]) == config["unchanged_chain"]["v19br_config_sha256"]
            and sha256(ROOT / config["unchanged_chain"]["v19br_runner"]) == config["unchanged_chain"]["v19br_runner_sha256"]
        ),
        "authorization_source_only": not config["authorization"]["change_product_or_other_scientific_metadata"] and not config["authorization"]["run_v19bs_or_derive_action"] and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"],
    }
    if not all(preflight.values()):
        raise RuntimeError(f"V19CO preflight failed: {preflight}")

    corrected = dict(before)
    corrected["attempt"] = config["checkpoint"]["attempt_after"]
    atomic_json(checkpoint, corrected)
    after = load_json(checkpoint)
    differences = {key: {"before": before.get(key), "after": after.get(key)} for key in set(before) | set(after) if before.get(key) != after.get(key)}
    after_product_hashes = {role: sha256(path) for role, path in products.items()}
    manifest_row = load_manifest_row(after)
    audited = v19w_audit.validate_checkpoint(checkpoint, manifest_row)

    chain = config["unchanged_chain"]
    w5_command = [
        sys.executable, str(ROOT / chain["v19w5_runner"]), "--config", str(ROOT / chain["v19w5_config"]),
        "--output", str(ROOT / chain["v19w5_output"]), "--scratch", config["archive"]["scratch"], "--base-scratch", config["archive"]["base_scratch"],
    ]
    w5_execution = run_logged(w5_command, ROOT / config["outputs"]["v19w5_log"])
    if w5_execution["returncode"] != 0:
        raise RuntimeError(f"V19CO V19W5 exited {w5_execution['returncode']}")
    w5_report = load_json(ROOT / chain["v19w5_output"] / "report.json")
    br_command = [sys.executable, str(ROOT / chain["v19br_runner"]), "--config", str(ROOT / chain["v19br_config"]), "--execute"]
    br_execution = run_logged(br_command, ROOT / config["outputs"]["v19br_log"])
    if br_execution["returncode"] != 0:
        raise RuntimeError(f"V19CO V19BR exited {br_execution['returncode']}")
    br_report = load_json(ROOT / "results" / "sigma_v19br_target_sealed_terminal_chain" / "report.json")
    gates = {
        "v19cn_failure_and_checkpoint_boundary_exact": all(preflight.values()),
        "all_four_product_hashes_unchanged": after_product_hashes == config["checkpoint"]["product_hashes"],
        "only_attempt_field_changes_from_3_to_1": differences == {"attempt": {"before": 3, "after": 1}},
        "corrected_checkpoint_passes_independent_manifest_audit": bool(audited) and audited["attempt"] == 1,
        "v19w5_passes_5082_cells_and_20328_products": w5_report["status"] == "ccd7_hardened_unified_5082_response_archive_passed" and w5_report["unified_cells"] == 5082 and w5_report["unified_product_files"] == 20328 and all(w5_report["gates"].values()),
        "v19br_reaches_terminal_source_decision": br_report["status"] == "terminal_chain_complete" and br_report.get("source_decision") is not None,
        "no_target_action_gravity_holdout_or_solar_access": not br_report["lensing_halo_action_gravity_or_holdout_payload_opened"] and not config["authorization"]["open_lensing_halo_gravity_holdout_or_solar_optimization"],
    }
    if set(gates) != set(config["required_gates"]):
        raise RuntimeError("V19CO gate schema changed")
    return {
        "protocol_version": config["protocol_version"], "status": "checkpoint_metadata_corrected_full_archive_and_source_chain_complete" if all(gates.values()) else "checkpoint_metadata_correction_failed_closed",
        "decision": "run_frozen_v19bs_disposition_next" if all(gates.values()) else "no_action_authorized", "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(DEFAULT_CONFIG), "preflight": preflight, "json_differences": differences, "checkpoint_sha256_after": sha256(checkpoint),
        "product_hashes_after": after_product_hashes, "independent_checkpoint_audit": audited, "v19w5_execution": w5_execution,
        "v19w5_summary": {key: w5_report[key] for key in ("status", "unified_cells", "unified_product_files", "unified_product_bytes", "gates")},
        "v19br_execution": br_execution, "v19br_summary": {"status": br_report["status"], "executed_stages": br_report["executed_stages"], "already_passing_stages": br_report["already_passing_stages"], "source_decision": br_report["source_decision"]},
        "gate_results": gates, "authorization_boundary": {"v19bs_run": False, "action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False}, "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    config = load_json(DEFAULT_CONFIG)
    output = ROOT / config["outputs"]["report"]
    try:
        report = execute()
    except Exception as exc:
        report = {"protocol_version": config["protocol_version"], "status": "checkpoint_metadata_correction_failed_closed", "exception": f"{type(exc).__name__}: {exc}", "generated_utc": datetime.now(UTC).isoformat(), "authorization_boundary": {"action_derived": False, "target_or_gravity_opened": False, "solar_optimized": False}, "claim_boundary": config["claim_boundary"]}
    atomic_json(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "exception")}, indent=2, sort_keys=True))
    if report["status"] != "checkpoint_metadata_corrected_full_archive_and_source_chain_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
