#!/usr/bin/env python3
"""Run terminal response recovery only after the CCD7 boundary passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19w3_full_response_recovery as v19w3
import run_sigma_v19w4_hardened_response_recovery as v19w4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19w5_ccd7_hardened_response_recovery.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w5_ccd7_hardened_response_recovery"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19w5-response-recovery/v100")
DEFAULT_BASE_SCRATCH = Path("/home/henry/sigma-v19w-response-production/v100")
PRODUCT_ROLES = v19w4.PRODUCT_ROLES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_ccd7_parent(config: dict[str, Any]) -> dict[str, Any]:
    report = load_json(ROOT / config["parents"]["v19w2c_report"]["path"])
    gate = config["ccd7_absent_background_gate"]
    cells = report.get("completed_cells", [])
    if (
        report.get("status") != gate["required_status"]
        or len(cells) != int(gate["required_completed_cells"])
        or {int(row["ccd_id"]) for row in cells}
        != set(map(int, gate["required_ccd_ids"]))
        or {int(row["obsid"]) for row in cells}
        != set(map(int, gate["required_observation_contexts"]))
        or any(int(row["background_all_energy_events"]) != 0 for row in cells)
        or not all(bool(row["used_zero_background_path"]) for row in cells)
        or not all(report.get("gates", {}).values())
        or not report.get("v19w5_hardened_recovery_may_be_frozen")
    ):
        raise RuntimeError("V19W5 CCD7 absent-background parent gate failed")
    return report


def verify_parents(
    config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    hashes: dict[str, str] = {}
    for name, spec in config["parents"].items():
        path = ROOT / spec["path"]
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"V19W5 parent hash mismatch for {name}: {actual}")
        hashes[name] = actual

    v19w4_config = load_json(ROOT / config["parents"]["v19w4_config"]["path"])
    _, v19w3_config = v19w4.verify_parents(v19w4_config)
    ccd7_report = validate_ccd7_parent(config)
    runner = ROOT / config["execution"]["runner"]
    if runner.resolve() != Path(__file__).resolve():
        raise RuntimeError("V19W5 frozen runner path identifies another implementation")
    if sha256(runner) != config["execution"]["runner_sha256"]:
        raise RuntimeError("V19W5 frozen runner changed")
    return hashes, v19w3_config, ccd7_report


def relabel_recovery_rows(rows: list[dict[str, Any]], index_path: Path) -> None:
    for row in rows:
        if row["archive"] == "v19w3_recovery":
            row["archive"] = "v19w5_recovery"
    fields = list(rows[0])
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    config_path: Path,
    output: Path,
    scratch: Path,
    base_scratch: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output = output.resolve()
    scratch = scratch.resolve()
    base_scratch = base_scratch.resolve()
    config = load_json(config_path)
    parent_hashes, v19w3_config, ccd7_report = verify_parents(config)
    if scratch == base_scratch or scratch.is_relative_to(base_scratch):
        raise RuntimeError("V19W5 recovery scratch overlaps the protected base archive")

    base_report = v19w3.validate_base_terminal_state(v19w3_config)
    base_config = load_json(ROOT / v19w3_config["parents"]["v19w_config"]["path"])
    manifest = v19w3.v19w.load_manifest(base_config)
    roots = list(config["protected_base_audit"]["recursively_hashed_roots"])
    tree_before = v19w4.protected_tree_snapshot(base_scratch, roots)
    valid_before, invalid_before = v19w3.inventory_valid_base(manifest, base_scratch)
    inventory_before = v19w4.inventory_digest(valid_before, invalid_before)
    missing = v19w3.missing_manifest_rows(manifest, valid_before)
    scratch.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(Path(config["execution"]["free_space_probe"])).free
    if free_bytes < int(config["execution"]["minimum_free_bytes_at_launch"]):
        raise RuntimeError(f"V19W5 free-space gate failed: {free_bytes}")

    failures = v19w3.recover_missing(config, base_config, missing, scratch, output)
    if failures:
        raise RuntimeError(f"V19W5 recovery failed closed: {failures}")

    index_path = output / "unified_product_index.csv"
    unified_rows, product_bytes = v19w3.write_unified_index(
        manifest, valid_before, scratch, index_path
    )
    relabel_recovery_rows(unified_rows, index_path)
    second_audit = v19w4.revalidate_unified_index(manifest, index_path)
    valid_after, invalid_after = v19w3.inventory_valid_base(manifest, base_scratch)
    inventory_after = v19w4.inventory_digest(valid_after, invalid_after)
    tree_after = v19w4.protected_tree_snapshot(base_scratch, roots)
    task_keys = [
        (row["cluster"], row["bin_id"], row["obsid"], row["ccd_id"])
        for row in unified_rows
    ]
    recovered = sum(row["archive"] == "v19w5_recovery" for row in unified_rows)
    gates = {
        "all_frozen_parent_hashes_and_parent_gates_pass": True,
        "base_process_exited_and_terminal_report_passes": True,
        "manifest_has_5082_unique_tasks": len(manifest) == len(set(task_keys)) == 5082,
        "cross_detector_commissioning_passes": True,
        "ccd7_absent_background_commissioning_passes": True,
        "every_terminal_base_checkpoint_independently_audited": len(valid_before)
        + len(missing)
        == 5082,
        "every_missing_or_invalid_cell_has_one_recovery_checkpoint": recovered
        == len(missing),
        "unified_index_has_5082_unique_cells_and_20328_products": len(unified_rows)
        == len(set(task_keys))
        == 5082
        and second_audit["product_files"] == 20328,
        "second_full_unified_index_checkpoint_and_product_audit_passes": second_audit[
            "cells"
        ]
        == second_audit["unique_task_keys"]
        == 5082,
        "protected_base_tree_is_byte_identical_before_and_after": tree_before
        == tree_after,
        "protected_base_checkpoint_inventory_is_identical_before_and_after": inventory_before
        == inventory_after,
        "no_recovery_failure_remains": not failures,
    }
    required = config["required_gates"]
    if set(gates) != set(required) or not all(required.values()):
        raise RuntimeError("V19W5 frozen gate schema changed")
    passed = all(gates.values())
    report = {
        "status": (
            "ccd7_hardened_unified_5082_response_archive_passed"
            if passed
            else "v19w5_ccd7_hardened_unified_archive_failed_closed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": parent_hashes,
        "ccd7_commissioning_parent": {
            "status": ccd7_report["status"],
            "completed_cells": len(ccd7_report["completed_cells"]),
            "report_sha256": parent_hashes["v19w2c_report"],
        },
        "base_terminal_report": {
            "path": base_report["path"],
            "sha256": base_report["sha256"],
            "reported_completed_cells": int(base_report["completed_cells"]),
            "reported_status": base_report["status"],
        },
        "base_tree_before": tree_before,
        "base_tree_after": tree_after,
        "base_inventory_before": inventory_before,
        "base_inventory_after": inventory_after,
        "base_valid_cells": len(valid_before),
        "base_invalid_checkpoint_errors": invalid_before,
        "missing_cells_at_launch": len(missing),
        "recovered_cells": recovered,
        "unified_cells": len(unified_rows),
        "unified_product_files": len(unified_rows) * len(PRODUCT_ROLES),
        "unified_product_bytes": product_bytes,
        "unified_product_index": {
            "path": index_path.relative_to(ROOT).as_posix(),
            "bytes": index_path.stat().st_size,
            "sha256": sha256(index_path),
            "rows": len(unified_rows),
        },
        "second_full_unified_audit": second_audit,
        "gates": gates,
        "superseded_v19w4_executed": False,
        "original_v19x_authorized": False,
        "v19x_successor_configuration_may_be_frozen": passed,
        "base_v19w_archive_modified": False,
        "spectrum_combined_or_fitted": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output / "report.json", report)
    if not passed:
        raise RuntimeError(f"V19W5 final audit failed closed: {gates}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--base-scratch", type=Path, default=DEFAULT_BASE_SCRATCH)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    if args.status_only:
        _, v19w3_config, ccd7_report = verify_parents(config)
        final_report = ROOT / v19w3_config["base_terminal_gate"][
            "required_final_report"
        ]
        print(f"active_base_pids: {v19w3.running_base_processes()}")
        print(f"final_report_exists: {final_report.is_file()}")
        print(f"ccd7_parent_status: {ccd7_report['status']}")
        print(f"recovery_scratch_exists: {args.scratch.resolve().exists()}")
        return
    try:
        report = run(args.config, args.output, args.scratch, args.base_scratch)
    except Exception as exc:
        failure = {
            "status": "v19w5_execution_failed_closed",
            "generated_utc": datetime.now(UTC).isoformat(),
            "exception": f"{type(exc).__name__}: {exc}",
            "base_v19w_archive_modified_by_protocol": False,
            "spectrum_combined_or_fitted": False,
            "lensing_halo_or_gravity_payload_opened": False,
            "gravity_formula_or_parameter_changed": False,
        }
        atomic_json(args.output.resolve() / "failure_report.json", failure)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
