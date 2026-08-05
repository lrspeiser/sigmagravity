#!/usr/bin/env python3
"""Recover final V19W missing cells and independently audit one unified archive."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigma_v19w_live_archive as v19w_audit
import run_sigma_v19p_exact_flux_obs_support as v19p
import run_sigma_v19w2_exact_binmap_response_commissioning as v19w2
import run_sigma_v19w_full_response_production as v19w

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19w3_full_response_recovery.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w3_full_response_recovery"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19w3-response-recovery/v100")
DEFAULT_BASE_SCRATCH = Path("/home/henry/sigma-v19w-response-production/v100")
PRODUCT_ROLES = ("source_pha", "background_pha", "arf", "rmf")


def sha256(path: Path) -> str:
    return v19p.sha256(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_static_parents(config: dict[str, Any]) -> None:
    for spec in config["parents"].values():
        path = ROOT / spec["path"]
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise RuntimeError(f"V19W3 parent hash mismatch: {path}")
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    v19w.validate_parent_hashes(base_config)
    commissioning = load_json(
        ROOT / config["parents"]["v19w2_config"]["path"]
    )
    v19w2.verify_parent_hashes(commissioning)
    report = load_json(ROOT / config["parents"]["v19w2_report"]["path"])
    if (
        report["status"]
        != "exact_binmap_response_commissioning_passed_and_recovery_protocol_authorized"
        or not report["full_missing_cell_recovery_authorized"]
        or not all(report["gates"].values())
        or report["gravity_formula_or_parameter_changed"]
    ):
        raise RuntimeError("V19W2 did not authorize V19W3 recovery")


def running_base_processes(proc_root: Path = Path("/proc")) -> list[int]:
    matches: list[int] = []
    if not proc_root.is_dir():
        return matches
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"run_sigma_v19w_full_response_production.py" in command:
            matches.append(int(entry.name))
    return sorted(matches)


def validate_base_terminal_state(
    config: dict[str, Any],
    proc_root: Path = Path("/proc"),
    report_path: Path | None = None,
) -> dict[str, Any]:
    active = running_base_processes(proc_root)
    if active:
        raise RuntimeError(f"V19W base production is still running: {active}")
    path = (
        report_path
        if report_path is not None
        else ROOT / config["base_terminal_gate"]["required_final_report"]
    )
    if not path.is_file():
        raise RuntimeError(f"V19W final report is absent: {path}")
    report = load_json(path)
    base_config_path = ROOT / config["parents"]["v19w_config"]["path"]
    base_runner_path = ROOT / config["parents"]["v19w_runner"]["path"]
    if (
        report.get("config_sha256") != sha256(base_config_path)
        or report.get("runner_sha256") != sha256(base_runner_path)
        or int(report.get("expected_cells", -1)) != 5082
        or not report.get("full_interval_requested")
        or report.get("status")
        not in {
            "response_production_incomplete",
            "all_response_cells_passed_and_regional_spectral_fitting_authorized",
        }
    ):
        raise RuntimeError("V19W final report does not describe the frozen full run")
    index = ROOT / report["product_index"]["path"]
    if (
        not index.is_file()
        or index.stat().st_size != int(report["product_index"]["bytes"])
        or sha256(index) != report["product_index"]["sha256"]
    ):
        raise RuntimeError("V19W final product index is absent or changed")
    return {**report, "path": str(path), "sha256": sha256(path)}


def inventory_valid_base(
    manifest: list[dict[str, str]], base_scratch: Path
) -> tuple[dict[tuple[str, int, int, int], tuple[Path, dict[str, Any]]], dict[str, str]]:
    valid: dict[
        tuple[str, int, int, int], tuple[Path, dict[str, Any]]
    ] = {}
    invalid: dict[str, str] = {}
    for row in manifest:
        name = v19w.cell_name(row)
        report_path = base_scratch / "completed" / name / "cell_report.json"
        if not report_path.is_file():
            continue
        try:
            record = v19w_audit.validate_checkpoint(report_path, row)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            invalid[name] = f"{type(exc).__name__}: {exc}"
            continue
        valid[v19w.task_key(row)] = (report_path, record)
    return valid, invalid


def missing_manifest_rows(
    manifest: list[dict[str, str]],
    valid_base: dict[tuple[str, int, int, int], tuple[Path, dict[str, Any]]],
) -> list[dict[str, str]]:
    return [row for row in manifest if v19w.task_key(row) not in valid_base]


def progress_payload(
    config: dict[str, Any],
    missing: list[dict[str, str]],
    recovery_scratch: Path,
    failures: dict[str, str],
    started: float,
) -> dict[str, Any]:
    completed = len(
        list((recovery_scratch / "completed").glob("*/cell_report.json"))
    )
    return {
        "status": "v19w3_missing_cell_recovery_running",
        "protocol_version": config["protocol_version"],
        "updated_utc": datetime.now(UTC).isoformat(),
        "missing_cells_at_launch": len(missing),
        "recovery_completed_cells": completed,
        "recovery_failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "base_archive_modified": False,
        "gravity_formula_or_parameter_changed": False,
    }


def recover_missing(
    config: dict[str, Any],
    base_config: dict[str, Any],
    missing: list[dict[str, str]],
    scratch: Path,
    output: Path,
) -> dict[str, str]:
    if not missing:
        return {}
    contexts = v19w.observation_contexts(base_config, missing, scratch)
    prepared = [
        v19w2.prepare_mask_cell(
            row,
            contexts[(row["cluster"], int(row["obsid"]))],
            scratch,
        )
        for row in missing
    ]
    failures: dict[str, str] = {}
    started = time.perf_counter()
    workers = int(config["execution"]["maximum_concurrent_cells"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(v19w2.execute_mask_cell, cell, scratch): cell
            for cell in prepared
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            cell = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - retain every recovery failure
                failures[cell["cell_name"]] = f"{type(exc).__name__}: {exc}"
            if completed_count % 16 == 0 or completed_count == len(futures):
                atomic_json(
                    output / "progress.json",
                    progress_payload(config, missing, scratch, failures, started),
                )
                print(
                    f"recovery {completed_count}/{len(futures)}; "
                    f"failed={len(failures)}"
                )
                sys.stdout.flush()
    return failures


def selected_checkpoint(
    row: dict[str, str],
    valid_base: dict[tuple[str, int, int, int], tuple[Path, dict[str, Any]]],
    recovery_scratch: Path,
) -> tuple[str, Path]:
    key = v19w.task_key(row)
    if key in valid_base:
        return "base_v19w", valid_base[key][0]
    path = (
        recovery_scratch
        / "completed"
        / v19w.cell_name(row)
        / "cell_report.json"
    )
    return "v19w3_recovery", path


def write_unified_index(
    manifest: list[dict[str, str]],
    valid_base: dict[tuple[str, int, int, int], tuple[Path, dict[str, Any]]],
    recovery_scratch: Path,
    path: Path,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for manifest_row in manifest:
        archive, report_path = selected_checkpoint(
            manifest_row, valid_base, recovery_scratch
        )
        if not report_path.is_file():
            raise RuntimeError(f"V19W3 unified archive lacks {report_path}")
        audited = v19w_audit.validate_checkpoint(report_path, manifest_row)
        report = load_json(report_path)
        products = report["products"]
        completed = report_path.parent
        row: dict[str, Any] = {
            "production_index": int(manifest_row["production_index"]),
            "batch_id": int(manifest_row["batch_id"]),
            "cluster": manifest_row["cluster"],
            "bin_id": int(manifest_row["bin_id"]),
            "obsid": int(manifest_row["obsid"]),
            "ccd_id": int(manifest_row["ccd_id"]),
            "cell_name": audited["cell_name"],
            "archive": archive,
            "cell_directory": str(completed),
            "cell_report_sha256": audited["cell_report_sha256"],
            "four_product_bytes": audited["four_product_bytes"],
        }
        for role in PRODUCT_ROLES:
            row[f"{role}_name"] = products[role]["name"]
            row[f"{role}_bytes"] = int(products[role]["bytes"])
            row[f"{role}_sha256"] = audited["product_hashes"][role]
        rows.append(row)
        total_bytes += audited["four_product_bytes"]
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, total_bytes


def run(
    config_path: Path,
    output: Path,
    scratch: Path,
    base_scratch: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    runner = ROOT / config["execution"]["runner"]
    if runner.resolve() != Path(__file__).resolve() or sha256(runner) != config[
        "execution"
    ]["runner_sha256"]:
        raise RuntimeError("V19W3 frozen runner changed")
    verify_static_parents(config)
    if scratch == base_scratch or scratch.is_relative_to(base_scratch):
        raise RuntimeError("V19W3 recovery scratch overlaps the read-only base archive")
    base_report = validate_base_terminal_state(config)
    base_config = load_json(ROOT / config["parents"]["v19w_config"]["path"])
    manifest = v19w.load_manifest(base_config)
    valid_base, invalid_base = inventory_valid_base(manifest, base_scratch)
    missing = missing_manifest_rows(manifest, valid_base)
    scratch.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch).free
    if free_bytes < int(config["execution"]["minimum_free_bytes_at_launch"]):
        raise RuntimeError(f"V19W3 free-space gate failed: {free_bytes}")
    failures = recover_missing(config, base_config, missing, scratch, output)
    if failures:
        raise RuntimeError(f"V19W3 recovery failed closed: {failures}")
    index_path = output / "unified_product_index.csv"
    unified_rows, product_bytes = write_unified_index(
        manifest, valid_base, scratch, index_path
    )
    task_keys = [
        (row["cluster"], row["bin_id"], row["obsid"], row["ccd_id"])
        for row in unified_rows
    ]
    recovered = sum(row["archive"] == "v19w3_recovery" for row in unified_rows)
    gates = {
        "base_process_exited_and_final_report_hash_valid": True,
        "base_archive_was_read_only": True,
        "manifest_has_5082_unique_tasks": len(manifest) == len(set(task_keys)) == 5082,
        "every_valid_base_checkpoint_passed_independent_hash_audit": True,
        "every_missing_or_invalid_base_cell_has_one_recovery_checkpoint": recovered
        == len(missing),
        "every_unified_cell_preflight_histogram_response_link_size_and_hash_passes": True,
        "unified_product_index_has_5082_unique_rows": len(unified_rows)
        == len(set(task_keys))
        == 5082,
        "no_recovery_failure_remains": not failures,
    }
    report = {
        "status": (
            "unified_5082_response_archive_passed_and_v19x_successor_may_be_frozen"
            if all(gates.values())
            else "v19w3_unified_archive_failed_closed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(runner),
        "base_terminal_report": {
            "path": base_report["path"],
            "sha256": base_report["sha256"],
            "reported_completed_cells": int(base_report["completed_cells"]),
            "reported_status": base_report["status"],
        },
        "base_valid_cells": len(valid_base),
        "base_invalid_checkpoint_errors": invalid_base,
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
        "gates": gates,
        "original_v19x_authorized": False,
        "v19x_successor_configuration_may_be_frozen": all(gates.values()),
        "base_v19w_archive_modified": False,
        "spectrum_combined_or_fitted": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output / "report.json", report)
    if not all(gates.values()):
        raise RuntimeError(f"V19W3 unified audit failed closed: {gates}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--base-scratch", type=Path, default=DEFAULT_BASE_SCRATCH)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.status_only:
        print(f"active_base_pids: {running_base_processes()}")
        print(
            "final_report_exists: "
            f"{(ROOT / load_json(args.config)['base_terminal_gate']['required_final_report']).is_file()}"
        )
        return
    report = run(
        args.config,
        args.output.resolve(),
        args.scratch.resolve(),
        args.base_scratch.resolve(),
    )
    print(args.output.resolve() / "report.json")
    print(f"status: {report['status']}")


if __name__ == "__main__":
    main()
