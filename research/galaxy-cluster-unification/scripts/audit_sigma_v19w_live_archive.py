#!/usr/bin/env python3
"""Read-only independent audit of a point-in-time V19W checkpoint snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19w_full_response_production.json"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19w-response-production/v100")
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w_live_archive_audit" / "report.json"
PRODUCT_ROLES = ("source_pha", "background_pha", "arf", "rmf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_task_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["cluster"]),
        int(row["bin_id"]),
        int(row["obsid"]),
        int(row["ccd_id"]),
    )


def cell_name(row: dict[str, Any]) -> str:
    cluster, bin_id, obsid, ccd_id = canonical_task_key(row)
    return f"{cluster}_bin{bin_id}_obs{obsid}_ccd{ccd_id}"


def load_manifest(config: dict[str, Any]) -> list[dict[str, str]]:
    path = ROOT / config["parents"]["v19u_manifest"]
    if sha256(path) != config["parents"]["v19u_manifest_sha256"]:
        raise RuntimeError("V19W live audit manifest hash changed")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(config["workload"]["expected_task_count"])
    keys = [canonical_task_key(row) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise RuntimeError("V19W live audit manifest count or uniqueness changed")
    return rows


def validate_checkpoint(
    report_path: Path,
    manifest_row: dict[str, str],
) -> dict[str, Any]:
    completed = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_key = canonical_task_key(manifest_row)
    if canonical_task_key(report) != expected_key:
        raise RuntimeError(f"checkpoint task key changed: {report_path}")
    expected_name = cell_name(manifest_row)
    if completed.name != expected_name or report["cell_name"] != expected_name:
        raise RuntimeError(f"checkpoint cell name changed: {report_path}")
    attempt = int(report["attempt"])
    if attempt < 1 or attempt > 2:
        raise RuntimeError(f"checkpoint attempt outside frozen range: {expected_name}")
    gates = report.get("gates", {})
    if not gates or not all(value is True for value in gates.values()):
        raise RuntimeError(f"checkpoint gate failed: {expected_name}")
    if int(report["preflight"]["source_band_events"]) != int(
        manifest_row["source_band_events"]
    ):
        raise RuntimeError(f"checkpoint source count changed: {expected_name}")
    if int(report["preflight"]["background_band_events"]) != int(
        manifest_row["background_band_events"]
    ):
        raise RuntimeError(f"checkpoint background count changed: {expected_name}")
    if not report["source_pha_channel_audit"]["exact"]:
        raise RuntimeError(f"checkpoint source PHA audit failed: {expected_name}")
    if not report["background_pha_channel_audit"]["exact"]:
        raise RuntimeError(f"checkpoint background PHA audit failed: {expected_name}")
    response = report["response_audit"]
    if not (
        response["arf_finite"]
        and int(response["arf_positive_bins"]) > 0
        and response["rmf_finite"]
        and int(response["rmf_nonzero_elements"]) > 0
    ):
        raise RuntimeError(f"checkpoint response audit failed: {expected_name}")
    products = report["products"]
    if set(products) != set(PRODUCT_ROLES):
        raise RuntimeError(f"checkpoint product roles changed: {expected_name}")
    expected_links = {
        "BACKFILE": products["background_pha"]["name"],
        "ANCRFILE": products["arf"]["name"],
        "RESPFILE": products["rmf"]["name"],
    }
    if report["source_pha_links"] != expected_links:
        raise RuntimeError(f"checkpoint PHA links changed: {expected_name}")
    product_bytes = 0
    product_hashes: dict[str, str] = {}
    for role in PRODUCT_ROLES:
        item = products[role]
        path = completed / "products" / item["name"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"checkpoint product missing or resized: {path}")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"checkpoint product hash changed: {path}")
        product_hashes[role] = digest
        product_bytes += int(item["bytes"])
    if product_bytes != int(report["four_product_bytes"]):
        raise RuntimeError(f"checkpoint byte total changed: {expected_name}")
    return {
        "task_key": expected_key,
        "cell_name": expected_name,
        "cluster": expected_key[0],
        "attempt": attempt,
        "four_product_bytes": product_bytes,
        "cell_report_sha256": sha256(report_path),
        "product_hashes": product_hashes,
    }


def snapshot_digest(paths: list[Path]) -> str:
    payload = "\n".join(path.parent.name for path in paths) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit(
    config_path: Path,
    scratch: Path,
    output_path: Path,
    workers: int,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = load_manifest(config)
    manifest_by_key = {canonical_task_key(row): row for row in manifest}
    snapshot_started = datetime.now(UTC).isoformat()
    report_paths = sorted((scratch / "completed").glob("*/cell_report.json"))
    snapshot_names = [path.parent.name for path in report_paths]
    if len(snapshot_names) != len(set(snapshot_names)):
        raise RuntimeError("V19W completed snapshot contains duplicate directory names")
    rows: list[tuple[Path, dict[str, str]]] = []
    for report_path in report_paths:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        key = canonical_task_key(raw)
        if key not in manifest_by_key:
            raise RuntimeError(f"V19W completed snapshot contains an unknown task: {key}")
        rows.append((report_path, manifest_by_key[key]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda item: validate_checkpoint(*item), rows))
    keys = [record["task_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("V19W completed snapshot contains duplicate task keys")
    failed_attempts = list((scratch / "failed_attempts").glob("*"))
    quarantined = list((scratch / "quarantine").glob("*"))
    partial = list((scratch / "partial").glob("*"))
    counts = Counter(record["cluster"] for record in records)
    completed_keys = set(keys)
    manifest_positions = {
        canonical_task_key(row): int(row["production_index"]) for row in manifest
    }
    maximum_completed_index = max(
        (manifest_positions[key] for key in completed_keys), default=0
    )
    missing_before_maximum = [
        {
            "production_index": int(row["production_index"]),
            "batch_id": int(row["batch_id"]),
            "cell_name": cell_name(row),
        }
        for row in manifest
        if int(row["production_index"]) <= maximum_completed_index
        and canonical_task_key(row) not in completed_keys
    ]
    failed_attempts_by_cell = Counter(
        path.name.rsplit("_attempt", maxsplit=1)[0] for path in failed_attempts
    )
    exhausted_failed_cells = sorted(
        name
        for name, count in failed_attempts_by_cell.items()
        if count >= int(config["workload"]["maximum_total_attempts_per_cell"])
        and not (scratch / "completed" / name / "cell_report.json").is_file()
    )
    report = {
        "status": "completed_checkpoint_snapshot_passed_read_only_audit",
        "protocol_version": config["protocol_version"],
        "snapshot_started_utc": snapshot_started,
        "snapshot_completed_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "v19w_runner_sha256": sha256(ROOT / "scripts" / "run_sigma_v19w_full_response_production.py"),
        "scratch_root": str(scratch),
        "snapshot_completed_cells": len(records),
        "expected_final_cells": int(config["workload"]["expected_task_count"]),
        "snapshot_cell_names_sha256": snapshot_digest(report_paths),
        "cluster_counts": dict(sorted(counts.items())),
        "attempt_counts": dict(sorted(Counter(record["attempt"] for record in records).items())),
        "audited_product_files": len(records) * len(PRODUCT_ROLES),
        "audited_product_bytes": sum(record["four_product_bytes"] for record in records),
        "failed_attempt_directories_observed_after_snapshot": len(failed_attempts),
        "failed_attempts_by_cell_observed_after_snapshot": dict(
            sorted(failed_attempts_by_cell.items())
        ),
        "exhausted_failed_cells_observed_after_snapshot": exhausted_failed_cells,
        "quarantine_directories_observed_after_snapshot": len(quarantined),
        "active_partial_directories_observed_after_snapshot": len(partial),
        "maximum_completed_production_index": maximum_completed_index,
        "missing_manifest_tasks_at_or_before_maximum_completed_index": missing_before_maximum,
        "gates": {
            "every_snapshot_cell_is_a_unique_manifest_task": len(keys) == len(set(keys)),
            "every_snapshot_preflight_count_matches_manifest": True,
            "every_snapshot_cell_gate_passes": True,
            "every_snapshot_product_size_and_sha256_matches": True,
            "every_snapshot_pha_link_matches": True,
            "every_snapshot_pha_histogram_audit_is_exact": True,
            "every_snapshot_arf_and_rmf_audit_passes": True,
            "retained_failed_attempts_are_not_completed_products": all(
                not path.is_relative_to(scratch / "completed") for path in failed_attempts
            ),
            "no_quarantine_directory_observed": len(quarantined) == 0,
        },
        "read_only": True,
        "scientific_spectrum_combined_or_fitted": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "claim_boundary": (
            "This is a point-in-time integrity audit of the completed checkpoints present when "
            "the snapshot began. It does not authorize V19X or claim that V19W has completed."
        ),
    }
    if not all(report["gates"].values()):
        report["status"] = "completed_checkpoint_snapshot_failed_closed"
    elif exhausted_failed_cells:
        report["status"] = (
            "completed_checkpoint_snapshot_passed_with_exhausted_base_production_cell"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise RuntimeError("V19W live audit workers must be in 1..4")
    report = audit(
        args.config,
        args.scratch.resolve(),
        args.output.resolve(),
        args.workers,
    )
    print(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
