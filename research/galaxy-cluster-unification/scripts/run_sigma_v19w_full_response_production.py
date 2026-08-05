#!/usr/bin/env python3
"""Run the checkpointed 5,082-cell V19W response production archive."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v17c_integrated_spectra as inherited
import run_sigma_v19p_exact_flux_obs_support as v19p
import run_sigma_v19v_response_throughput_pilot as v19v

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19w_full_response_production.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19w_full_response_production"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19w-response-production/v100")


def validate_parent_hashes(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and v19p.sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19W parent hash mismatch: {value}")


def task_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["cluster"]),
        int(row["bin_id"]),
        int(row["obsid"]),
        int(row["ccd_id"]),
    )


def cell_name(row: dict[str, Any]) -> str:
    cluster, bin_id, obsid, ccd_id = task_key(row)
    return f"{cluster}_bin{bin_id}_obs{obsid}_ccd{ccd_id}"


def load_manifest(config: dict[str, Any]) -> list[dict[str, str]]:
    path = ROOT / config["parents"]["v19u_manifest"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["production_index"]))
    keys = [task_key(row) for row in rows]
    batches = Counter(int(row["batch_id"]) for row in rows)
    counts = Counter(row["cluster"] for row in rows)
    expected_counts = config["workload"]["expected_task_count_by_cluster"]
    if (
        len(rows) != int(config["workload"]["expected_task_count"])
        or len(keys) != len(set(keys))
        or sorted(int(row["production_index"]) for row in rows)
        != list(range(1, len(rows) + 1))
        or len(batches) != int(config["workload"]["expected_batch_count"])
        or max(batches.values()) > int(config["workload"]["batch_size"])
        or any(counts[name] != expected for name, expected in expected_counts.items())
    ):
        raise RuntimeError("V19W manifest structure differs from frozen workload")
    edge = config["known_positive_exposure_edge_case"]
    if (
        task_key(edge) in set(keys)
        or edge["bin_passed_v19m_region_admission"]
        or edge["task_exists_in_v19u_manifest"]
    ):
        raise RuntimeError("V19W zero-exposure edge-case status changed")
    return rows


def indexed(rows: list[dict[str, Any]], *fields: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def observation_contexts(
    config: dict[str, Any], manifest: list[dict[str, str]], scratch: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    inventory = v19p.load_json(ROOT / config["parents"]["input_inventory"])
    source = v19p.load_json(ROOT / config["parents"]["source_map_report"])
    regions = v19p.load_json(ROOT / config["parents"]["region_report"])
    repro = v19p.load_json(ROOT / config["parents"]["repro_report"])
    astrometry = v19p.load_json(ROOT / config["parents"]["astrometry_report"])
    inventory_rows = indexed(
        [
            {**observation, "cluster": cluster["cluster"]}
            for cluster in inventory["clusters"]
            for observation in cluster["observations"]
        ],
        "cluster",
        "obsid",
    )
    source_rows = indexed(source["clusters"], "cluster")
    region_rows = indexed(regions["clusters"], "cluster")
    repro_rows = indexed(repro["observations"], "cluster", "obsid")
    astrometry_rows = indexed(astrometry["observations"], "cluster", "obsid")
    contexts = {}
    for cluster, obsid in sorted({(row["cluster"], int(row["obsid"])) for row in manifest}):
        inventory_row = inventory_rows[(cluster, obsid)]
        repro_row = repro_rows[(cluster, obsid)]
        astrometry_row = astrometry_rows[(cluster, obsid)]
        science = v19v.checked_inventory_product(inventory_row, "science_event")
        background = v19v.checked_inventory_product(inventory_row, "blanksky_event")
        fov = v19v.checked_inventory_product(inventory_row, "flux_obs_support_fov")
        mask = v19v.checked_repro_product(repro_row, "_msk1.fits")
        badpix = v19v.checked_repro_product(repro_row, "_repro_bpix1.fits")
        aspect = v19v.checked_aspect_list(astrometry_row)
        env = inherited.isolated_environment(
            os.environ,
            scratch / "pfiles_geometry" / cluster / str(obsid),
            scratch / "tmp_geometry" / cluster / str(obsid),
        )
        corrected_background = (
            scratch
            / "background_geometry"
            / cluster
            / str(obsid)
            / f"acisf{obsid}_blanksky_geometry.fits"
        )
        geometry = inherited.prepare_background_geometry(
            science, background, corrected_background, env
        )
        contexts[(cluster, obsid)] = {
            "science": science,
            "background": corrected_background,
            "fov": fov,
            "mask": mask,
            "badpix": badpix,
            "aspect": aspect,
            "source_row": source_rows[(cluster,)],
            "region_row": region_rows[(cluster,)],
            "background_geometry": geometry,
        }
    return contexts


def prepare_task(
    row: dict[str, str], context: dict[str, Any], scratch: Path
) -> dict[str, Any]:
    cluster, bin_id, obsid, ccd_id = task_key(row)
    name = cell_name(row)
    region = v19v.checked_region(context["region_row"], bin_id)
    env = inherited.isolated_environment(
        os.environ,
        scratch / "pfiles_preflight" / name,
        scratch / "tmp_preflight" / name,
    )
    source_filter = (
        f"{context['science']}[ccd_id={ccd_id}]"
        f"[sky=region({context['fov']})][sky=region({region})]"
    )
    background_filter = (
        f"{context['background']}[ccd_id={ccd_id}][sky=region({region})]"
    )
    source_events = inherited.event_count(source_filter + "[energy=500:7000]", env)
    background_events = inherited.event_count(
        background_filter + "[energy=500:7000]", env
    )
    if source_events != int(row["source_band_events"]) or background_events != int(
        row["background_band_events"]
    ):
        raise RuntimeError(
            f"V19W exact count preflight failed for {name}: "
            f"{source_events}/{background_events} vs "
            f"{row['source_band_events']}/{row['background_band_events']}"
        )
    reference = inherited.event_reference_coordinate(source_filter, context["science"], env)
    source_chip = inherited.celestial_coordinate_chip(
        context["science"],
        context["aspect"],
        reference["ra_deg"],
        reference["dec_deg"],
        env,
    )
    background_chip = inherited.celestial_coordinate_chip(
        context["background"],
        context["aspect"],
        reference["ra_deg"],
        reference["dec_deg"],
        env,
    )
    reference["science_aspect_chip_id"] = source_chip
    reference["background_aspect_chip_id"] = background_chip
    if reference["events"] != source_events or any(
        value != ccd_id
        for value in (reference["dmcoords_chip_id"], source_chip, background_chip)
    ):
        raise RuntimeError(f"V19W response reference is off CCD for {name}: {reference}")
    return {
        "cluster": cluster,
        "bin_id": bin_id,
        "obsid": obsid,
        "ccd_id": ccd_id,
        "quantile": -1.0,
        "source_band_events": source_events,
        "background_band_events": background_events,
        "blanksky_scale": float(row["blanksky_scale"]),
        "cell_name": name,
        "source_filter": source_filter,
        "background_filter": background_filter,
        "aspect": context["aspect"],
        "mask": context["mask"],
        "badpix": context["badpix"],
        "fov": context["fov"],
        "reference": reference,
        "preflight": {
            "source_band_events": source_events,
            "background_band_events": background_events,
        },
        "background_geometry": context["background_geometry"],
    }


def completed_record(row: dict[str, str], scratch: Path) -> dict[str, Any] | None:
    name = cell_name(row)
    completed = scratch / "completed" / name
    report_path = completed / "cell_report.json"
    if not completed.exists():
        return None
    try:
        record = json.loads(report_path.read_text(encoding="utf-8"))
        expected = task_key(row)
        actual = (
            record["cluster"],
            int(record["bin_id"]),
            int(record["obsid"]),
            int(record["ccd_id"]),
        )
        if actual != expected or not all(record["gates"].values()):
            raise RuntimeError("task key or recorded gate mismatch")
        if record["preflight"]["source_band_events"] != int(row["source_band_events"]):
            raise RuntimeError("source count mismatch")
        if record["preflight"]["background_band_events"] != int(
            row["background_band_events"]
        ):
            raise RuntimeError("background count mismatch")
        for item in record["products"].values():
            path = completed / "products" / item["name"]
            if path.stat().st_size != int(item["bytes"]) or v19p.sha256(path) != item["sha256"]:
                raise RuntimeError(f"product mismatch: {path}")
        return {**record, "completed_directory": str(completed), "reused": True}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        quarantine = scratch / "quarantine" / f"{name}_{time.time_ns()}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        completed.rename(quarantine)
        return {
            "invalid_checkpoint": True,
            "exception": f"{type(exc).__name__}: {exc}",
            "quarantined_directory": str(quarantine),
        }


def failed_attempt_count(name: str, scratch: Path) -> int:
    directory = scratch / "failed_attempts"
    return len(list(directory.glob(f"{name}_attempt*"))) if directory.exists() else 0


def execute_task(
    row: dict[str, str],
    context: dict[str, Any],
    scratch: Path,
    maximum_attempts: int,
) -> dict[str, Any]:
    cached = completed_record(row, scratch)
    invalid_checkpoint = None
    if cached and not cached.get("invalid_checkpoint"):
        return cached
    if cached:
        invalid_checkpoint = cached
    prepared = prepare_task(row, context, scratch)
    name = prepared["cell_name"]
    failures = []
    used = failed_attempt_count(name, scratch)
    for attempt in range(used + 1, maximum_attempts + 1):
        try:
            record = v19v.execute_attempt(prepared, scratch, attempt)
            return {
                **record,
                "reused": False,
                "failures_before_success": failures,
                "invalid_checkpoint": invalid_checkpoint,
            }
        except Exception as exc:  # preserve every unchanged production attempt
            partial = scratch / "partial" / f"{name}_attempt{attempt}"
            failed = scratch / "failed_attempts" / f"{name}_attempt{attempt}"
            if partial.exists():
                failed.parent.mkdir(parents=True, exist_ok=True)
                partial.rename(failed)
            failures.append(
                {
                    "attempt": attempt,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "retained_directory": str(failed) if failed.exists() else None,
                }
            )
    return {
        "cell_name": name,
        "cluster": row["cluster"],
        "bin_id": int(row["bin_id"]),
        "obsid": int(row["obsid"]),
        "ccd_id": int(row["ccd_id"]),
        "passed": False,
        "reused": False,
        "failures": failures,
        "invalid_checkpoint": invalid_checkpoint,
    }


def compact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "cell_name",
            "cluster",
            "bin_id",
            "obsid",
            "ccd_id",
            "attempt",
            "elapsed_seconds",
            "four_product_bytes",
            "reused",
            "passed",
            "failures",
            "failures_before_success",
            "invalid_checkpoint",
        )
        if key in record
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress_snapshot(
    config: dict[str, Any],
    config_path: Path,
    scratch: Path,
    current_batch: int | None,
    invocation_results: list[dict[str, Any]],
    start_monotonic: float,
) -> dict[str, Any]:
    completed = list((scratch / "completed").glob("*/cell_report.json"))
    failed = list((scratch / "failed_attempts").glob("*")) if (scratch / "failed_attempts").exists() else []
    return {
        "status": "response_production_running",
        "protocol_version": config["protocol_version"],
        "updated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19p.sha256(config_path),
        "runner_sha256": v19p.sha256(Path(__file__).resolve()),
        "current_batch": current_batch,
        "completed_cells": len(completed),
        "expected_cells": int(config["workload"]["expected_task_count"]),
        "failed_attempt_directories": len(failed),
        "invocation_elapsed_seconds": time.perf_counter() - start_monotonic,
        "latest_results": [compact(row) for row in invocation_results[-64:]],
        "temperature_density_mach_or_speed_fitted": False,
        "gravity_formula_or_parameter_changed": False,
    }


def write_product_index(
    manifest: list[dict[str, str]], scratch: Path, path: Path
) -> tuple[int, int]:
    fields = [
        "cluster",
        "bin_id",
        "obsid",
        "ccd_id",
        "cell_name",
        "attempt",
        "source_pha_sha256",
        "background_pha_sha256",
        "arf_sha256",
        "rmf_sha256",
        "four_product_bytes",
    ]
    rows = []
    total_bytes = 0
    for task in manifest:
        record = completed_record(task, scratch)
        if not record or record.get("invalid_checkpoint"):
            continue
        products = record["products"]
        total_bytes += int(record["four_product_bytes"])
        rows.append(
            {
                "cluster": task["cluster"],
                "bin_id": task["bin_id"],
                "obsid": task["obsid"],
                "ccd_id": task["ccd_id"],
                "cell_name": record["cell_name"],
                "attempt": record["attempt"],
                "source_pha_sha256": products["source_pha"]["sha256"],
                "background_pha_sha256": products["background_pha"]["sha256"],
                "arf_sha256": products["arf"]["sha256"],
                "rmf_sha256": products["rmf"]["sha256"],
                "four_product_bytes": record["four_product_bytes"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--batch-start", type=int, default=1)
    parser.add_argument("--batch-stop", type=int, default=80)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    scratch = args.scratch.resolve()
    config = v19p.load_json(config_path)
    validate_parent_hashes(config)
    pilot = v19p.load_json(ROOT / config["parents"]["v19v_report"])
    if not pilot["full_response_production_authorized"]:
        raise RuntimeError("V19V did not authorize full response production")
    manifest = load_manifest(config)
    if args.status_only:
        completed = len(list((scratch / "completed").glob("*/cell_report.json")))
        print(f"completed: {completed}/{len(manifest)}")
        progress = output / "progress.json"
        if progress.exists():
            print(progress.read_text(encoding="utf-8"))
        return
    expected_batches = int(config["workload"]["expected_batch_count"])
    if not (1 <= args.batch_start <= args.batch_stop <= expected_batches):
        raise RuntimeError("V19W batch interval is outside 1..80")
    scratch.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch).free
    if free_bytes < int(config["resources"]["minimum_free_bytes_at_launch"]):
        raise RuntimeError(f"V19W free-space gate failed: {free_bytes}")
    contexts = observation_contexts(config, manifest, scratch)
    invocation_start = time.perf_counter()
    invocation_results: list[dict[str, Any]] = []
    progress_path = output / "progress.json"
    maximum_workers = int(config["workload"]["maximum_concurrent_cells"])
    maximum_attempts = int(config["workload"]["maximum_total_attempts_per_cell"])
    for batch_id in range(args.batch_start, args.batch_stop + 1):
        batch = [row for row in manifest if int(row["batch_id"]) == batch_id]
        with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
            futures = {
                pool.submit(
                    execute_task,
                    row,
                    contexts[(row["cluster"], int(row["obsid"]))],
                    scratch,
                    maximum_attempts,
                ): row
                for row in batch
            }
            for future in as_completed(futures):
                try:
                    invocation_results.append(future.result())
                except Exception as exc:
                    row = futures[future]
                    invocation_results.append(
                        {
                            "cell_name": cell_name(row),
                            "cluster": row["cluster"],
                            "bin_id": int(row["bin_id"]),
                            "obsid": int(row["obsid"]),
                            "ccd_id": int(row["ccd_id"]),
                            "passed": False,
                            "exception": f"{type(exc).__name__}: {exc}",
                        }
                    )
        atomic_json(
            progress_path,
            progress_snapshot(
                config,
                config_path,
                scratch,
                batch_id,
                invocation_results,
                invocation_start,
            ),
        )
        print(
            f"batch {batch_id}/{args.batch_stop}; "
            f"completed={len(list((scratch / 'completed').glob('*/cell_report.json')))}"
        )
        sys.stdout.flush()

    full_interval = args.batch_start == 1 and args.batch_stop == expected_batches
    index_path = output / "product_index.csv"
    indexed_cells, product_bytes = write_product_index(manifest, scratch, index_path)
    failed_cells = [row for row in invocation_results if row.get("passed") is False]
    completed = indexed_cells
    all_complete = completed == len(manifest)
    final = {
        "status": (
            "all_response_cells_passed_and_regional_spectral_fitting_authorized"
            if all_complete
            else "response_production_incomplete"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19p.sha256(config_path),
        "runner_sha256": v19p.sha256(Path(__file__).resolve()),
        "batch_interval": [args.batch_start, args.batch_stop],
        "full_interval_requested": full_interval,
        "launch_free_bytes": free_bytes,
        "invocation_elapsed_seconds": time.perf_counter() - invocation_start,
        "completed_cells": completed,
        "expected_cells": len(manifest),
        "failed_cells_this_invocation": [compact(row) for row in failed_cells],
        "product_bytes": product_bytes,
        "product_index": {
            "path": index_path.relative_to(ROOT).as_posix(),
            "bytes": index_path.stat().st_size,
            "sha256": v19p.sha256(index_path),
            "rows": indexed_cells,
        },
        "gates": {
            "all_parent_hashes_exact_and_v19v_authorized": True,
            "manifest_has_5082_unique_tasks_in_80_batches": len(manifest) == 5082,
            "known_zero_exposure_event_has_no_admitted_task": task_key(
                config["known_positive_exposure_edge_case"]
            )
            not in {task_key(row) for row in manifest},
            "launch_free_space_passed": free_bytes
            >= int(config["resources"]["minimum_free_bytes_at_launch"]),
            "all_5082_completed_checkpoints_hash_exactly": all_complete,
            "product_index_has_5082_unique_rows": indexed_cells == len(manifest),
            "no_failed_cell_remains": not failed_cells and all_complete,
        },
        "regional_spectral_fitting_authorized": all_complete,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    atomic_json(output / "report.json", final)
    final_progress = progress_snapshot(
        config, config_path, scratch, None, invocation_results, invocation_start
    )
    final_progress["status"] = final["status"]
    atomic_json(progress_path, final_progress)
    print(f"status: {final['status']}")
    print(f"completed: {completed}/{len(manifest)}")
    print(f"product bytes: {product_bytes}")
    if full_interval and not all_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
