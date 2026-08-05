#!/usr/bin/env python3
"""Run the four frozen V19V response-throughput pilot cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v17c_integrated_spectra as inherited
import run_sigma_v19p_exact_flux_obs_support as v19p
import run_sigma_v19q_positive_exposure_response_workload as v19q
import run_sigma_v19r_response_commissioning as v19r

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19v_response_throughput_pilot.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19v_response_throughput_pilot"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19v-response-throughput-pilot/v100")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate_parent_hashes(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and v19p.sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19V parent hash mismatch: {value}")


def key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["cluster"]),
        int(row["bin_id"]),
        int(row["obsid"]),
        int(row["ccd_id"]),
    )


def validate_selection(config: dict[str, Any], plan: dict[str, Any]) -> None:
    frozen = {
        key(row): (
            float(row["quantile"]),
            int(row["source_band_events"]),
            int(row["background_band_events"]),
            float(row["blanksky_scale"]),
        )
        for row in config["pilot_cells"]
    }
    planned = {
        key(row): (
            float(row["pilot_quantile"]),
            int(row["source_band_events"]),
            int(row["background_band_events"]),
            float(row["blanksky_scale"]),
        )
        for row in plan["pilot_cells"]
    }
    if frozen != planned:
        raise RuntimeError("V19V pilot cells differ from V19U selection")
    manifest = ROOT / config["parents"]["v19u_manifest"]
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = [
            row for row in csv.DictReader(handle) if row["throughput_pilot"] == "true"
        ]
    if {key(row) for row in manifest_rows} != set(frozen):
        raise RuntimeError("V19V pilot cells differ from V19U manifest flags")


def indexed(rows: list[dict[str, Any]], *fields: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def checked_inventory_product(observation: dict[str, Any], role: str) -> Path:
    matches = [item for item in observation["products"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"V19V expected one {role} product")
    item = matches[0]
    path = resolve(item["path"])
    if path.stat().st_size != int(item["bytes"]) or v19p.sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19V inventory product changed: {path}")
    return path


def checked_region(cluster_row: dict[str, Any], bin_id: int) -> Path:
    suffix = f"/regions/xaf_{bin_id}.reg"
    matches = [
        item
        for item in cluster_row["products"]
        if item["role"] == "spectral_region"
        and item["relative_path"].replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"V19V expected one region {bin_id}")
    item = matches[0]
    path = ROOT / item["relative_path"]
    if path.stat().st_size != int(item["bytes"]) or v19p.sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19V region changed: {path}")
    return path


def checked_repro_product(row: dict[str, Any], suffix: str) -> Path:
    matches = [item for item in row["products"] if item["relative_path"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"V19V expected one repro product ending {suffix}")
    item = matches[0]
    path = Path(row["output_directory"]) / item["relative_path"]
    if path.stat().st_size != int(item["bytes"]) or v19p.sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19V repro product changed: {path}")
    return path


def checked_aspect_list(row: dict[str, Any]) -> Path:
    application = row["application"]
    path = Path(application["corrected_aspect_list"])
    relative = path.relative_to(Path(row["work"])).as_posix()
    matches = [
        item for item in application["products"] if item["relative_path"] == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"V19V corrected aspect list absent from audit: {path}")
    item = matches[0]
    if path.stat().st_size != int(item["bytes"]) or v19p.sha256(path) != item["sha256"]:
        raise RuntimeError(f"V19V corrected aspect list changed: {path}")
    for aspect in application["corrected_aspects"]:
        aspect_path = Path(aspect["path"])
        if not aspect_path.is_file() or v19p.sha256(aspect_path) != aspect["sha256"]:
            raise RuntimeError(f"V19V corrected aspect changed: {aspect_path}")
    return path


def prepare_cell(
    selection: dict[str, Any],
    inventory_rows: dict[tuple[Any, ...], dict[str, Any]],
    source_rows: dict[tuple[Any, ...], dict[str, Any]],
    region_rows: dict[tuple[Any, ...], dict[str, Any]],
    repro_rows: dict[tuple[Any, ...], dict[str, Any]],
    astrometry_rows: dict[tuple[Any, ...], dict[str, Any]],
    scratch: Path,
) -> dict[str, Any]:
    cluster, bin_id, obsid, ccd_id = key(selection)
    inventory = inventory_rows[(cluster, obsid)]
    source_row = source_rows[(cluster,)]
    region_row = region_rows[(cluster,)]
    repro = repro_rows[(cluster, obsid)]
    astrometry = astrometry_rows[(cluster, obsid)]
    science = checked_inventory_product(inventory, "science_event")
    background = checked_inventory_product(inventory, "blanksky_event")
    fov = checked_inventory_product(inventory, "flux_obs_support_fov")
    exposure_path = checked_inventory_product(inventory, "flux_obs_broad_exposure")
    region = checked_region(region_row, bin_id)
    mask = checked_repro_product(repro, "_msk1.fits")
    badpix = checked_repro_product(repro, "_repro_bpix1.fits")
    aspect = checked_aspect_list(astrometry)
    cell_name = f"{cluster}_bin{bin_id}_obs{obsid}_ccd{ccd_id}"
    preflight_env = inherited.isolated_environment(
        os.environ,
        scratch / "pfiles_preflight" / cell_name,
        scratch / "tmp_preflight" / cell_name,
    )
    binmap = v19p.image(v19p.region_product(region_row, "binmap")).astype(int)
    exposure = np.nan_to_num(v19p.image(exposure_path), nan=0.0)
    science_table, _, _, _ = v19q.science_assignments(
        f"{science}[sky=region({fov})]",
        binmap,
        exposure,
        source_row["grid"],
        (500, 7000),
        0.0,
    )
    positive_events = int(science_table.get((bin_id, ccd_id), 0))
    source_filter = (
        f"{science}[ccd_id={ccd_id}]"
        f"[sky=region({fov})][sky=region({region})]"
    )
    source_events = inherited.event_count(source_filter + "[energy=500:7000]", preflight_env)
    corrected_background = (
        scratch / "background_geometry" / cluster / str(obsid) / f"acisf{obsid}_blanksky_geometry.fits"
    )
    background_geometry = inherited.prepare_background_geometry(
        science, background, corrected_background, preflight_env
    )
    background_filter = (
        f"{corrected_background}[ccd_id={ccd_id}][sky=region({region})]"
    )
    background_events = inherited.event_count(
        background_filter + "[energy=500:7000]", preflight_env
    )
    expected_source = int(selection["source_band_events"])
    expected_background = int(selection["background_band_events"])
    if (positive_events, source_events, background_events) != (
        expected_source,
        expected_source,
        expected_background,
    ):
        raise RuntimeError(
            f"V19V preflight mismatch for {cell_name}: "
            f"{positive_events}/{source_events}/{background_events}"
        )
    reference = inherited.event_reference_coordinate(source_filter, science, preflight_env)
    source_chip = inherited.celestial_coordinate_chip(
        science, aspect, reference["ra_deg"], reference["dec_deg"], preflight_env
    )
    background_chip = inherited.celestial_coordinate_chip(
        corrected_background,
        aspect,
        reference["ra_deg"],
        reference["dec_deg"],
        preflight_env,
    )
    reference["science_aspect_chip_id"] = source_chip
    reference["background_aspect_chip_id"] = background_chip
    if reference["events"] != source_events or any(
        chip != ccd_id
        for chip in (reference["dmcoords_chip_id"], source_chip, background_chip)
    ):
        raise RuntimeError(f"V19V response reference is off CCD: {reference}")
    return {
        **selection,
        "cell_name": cell_name,
        "source_filter": source_filter,
        "background_filter": background_filter,
        "aspect": aspect,
        "mask": mask,
        "badpix": badpix,
        "fov": fov,
        "reference": reference,
        "preflight": {
            "positive_exposure_events": positive_events,
            "source_band_events": source_events,
            "background_band_events": background_events,
        },
        "background_geometry": background_geometry,
    }


def execute_attempt(cell: dict[str, Any], scratch: Path, attempt: int) -> dict[str, Any]:
    partial = scratch / "partial" / f"{cell['cell_name']}_attempt{attempt}"
    if partial.exists():
        raise RuntimeError(f"V19V partial attempt already exists: {partial}")
    products = partial / "products"
    logs = partial / "logs"
    products.mkdir(parents=True)
    outroot = products / cell["cell_name"]
    source_pha = outroot.with_suffix(".pi")
    background_pha = outroot.with_name(outroot.name + "_bkg.pi")
    arf = outroot.with_suffix(".arf")
    rmf = outroot.with_suffix(".rmf")
    reference = cell["reference"]
    command = [
        "specextract",
        f"infile={cell['source_filter']}",
        f"outroot={outroot}",
        f"bkgfile={cell['background_filter']}",
        f"asp=@{cell['aspect']}",
        f"mskfile={cell['mask']}",
        f"badpixfile={cell['badpix']}",
        "dafile=CALDB",
        "bkgresp=no",
        "weight=yes",
        "weight_rmf=yes",
        "resp_pos=CENTROID",
        f"refcoord={reference['ra_deg']:.14f},{reference['dec_deg']:.14f}",
        "correctpsf=no",
        "combine=no",
        "grouptype=NONE",
        "binspec=NONE",
        "bkg_grouptype=NONE",
        "bkg_binspec=NONE",
        "energy=0.3:11.0:0.01",
        "energy_wmap=500:7000",
        "binwmap=det=8",
        "binarfwmap=1",
        "parallel=no",
        "nproc=1",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    env = inherited.isolated_environment(
        os.environ,
        scratch / "pfiles_cells" / cell["cell_name"] / f"attempt{attempt}",
        scratch / "tmp_cells" / cell["cell_name"] / f"attempt{attempt}",
    )
    start_monotonic = time.perf_counter()
    start_utc = datetime.now(UTC).isoformat()
    step = inherited.run_step(
        command, logs / "specextract.log", [source_pha, background_pha, arf, rmf], env
    )
    scaling = inherited.verify_blanksky_scaling(
        source_pha, background_pha, float(cell["blanksky_scale"]), env
    )
    source_audit = v19r.pha_channel_audit(source_pha, cell["source_filter"])
    background_audit = v19r.pha_channel_audit(background_pha, cell["background_filter"])
    response = v19r.response_audit(arf, rmf)
    links = v19r.pha_links(source_pha, env)
    end_monotonic = time.perf_counter()
    four_products = (source_pha, background_pha, arf, rmf)
    gates = {
        "source_and_background_pha_channel_histograms_match_events": source_audit[
            "exact"
        ]
        and background_audit["exact"],
        "arf_is_finite_positive": response["arf_finite"]
        and response["arf_positive_bins"] > 0,
        "rmf_is_finite_nonzero": response["rmf_finite"]
        and response["rmf_nonzero_elements"] > 0,
        "pha_links_present": all(value and value.upper() != "NONE" for value in links.values()),
        "blanksky_scale_exact": scaling[
            "effective_scale_relative_error_from_BKGSCALn"
        ]
        <= 1e-6,
    }
    if not all(gates.values()):
        raise RuntimeError(f"V19V cell audit failed: {gates}")
    completed = scratch / "completed" / cell["cell_name"]
    if completed.exists():
        raise RuntimeError(f"V19V completed cell already exists: {completed}")
    record = {
        "cell_name": cell["cell_name"],
        "cluster": cell["cluster"],
        "bin_id": int(cell["bin_id"]),
        "obsid": int(cell["obsid"]),
        "ccd_id": int(cell["ccd_id"]),
        "quantile": float(cell["quantile"]),
        "attempt": attempt,
        "start_utc": start_utc,
        "start_monotonic_seconds": start_monotonic,
        "end_monotonic_seconds": end_monotonic,
        "elapsed_seconds": end_monotonic - start_monotonic,
        "preflight": cell["preflight"],
        "response_reference": cell["reference"],
        "step": step,
        "blanksky_scaling": scaling,
        "source_pha_channel_audit": source_audit,
        "background_pha_channel_audit": background_audit,
        "response_audit": response,
        "source_pha_links": links,
        "four_product_bytes": sum(path.stat().st_size for path in four_products),
        "products": {
            "source_pha": {"name": source_pha.name, "bytes": source_pha.stat().st_size, "sha256": v19p.sha256(source_pha)},
            "background_pha": {"name": background_pha.name, "bytes": background_pha.stat().st_size, "sha256": v19p.sha256(background_pha)},
            "arf": {"name": arf.name, "bytes": arf.stat().st_size, "sha256": v19p.sha256(arf)},
            "rmf": {"name": rmf.name, "bytes": rmf.stat().st_size, "sha256": v19p.sha256(rmf)},
        },
        "gates": gates,
    }
    (partial / "cell_report.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    completed.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(completed)
    record["completed_directory"] = str(completed)
    return record


def run_cell(cell: dict[str, Any], scratch: Path, maximum_attempts: int) -> dict[str, Any]:
    failures = []
    for attempt in range(1, maximum_attempts + 1):
        try:
            record = execute_attempt(cell, scratch, attempt)
            return {**record, "failures_before_success": failures}
        except Exception as exc:  # retain complete pilot evidence before deciding
            partial = scratch / "partial" / f"{cell['cell_name']}_attempt{attempt}"
            failed = scratch / "failed_attempts" / f"{cell['cell_name']}_attempt{attempt}"
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
        "cell_name": cell["cell_name"],
        "cluster": cell["cluster"],
        "bin_id": int(cell["bin_id"]),
        "obsid": int(cell["obsid"]),
        "ccd_id": int(cell["ccd_id"]),
        "failures": failures,
        "passed": False,
    }


def maximum_concurrency(records: list[dict[str, Any]]) -> int:
    events = []
    for record in records:
        events.append((float(record["start_monotonic_seconds"]), 1))
        events.append((float(record["end_monotonic_seconds"]), -1))
    active = maximum = 0
    for _, change in sorted(events, key=lambda item: (item[0], -item[1])):
        active += change
        maximum = max(maximum, active)
    return maximum


def snapshot(records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        completed = Path(record["completed_directory"])
        destination = output / "frozen_products" / record["cell_name"]
        products = {}
        for role, item in record["products"].items():
            products[role] = inherited.copy_snapshot(
                completed / "products" / item["name"], destination / item["name"]
            )
        products["specextract_log"] = inherited.copy_snapshot(
            completed / "logs" / "specextract.log", destination / "specextract.log"
        )
        products["cell_report"] = inherited.copy_snapshot(
            completed / "cell_report.json", destination / "cell_report.json"
        )
        rows.append({"cell_name": record["cell_name"], "products": products})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    scratch = args.scratch.resolve()
    config = v19p.load_json(config_path)
    validate_parent_hashes(config)
    plan = v19p.load_json(ROOT / config["parents"]["v19u_report"])
    validate_selection(config, plan)
    if not plan["throughput_pilot_authorized"] or plan["full_production_authorized"]:
        raise RuntimeError("V19U authorization state is not the frozen pilot-only state")

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
    prepared = [
        prepare_cell(
            row,
            inventory_rows,
            source_rows,
            region_rows,
            repro_rows,
            astrometry_rows,
            scratch,
        )
        for row in config["pilot_cells"]
    ]

    pilot_start = time.perf_counter()
    records = []
    maximum_workers = int(config["execution"]["maximum_concurrent_cells"])
    with ThreadPoolExecutor(max_workers=maximum_workers) as pool:
        futures = {
            pool.submit(
                run_cell,
                cell,
                scratch,
                int(config["execution"]["maximum_attempts_per_cell"]),
            ): cell["cell_name"]
            for cell in prepared
        }
        for future in as_completed(futures):
            records.append(future.result())
    pilot_elapsed = time.perf_counter() - pilot_start
    records.sort(key=lambda row: row["cell_name"])
    passed_records = [row for row in records if row.get("completed_directory")]
    observed_concurrency = maximum_concurrency(passed_records) if passed_records else 0
    pilot_snapshots = snapshot(passed_records, output)
    snapshot_count = sum(len(row["products"]) for row in pilot_snapshots)
    maximum_bytes = int(config["resource_gates"]["maximum_four_product_bytes_per_pilot_cell"])
    gates = {
        "all_parent_and_external_input_hashes_exact": True,
        "pilot_cells_exactly_equal_v19u_selection": True,
        "pilot_cells_have_unique_observation_ccd_pairs": len(
            {(int(row["obsid"]), int(row["ccd_id"])) for row in config["pilot_cells"]}
        )
        == len(config["pilot_cells"]),
        "positive_exposure_source_and_background_preflight_counts_exact": all(
            row["preflight"]["positive_exposure_events"]
            == row["preflight"]["source_band_events"]
            for row in prepared
        ),
        "response_reference_maps_to_the_frozen_ccd_for_science_and_background": all(
            all(
                int(value) == int(row["ccd_id"])
                for value in (
                    row["reference"]["dmcoords_chip_id"],
                    row["reference"]["science_aspect_chip_id"],
                    row["reference"]["background_aspect_chip_id"],
                )
            )
            for row in prepared
        ),
        "all_four_cells_pass_on_at_most_two_unchanged_attempts": len(passed_records)
        == len(config["pilot_cells"])
        and all(int(row["attempt"]) <= 2 for row in passed_records),
        "source_and_background_pha_channel_histograms_match_events": all(
            row["source_pha_channel_audit"]["exact"]
            and row["background_pha_channel_audit"]["exact"]
            for row in passed_records
        ),
        "arf_is_finite_positive_and_rmf_is_finite_nonzero": all(
            row["response_audit"]["arf_finite"]
            and row["response_audit"]["arf_positive_bins"] > 0
            and row["response_audit"]["rmf_finite"]
            and row["response_audit"]["rmf_nonzero_elements"] > 0
            for row in passed_records
        ),
        "pha_background_arf_and_rmf_links_are_present": all(
            all(value and value.upper() != "NONE" for value in row["source_pha_links"].values())
            for row in passed_records
        ),
        "effective_blanksky_scale_matches_manifest_within_1e_6_relative": all(
            row["blanksky_scaling"]["effective_scale_relative_error_from_BKGSCALn"]
            <= 1e-6
            for row in passed_records
        ),
        "at_least_two_cells_overlap_in_execution": observed_concurrency
        >= int(config["execution"]["minimum_observed_concurrency"]),
        "pilot_wall_time_at_most_600_seconds": pilot_elapsed
        <= float(config["execution"]["maximum_pilot_wall_seconds"]),
        "each_cell_four_product_bytes_at_most_twice_v19r": all(
            int(row["four_product_bytes"]) <= maximum_bytes for row in passed_records
        ),
        "all_passing_products_are_hashed_and_snapshotted": len(pilot_snapshots)
        == len(passed_records)
        and snapshot_count == 6 * len(passed_records),
    }
    passed = all(gates.values())
    median_bytes = median(row["four_product_bytes"] for row in passed_records) if passed_records else math.nan
    throughput = len(passed_records) / pilot_elapsed if pilot_elapsed > 0 else math.nan
    report = {
        "status": (
            "throughput_pilot_passed_and_full_response_production_authorized"
            if passed
            else "throughput_pilot_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19p.sha256(config_path),
        "runner_sha256": v19p.sha256(Path(__file__).resolve()),
        "pilot_wall_seconds": pilot_elapsed,
        "observed_maximum_concurrency": observed_concurrency,
        "successful_cells_per_second": throughput,
        "projected_full_response_hours_at_pilot_throughput": (
            int(config["resource_gates"]["full_workload_cells"]) / throughput / 3600
            if math.isfinite(throughput) and throughput > 0
            else None
        ),
        "median_four_product_bytes": median_bytes,
        "projected_full_response_bytes_at_pilot_median": (
            median_bytes * int(config["resource_gates"]["full_workload_cells"])
            if math.isfinite(median_bytes)
            else None
        ),
        "cells": records,
        "frozen_snapshots": pilot_snapshots,
        "gates": gates,
        "full_response_production_authorized": passed,
        "additional_temperature_density_mach_or_speed_fitted": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(
        f"cells/wall/concurrency: {len(passed_records)}/{pilot_elapsed:.3f}s/"
        f"{observed_concurrency}"
    )
    print(
        "projected full response hours: "
        f"{report['projected_full_response_hours_at_pilot_throughput']}"
    )
    print(f"report: {report_path}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
