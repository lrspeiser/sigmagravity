#!/usr/bin/env python3
"""Clean v19F Chandra events and construct source-only blank-sky controls."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import sigma_v19f_chandra_common as common

ROOT = common.ROOT
DEFAULT_CONFIG = common.DEFAULT_CONFIG
DEFAULT_REPRO = ROOT / "results" / "sigma_v19f_chandra_repro" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19f_chandra_cleaning"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19f-chandra")
SHARED_CLEANING = ROOT / "scripts" / "run_sigma_v17a_chandra_cleaning.py"


def shared_module():
    return common.load_module(SHARED_CLEANING, "sigma_v17a_cleaning_shared")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repro", type=Path, default=DEFAULT_REPRO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config, _, _ = common.validate_protocol(config_path)
    resolved_config = common.resolved_shared_config(config)
    repro_path = args.repro.resolve()
    repro = common.load_json(repro_path)
    if repro["config_sha256"] != common.sha256(config_path):
        raise RuntimeError("v19F repro output does not match the frozen protocol")
    if repro["observation_count"] != int(config["gates"]["required_observations"]):
        raise RuntimeError("all v19F observations must be reprocessed before cleaning")
    if repro["lensing_target_opened"] is not False:
        raise RuntimeError("v19F repro unexpectedly opened a lensing target")
    if repro["event_images_inspected"] is not False:
        raise RuntimeError("v19F repro unexpectedly inspected an event image")

    scratch = args.scratch.resolve()
    if str(scratch) != repro["scratch"]:
        raise RuntimeError("v19F cleaning scratch root must match repro provenance")
    jobs = args.jobs or int(config["event_reprocessing"]["parallel_observations"])
    maximum_jobs = int(config["gates"]["maximum_parallel_observations"])
    if jobs < 1 or jobs > maximum_jobs:
        raise ValueError(f"v19F permits between one and {maximum_jobs} jobs")

    shared = shared_module()
    completed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                shared.process_observation,
                row,
                scratch,
                resolved_config,
            ): (row["cluster"], row["obsid"])
            for row in repro["observations"]
        }
        for future in as_completed(futures):
            cluster, obsid = futures[future]
            completed.append(future.result())
            print(f"cleaned {cluster}/{obsid}", flush=True)
    completed.sort(key=lambda row: (row["cluster"], row["obsid"]))

    minimum_fraction = min(row["retained_exposure_fraction"] for row in completed)
    required_fraction = float(config["gates"]["minimum_retained_exposure_fraction"])
    if minimum_fraction < required_fraction:
        raise RuntimeError("v19F pair failed the frozen retained-exposure gate")
    if any(not row["blanksky_scaling"] for row in completed):
        raise RuntimeError("a v19F observation lacks blank-sky scaling keywords")

    report = {
        "status": "all_frozen_v19f_chandra_observations_flare_cleaned_with_blanksky",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": common.sha256(config_path),
        "repro_report_sha256": common.sha256(repro_path),
        "scratch": str(scratch),
        "jobs": jobs,
        "observations": completed,
        "observation_count": len(completed),
        "clean_exposure_seconds": sum(
            row["clean_exposure_seconds"] for row in completed
        ),
        "minimum_retained_exposure_fraction": minimum_fraction,
        "product_files": sum(row["product_files"] for row in completed),
        "product_bytes": sum(row["product_bytes"] for row in completed),
        "astrometry_completed": False,
        "event_images_visually_inspected": False,
        "shock_front_fitted": False,
        "source_constructed": False,
        "lensing_target_opened": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "observation_count": report["observation_count"],
                "clean_exposure_seconds": report["clean_exposure_seconds"],
                "minimum_retained_exposure_fraction": minimum_fraction,
                "product_files": report["product_files"],
                "product_bytes": report["product_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
