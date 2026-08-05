#!/usr/bin/env python3
"""Reprocess the frozen Sigma v19F source-only Chandra observations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import sigma_v19f_chandra_common as common

ROOT = common.ROOT
DEFAULT_CONFIG = common.DEFAULT_CONFIG
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19f_chandra_repro"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19f-chandra")
SHARED_REPRO = ROOT / "scripts" / "run_sigma_v17a_chandra_repro.py"


def shared_module():
    return common.load_module(SHARED_REPRO, "sigma_v17a_repro_shared")


def process_observation(
    cluster: str,
    obsid: int,
    raw_directory: Path,
    scratch: Path,
    config: dict,
    shared_config: dict,
    acquisition_records: list[dict],
) -> dict:
    shared = shared_module()
    source_directory = raw_directory / cluster / str(obsid)
    input_directory = scratch / "input" / cluster / str(obsid)
    output_directory = scratch / "repro" / cluster / str(obsid)
    pfiles = scratch / "pfiles" / cluster / str(obsid)
    tmp = scratch / "tmp" / cluster / str(obsid)
    logs = scratch / "logs" / cluster
    logs.mkdir(parents=True, exist_ok=True)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{obsid}_chandra_repro.log"
    env = shared.isolated_environment(os.environ, pfiles, tmp)

    staging = shared.stage_observation_inputs(
        cluster,
        obsid,
        source_directory,
        input_directory,
        acquisition_records,
    )
    archived_evt2 = [
        ROOT / row["relative_path"]
        for row in acquisition_records
        if row["role"] == "evt2"
    ]
    if len(archived_evt2) != 1:
        raise RuntimeError(f"expected one archived evt2 for {cluster}/{obsid}")
    archive_mode = shared.dmkeypar(archived_evt2[0], "DATAMODE", env)
    expected_mode = common.declared_mode(config, cluster, obsid)
    if archive_mode != expected_mode:
        raise RuntimeError(
            f"archived DATAMODE changed for {cluster}/{obsid}: "
            f"{archive_mode!r} != {expected_mode!r}"
        )

    existing_events = sorted(output_directory.glob("*repro_evt2.fits*"))
    reused = bool(existing_events)
    if output_directory.exists() and not reused:
        raise RuntimeError(f"partial output exists without repro event: {output_directory}")
    repro = shared_config["event_reprocessing"]
    check_vf_pha = expected_mode == "VFAINT"
    command = [
        "chandra_repro",
        f"indir={input_directory}",
        f"outdir={output_directory}",
        f"check_vf_pha={'yes' if check_vf_pha else 'no'}",
        f"pix_adj={repro['pix_adj']}",
        "set_ardlib=no",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    if reused:
        if not log_path.is_file():
            log_path.write_text(
                "reused complete output from an earlier invocation\n",
                encoding="utf-8",
            )
    else:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"chandra_repro failed for {cluster}/{obsid}; see {log_path}"
            )

    events = sorted(output_directory.glob("*repro_evt2.fits*"))
    if len(events) != 1:
        raise RuntimeError(
            f"expected one repro event for {cluster}/{obsid}, found {len(events)}"
        )
    event = events[0]
    inspection = shared.inspect_event(event, env, config["runtime"]["caldb_main"])
    if inspection["header"]["OBS_ID"] != str(obsid):
        raise RuntimeError(f"reprocessed OBS_ID mismatch: {event}")
    if inspection["header"]["DATAMODE"] != expected_mode:
        raise RuntimeError(f"reprocessed DATAMODE mismatch: {event}")
    if not inspection["current_caldb_comment_present"]:
        raise RuntimeError(f"current CALDB provenance is absent: {event}")
    repro_history = shared.command_output(
        ["dmhistory", str(event), "chandra_repro"], env=env
    )
    expected_history = f'check_vf_pha="{"yes" if check_vf_pha else "no"}"'
    if expected_history not in repro_history:
        raise RuntimeError(f"declared VF cleaning rule was not recorded: {event}")

    products = shared.output_inventory(output_directory)
    return {
        "cluster": cluster,
        "obsid": obsid,
        "archive_datamode": archive_mode,
        "check_vf_pha_requested": check_vf_pha,
        "check_vf_pha_history_value": "yes" if check_vf_pha else "no",
        "check_vf_pha_history_present": True,
        "input_directory": str(input_directory),
        "staging": staging,
        "output_directory": str(output_directory),
        "command": command,
        "reused": reused,
        "log": str(log_path),
        "log_sha256": common.sha256(log_path),
        "event": inspection,
        "products": products,
        "product_files": len(products),
        "product_bytes": sum(row["bytes"] for row in products),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config, acquisition, runtime = common.validate_protocol(config_path)
    shared_config = common.resolved_shared_config(config)
    acquisition_config = common.load_json(
        ROOT / config["parents"]["acquisition_config"]
    )
    raw_directory = ROOT / acquisition_config["raw_directory"]
    shared = shared_module()
    shared.validate_raw_inputs(acquisition)

    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or int(config["event_reprocessing"]["parallel_observations"])
    maximum_jobs = int(config["gates"]["maximum_parallel_observations"])
    if jobs < 1 or jobs > maximum_jobs:
        raise ValueError(f"v19F permits between one and {maximum_jobs} jobs")

    requested = common.requested_observations(config)
    acquisition_records = {
        (cluster, obsid): [
            row
            for row in acquisition["records"]
            if row["cluster"] == cluster and int(row["obsid"]) == obsid
        ]
        for cluster, obsid in requested
    }
    completed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                process_observation,
                cluster,
                obsid,
                raw_directory,
                scratch,
                config,
                shared_config,
                acquisition_records[(cluster, obsid)],
            ): (cluster, obsid)
            for cluster, obsid in requested
        }
        for future in as_completed(futures):
            cluster, obsid = futures[future]
            completed.append(future.result())
            print(f"completed {cluster}/{obsid}", flush=True)
    completed.sort(key=lambda row: (row["cluster"], row["obsid"]))

    report = {
        "status": "all_frozen_v19f_chandra_observations_reprocessed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": common.sha256(config_path),
        "acquisition_provenance_sha256": common.sha256(
            ROOT / config["parents"]["acquisition_report"]
        ),
        "runtime_audit_sha256": common.sha256(
            ROOT / config["parents"]["runtime_audit"]
        ),
        "runtime_gate_passed": runtime["gates"]["runtime_gate_passed"],
        "ciaover": shared.command_output(["ciaover", "-v"]),
        "scratch": str(scratch),
        "jobs": jobs,
        "requested_observations": [
            {"cluster": cluster, "obsid": obsid} for cluster, obsid in requested
        ],
        "observations": completed,
        "observation_count": len(completed),
        "product_files": sum(row["product_files"] for row in completed),
        "product_bytes": sum(row["product_bytes"] for row in completed),
        "lensing_target_opened": False,
        "event_images_inspected": False,
        "shock_front_fitted": False,
        "source_constructed": False,
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
                "product_files": report["product_files"],
                "product_bytes": report["product_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
