#!/usr/bin/env python3
"""Reprocess every frozen Sigma v17A Chandra observation with CIAO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
DEFAULT_ACQUISITION = ROOT / "results" / "sigma_v17a_chandra_acquisition" / "provenance.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a_chandra_repro"
FROZEN_STATUS = (
    "frozen before reprocessing, event-image inspection, temperature-region "
    "construction, spectral fitting, or reading a v17 dynamical-feature score"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def dmkeypar(path: Path, key: str, env: dict[str, str]) -> str | None:
    result = subprocess.run(
        ["dmkeypar", str(path), key, "echo+"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def isolated_environment(base: os._Environ[str], pfiles: Path, tmp: Path) -> dict[str, str]:
    env = dict(base)
    pfiles.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    existing = env.get("PFILES", "")
    system = existing.split(";", maxsplit=1)[1] if ";" in existing else existing
    env["PFILES"] = f"{pfiles};{system}" if system else str(pfiles)
    env["ASCDS_WORK_PATH"] = str(tmp)
    env["TMPDIR"] = str(tmp)
    return env


def validate_raw_inputs(acquisition: dict) -> None:
    for row in acquisition["records"]:
        path = ROOT / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"raw Chandra size changed: {path}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"raw Chandra hash changed: {path}")


def output_inventory(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def stage_observation_inputs(
    cluster: str,
    obsid: int,
    source_directory: Path,
    input_directory: Path,
    records: list[dict],
) -> dict:
    input_directory.mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    staged_bytes = 0
    for row in records:
        source = ROOT / row["relative_path"]
        relative = source.relative_to(source_directory)
        if relative.parts[0] == "root":
            relative = Path(*relative.parts[1:])
        destination = input_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"staged input size changed: {destination}")
            if sha256(destination) != row["sha256"]:
                raise RuntimeError(f"staged input hash changed: {destination}")
            reused += 1
        else:
            shutil.copy2(source, destination)
            if sha256(destination) != row["sha256"]:
                raise RuntimeError(f"staged input copy failed hash check: {destination}")
            copied += 1
        staged_bytes += int(row["bytes"])
    return {
        "cluster": cluster,
        "obsid": obsid,
        "source_directory": str(source_directory),
        "input_directory": str(input_directory),
        "files": len(records),
        "bytes": staged_bytes,
        "copied": copied,
        "reused": reused,
        "root_products_promoted_to_standard_obsid_layout": True,
    }


def inspect_event(event: Path, env: dict[str, str], expected_caldb: str) -> dict:
    counts = command_output(["dmlist", str(event), "counts"], env=env)
    repro_history = command_output(["dmhistory", str(event), "chandra_repro"], env=env)
    keys = (
        "OBS_ID",
        "EXPOSURE",
        "LIVETIME",
        "DATAMODE",
        "READMODE",
        "TIMEDEL",
        "DATE-OBS",
        "DATE-END",
        "CHECK_VF_PHA",
        "RAND_PI",
        "RAND_SKY",
        "PIX_ADJ",
        "CALDBVER",
    )
    return {
        "path": str(event),
        "event_rows": int(counts.split()[0]),
        "header": {key: dmkeypar(event, key, env) for key in keys},
        "repro_history_sha256": hashlib.sha256(repro_history.encode()).hexdigest(),
        "current_caldb_comment_present": f"CALDB {expected_caldb}" in repro_history,
        "caldbver_header_note": (
            "CALDBVER is inherited from the archive event; the chandra_repro "
            "HISTORY/COMMENT records the current reduction CALDB."
        ),
    }


def process_observation(
    cluster: str,
    obsid: int,
    raw_directory: Path,
    scratch: Path,
    config: dict,
    acquisition_records: list[dict],
) -> dict:
    source_directory = raw_directory / cluster / str(obsid)
    input_directory = scratch / "input" / cluster / str(obsid)
    output_directory = scratch / "repro" / cluster / str(obsid)
    pfiles = scratch / "pfiles" / cluster / str(obsid)
    tmp = scratch / "tmp" / cluster / str(obsid)
    logs = scratch / "logs" / cluster
    logs.mkdir(parents=True, exist_ok=True)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{obsid}_chandra_repro.log"
    env = isolated_environment(os.environ, pfiles, tmp)
    staging = stage_observation_inputs(
        cluster,
        obsid,
        source_directory,
        input_directory,
        acquisition_records,
    )

    existing_events = sorted(output_directory.glob("*repro_evt2.fits*"))
    reused = bool(existing_events)
    if output_directory.exists() and not reused:
        raise RuntimeError(f"partial output exists without repro event file: {output_directory}")

    repro = config["event_reprocessing"]
    command = [
        "chandra_repro",
        f"indir={input_directory}",
        f"outdir={output_directory}",
        f"check_vf_pha={'yes' if repro['check_vf_pha'] else 'no'}",
        f"pix_adj={repro['pix_adj']}",
        "set_ardlib=no",
        "clobber=no",
        "verbose=1",
        "mode=h",
    ]
    if reused:
        log = "reused complete output from an earlier invocation\n"
        if not log_path.exists():
            log_path.write_text(log, encoding="utf-8")
    else:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        log = result.stdout + result.stderr
        log_path.write_text(log, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"chandra_repro failed for {cluster}/{obsid}; see {log_path}")

    events = sorted(output_directory.glob("*repro_evt2.fits*"))
    if len(events) != 1:
        raise RuntimeError(f"expected one repro event for {cluster}/{obsid}, found {len(events)}")
    event = events[0]
    inspection = inspect_event(event, env, config["runtime"]["caldb_main"])
    if inspection["header"]["OBS_ID"] != str(obsid):
        raise RuntimeError(f"reprocessed event OBS_ID mismatch: {event}")
    if "VFAINT" not in (inspection["header"]["DATAMODE"] or ""):
        raise RuntimeError(f"reprocessed event is not VFAINT mode: {event}")
    if not inspection["current_caldb_comment_present"]:
        raise RuntimeError(f"current CALDB provenance is absent from history: {event}")

    products = output_inventory(output_directory)
    return {
        "cluster": cluster,
        "obsid": obsid,
        "input_directory": str(input_directory),
        "staging": staging,
        "output_directory": str(output_directory),
        "command": command,
        "reused": reused,
        "log": str(log_path),
        "log_sha256": sha256(log_path) if log_path.is_file() else None,
        "event": inspection,
        "products": products,
        "product_files": len(products),
        "product_bytes": sum(row["bytes"] for row in products),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    acquisition_path = args.acquisition.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if config["status"] != FROZEN_STATUS:
        raise RuntimeError("the v17A Chandra reduction protocol is not frozen")
    acquisition_config = ROOT / "configs" / "sigma_v17a_chandra_acquisition.json"
    if acquisition["config_sha256"] != sha256(acquisition_config):
        raise RuntimeError("acquisition protocol does not match its provenance")
    if acquisition["lensing_target_opened"]:
        raise RuntimeError("acquisition provenance unexpectedly opened a lensing target")

    validate_raw_inputs(acquisition)
    raw_directory = (
        ROOT / json.loads(acquisition_config.read_text(encoding="utf-8"))["raw_directory"]
    )
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or int(config["event_reprocessing"]["parallel_observations"])
    if jobs < 1 or jobs > 2:
        raise ValueError("the frozen v17A protocol permits one or two parallel observations")

    requested = [
        (cluster, int(obsid))
        for cluster, values in config["clusters"].items()
        for obsid in values["obsids"]
    ]
    acquisition_records = {
        (cluster, obsid): [
            row
            for row in acquisition["records"]
            if row["cluster"] == cluster and row["obsid"] == obsid
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
        "status": "all_frozen_chandra_observations_reprocessed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_sha256": sha256(config_path),
        "acquisition_provenance_sha256": sha256(acquisition_path),
        "ciaover": command_output(["ciaover", "-v"]),
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
        "temperature_map_constructed": False,
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
