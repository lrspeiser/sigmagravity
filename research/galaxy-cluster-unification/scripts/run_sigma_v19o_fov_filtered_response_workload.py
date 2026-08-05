#!/usr/bin/env python3
"""Enumerate V19O response cells after exact registered-FOV filtering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycrates

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v17c_integrated_spectra as inherited

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19o_fov_filtered_response_workload.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19o_fov_filtered_response_workload"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19o-response-workload/fov_v100")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_parent_hashes(config: dict[str, Any]) -> None:
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None and sha256(ROOT / value) != expected:
            raise RuntimeError(f"V19O parent hash mismatch: {value}")


def image(path: Path) -> np.ndarray:
    return np.asarray(pycrates.read_file(str(path)).get_image().values)


def source_product(row: dict[str, Any], role: str) -> Path:
    matches = [item for item in row["frozen_snapshot"]["products"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {role} source product for {row['cluster']}")
    item = matches[0]
    path = ROOT / item["relative_path"]
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"frozen source product changed: {path}")
    return path


def region_product(row: dict[str, Any], role: str) -> Path:
    matches = [item for item in row["products"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {role} region product for {row['cluster']}")
    item = matches[0]
    path = ROOT / item["relative_path"]
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"frozen region product changed: {path}")
    return path


def repro_fov(row: dict[str, Any]) -> tuple[Path, str]:
    matches = [
        item for item in row["products"] if item["relative_path"].endswith("_repro_fov1.fits")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one repro FOV for ObsID {row['obsid']}")
    item = matches[0]
    path = Path(row["output_directory"]) / item["relative_path"]
    return path, item["sha256"]


def assignments(
    virtual_file: str,
    binmap: np.ndarray,
    grid: dict[str, float],
) -> tuple[dict[tuple[int, int], int], int, list[int]]:
    events = pycrates.read_file(virtual_file)
    energy = np.asarray(events.get_column("ENERGY").values, dtype=float)
    selected = (energy >= 500.0) & (energy <= 7000.0)
    x = np.asarray(events.get_column("X").values[selected], dtype=float)
    y = np.asarray(events.get_column("Y").values[selected], dtype=float)
    ccd = np.asarray(events.get_column("CCD_ID").values[selected], dtype=int)
    column = np.floor((x - float(grid["xlo"])) / float(grid["binsize"])).astype(int)
    row = np.floor((y - float(grid["ylo"])) / float(grid["binsize"])).astype(int)
    inside = (
        (row >= 0)
        & (row < binmap.shape[0])
        & (column >= 0)
        & (column < binmap.shape[1])
    )
    labels = np.full(len(row), -1, dtype=int)
    labels[inside] = binmap[row[inside], column[inside]].astype(int)
    included = labels >= 0
    keys = labels[included] * 16 + ccd[included]
    unique, counts = np.unique(keys, return_counts=True)
    table = {
        (int(key // 16), int(key % 16)): int(count)
        for key, count in zip(unique, counts, strict=True)
    }
    return table, int(np.count_nonzero(included)), sorted(set(ccd.tolist()))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_cluster(
    config: dict[str, Any],
    source: dict[str, Any],
    region: dict[str, Any],
    repro_rows: dict[int, dict[str, Any]],
    astrometry_rows: dict[int, dict[str, Any]],
    parent_failure: dict[str, Any],
    scratch: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cluster = source["cluster"]
    binmap = image(region_product(region, "binmap")).astype(int)
    counts = image(source_product(source, "broad_counts")).astype(float)
    all_labels = binmap >= 0
    valid_ids = {int(value) for value in region["valid_region_ids"]}
    scale_by_obs_ccd = {
        (int(row["obsid"]), int(row["ccd_id"])): float(row["scale"])
        for row in source["broad_background_components"]
    }
    env = inherited.isolated_environment(
        os.environ,
        scratch / "pfiles" / cluster,
        scratch / "tmp" / cluster,
    )
    tasks = []
    science_total = 0
    unexpected_ccds = set()
    fov_records = []
    observations = []
    for observation in sorted(source["observations"], key=lambda row: int(row["obsid"])):
        obsid = int(observation["obsid"])
        source_fov, expected_fov_hash = repro_fov(repro_rows[obsid])
        translated = scratch / "translated_fov" / cluster / f"acisf{obsid}_gaia_fov1.fits"
        fov_record = inherited.prepare_translated_fov(
            source_fov,
            translated,
            astrometry_rows[obsid],
            expected_fov_hash,
            env,
        )
        fov_records.append(fov_record)
        science, assigned, science_ccds = assignments(
            f"{observation['science_reprojected']}[sky=region({translated})]",
            binmap,
            source["grid"],
        )
        background, _, background_ccds = assignments(
            f"{observation['blanksky_reprojected']}[sky=region({translated})]",
            binmap,
            source["grid"],
        )
        science_total += assigned
        supported = {
            ccd for candidate_obsid, ccd in scale_by_obs_ccd if candidate_obsid == obsid
        }
        unexpected_ccds.update(set(science_ccds) - supported)
        unexpected_ccds.update(set(background_ccds) - supported)
        count = 0
        for (bin_id, ccd_id), source_events in sorted(science.items()):
            if bin_id not in valid_ids:
                continue
            scale = scale_by_obs_ccd.get((obsid, ccd_id))
            if scale is None:
                raise RuntimeError(
                    f"source events lack frozen scale: {cluster} {obsid} CCD {ccd_id}"
                )
            background_events = background.get((bin_id, ccd_id), 0)
            tasks.append(
                {
                    "cluster": cluster,
                    "bin_id": bin_id,
                    "obsid": obsid,
                    "ccd_id": ccd_id,
                    "source_band_events": source_events,
                    "background_band_events": background_events,
                    "blanksky_scale": scale,
                    "scaled_background_events": background_events * scale,
                    "expected_output_files": int(
                        config["engineering_report"]["files_per_response_cell"]
                    ),
                    "translated_fov_sha256": fov_record["translated_sha256"],
                }
            )
            count += 1
        observations.append(
            {
                "obsid": obsid,
                "science_events_inside_all_bins": assigned,
                "response_task_count": count,
                "translated_fov_sha256": fov_record["translated_sha256"],
            }
        )
    expected_science = float(np.sum(counts[all_labels]))
    science_delta = science_total - expected_science
    admitted_with_tasks = {row["bin_id"] for row in tasks}
    missing = sorted(valid_ids - admitted_with_tasks)
    parent_cluster = next(
        row for row in parent_failure["clusters"] if row["cluster"] == cluster
    )
    fov_hashes = [row["translated_sha256"] for row in fov_records]
    gates = {
        "science_count_conservation_exact": science_delta == 0.0,
        "parent_scaled_background_conservation_retained": parent_cluster["gates"][
            "scaled_background_conservation_within_1e_5"
        ],
        "all_admitted_regions_have_a_response_task": not missing,
        "no_unexpected_ccd_id": not unexpected_ccds,
        "all_translated_fov_hashes_finite_and_unique": len(fov_hashes)
        == len(set(fov_hashes))
        == len(observations),
    }
    upper_mb = len(tasks) * float(
        config["engineering_report"]["provisional_upper_storage_megabytes_per_cell"]
    )
    return (
        {
            "cluster": cluster,
            "observation_count": len(observations),
            "admitted_region_count": len(valid_ids),
            "response_task_count": len(tasks),
            "expected_output_file_count": sum(row["expected_output_files"] for row in tasks),
            "provisional_upper_storage_gib": upper_mb / 1024.0,
            "science_events_assigned_inside_all_bins": science_total,
            "frozen_broad_counts_inside_all_bins": expected_science,
            "science_count_delta": science_delta,
            "v19n_unfiltered_upper_bound_response_tasks": parent_cluster[
                "response_task_count"
            ],
            "tasks_removed_by_fov_filter": parent_cluster["response_task_count"]
            - len(tasks),
            "missing_admitted_region_ids": missing,
            "unexpected_ccd_ids": sorted(unexpected_ccds),
            "observations": observations,
            "translated_fovs": fov_records,
            "gates": gates,
        },
        tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_parent_hashes(config)
    source_report = load_json(ROOT / config["parents"]["source_map_report"])
    region_report = load_json(ROOT / config["parents"]["v19m_report"])
    repro_report = load_json(ROOT / config["parents"]["repro_report"])
    astrometry_report = load_json(ROOT / config["parents"]["astrometry_report"])
    parent_failure = load_json(ROOT / config["parents"]["v19n_failure_report"])
    sources = {row["cluster"]: row for row in source_report["clusters"]}
    regions = {row["cluster"]: row for row in region_report["clusters"]}
    cluster_rows = []
    tasks = []
    for cluster in config["sample"]["clusters"]:
        repro_rows = {
            int(row["obsid"]): row
            for row in repro_report["observations"]
            if row["cluster"] == cluster
        }
        astrometry_rows = {
            int(row["obsid"]): row
            for row in astrometry_report["observations"]
            if row["cluster"] == cluster
        }
        summary, cluster_tasks = build_cluster(
            config,
            sources[cluster],
            regions[cluster],
            repro_rows,
            astrometry_rows,
            parent_failure,
            args.scratch.resolve(),
        )
        cluster_rows.append(summary)
        tasks.extend(cluster_tasks)
        print(f"{cluster}: {summary['response_task_count']} response cells", flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "response_task_manifest.csv"
    write_manifest(manifest_path, tasks)
    passed = all(all(row["gates"].values()) for row in cluster_rows)
    report = {
        "status": (
            "fov_filtered_regional_response_workload_conserved_and_enumerated"
            if passed
            else "fov_filtered_regional_response_workload_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "response_task_count": len(tasks),
        },
        "clusters": cluster_rows,
        "response_extraction_authorized": passed,
        "spectrum_or_response_constructed": False,
        "temperature_density_mach_or_speed_fitted": False,
        "lensing_target_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status: {report['status']}")
    print(f"tasks: {len(tasks)}")
    print(f"report: {report_path}")
    print(f"sha256: {sha256(report_path)}")


if __name__ == "__main__":
    main()
