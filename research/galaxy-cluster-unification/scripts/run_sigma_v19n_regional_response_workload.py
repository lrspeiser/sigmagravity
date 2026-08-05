#!/usr/bin/env python3
"""Enumerate frozen V19N observation/CCD/region response cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycrates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19n_regional_response_workload.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19n_regional_response_workload"


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
            raise RuntimeError(f"V19N parent hash mismatch: {value}")


def read_image(path: Path) -> np.ndarray:
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


def v19m_product(row: dict[str, Any], role: str) -> Path:
    matches = [item for item in row["products"] if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {role} V19M product for {row['cluster']}")
    item = matches[0]
    path = ROOT / item["relative_path"]
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"frozen V19M product changed: {path}")
    return path


def event_assignments(
    path: Path,
    binmap: np.ndarray,
    grid: dict[str, float],
    energy: tuple[int, int],
) -> tuple[dict[tuple[int, int], int], int, list[int]]:
    events = pycrates.read_file(str(path))
    energy_values = np.asarray(events.get_column("ENERGY").values, dtype=float)
    selected = (energy_values >= energy[0]) & (energy_values <= energy[1])
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
    admitted = labels >= 0
    keys = labels[admitted] * 16 + ccd[admitted]
    unique, counts = np.unique(keys, return_counts=True)
    assignments = {
        (int(key // 16), int(key % 16)): int(count)
        for key, count in zip(unique, counts, strict=True)
    }
    return assignments, int(np.count_nonzero(admitted)), sorted(set(ccd.tolist()))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_manifest(
    config: dict[str, Any], source: dict[str, Any], regions: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cluster = source["cluster"]
    binmap = read_image(v19m_product(regions, "binmap")).astype(int)
    counts = read_image(source_product(source, "broad_counts")).astype(float)
    background = read_image(source_product(source, "broad_scaled_background")).astype(float)
    valid_ids = {int(value) for value in regions["valid_region_ids"]}
    all_labels = binmap >= 0
    energy = tuple(int(value) for value in config["assignment"]["energy_eV"])
    scale_by_obs_ccd = {
        (int(row["obsid"]), int(row["ccd_id"])): float(row["scale"])
        for row in source["broad_background_components"]
    }
    observations = sorted(source["observations"], key=lambda row: int(row["obsid"]))
    tasks = []
    science_conservation = 0
    weighted_background_conservation = 0.0
    unexpected_ccds = set()
    observation_summaries = []
    for observation in observations:
        obsid = int(observation["obsid"])
        science, science_total, science_ccds = event_assignments(
            Path(observation["science_reprojected"]), binmap, source["grid"], energy
        )
        blanksky, blanksky_total, background_ccds = event_assignments(
            Path(observation["blanksky_reprojected"]), binmap, source["grid"], energy
        )
        science_conservation += science_total
        supported_ccds = {
            ccd for candidate_obsid, ccd in scale_by_obs_ccd if candidate_obsid == obsid
        }
        unexpected_ccds.update(set(science_ccds) - supported_ccds)
        unexpected_ccds.update(set(background_ccds) - supported_ccds)
        observation_task_count = 0
        for (bin_id, ccd_id), source_events in sorted(science.items()):
            if bin_id not in valid_ids:
                continue
            scale = scale_by_obs_ccd.get((obsid, ccd_id))
            if scale is None:
                raise RuntimeError(
                    f"source events lack frozen background scale: {cluster} {obsid} {ccd_id}"
                )
            background_events = blanksky.get((bin_id, ccd_id), 0)
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
                }
            )
            observation_task_count += 1
        for (bin_id, ccd_id), background_events in blanksky.items():
            scale = scale_by_obs_ccd.get((obsid, ccd_id))
            if scale is None:
                if background_events:
                    unexpected_ccds.add(ccd_id)
                continue
            weighted_background_conservation += background_events * scale
        observation_summaries.append(
            {
                "obsid": obsid,
                "source_events_inside_all_bins": science_total,
                "blanksky_events_inside_all_bins": blanksky_total,
                "response_task_count": observation_task_count,
            }
        )
    expected_science = float(np.sum(counts[all_labels]))
    expected_background = float(np.sum(background[all_labels]))
    admitted_with_tasks = {row["bin_id"] for row in tasks}
    missing_regions = sorted(valid_ids - admitted_with_tasks)
    science_delta = science_conservation - expected_science
    background_delta = weighted_background_conservation - expected_background
    gates = {
        "science_count_conservation_exact": science_delta == 0.0,
        "scaled_background_conservation_within_1e_5": abs(background_delta) <= 1e-5,
        "all_admitted_regions_have_a_response_task": not missing_regions,
        "no_unexpected_ccd_id": not unexpected_ccds,
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
            "science_events_assigned_inside_all_bins": science_conservation,
            "frozen_broad_counts_inside_all_bins": expected_science,
            "science_count_delta": science_delta,
            "weighted_blanksky_events_inside_all_bins": weighted_background_conservation,
            "frozen_scaled_background_inside_all_bins": expected_background,
            "scaled_background_count_delta": background_delta,
            "missing_admitted_region_ids": missing_regions,
            "unexpected_ccd_ids": sorted(unexpected_ccds),
            "observations": observation_summaries,
            "gates": gates,
        },
        tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_parent_hashes(config)
    source_report = load_json(ROOT / config["parents"]["source_map_report"])
    v19m_report = load_json(ROOT / config["parents"]["v19m_report"])
    sources = {row["cluster"]: row for row in source_report["clusters"]}
    regions = {row["cluster"]: row for row in v19m_report["clusters"]}
    cluster_rows = []
    task_rows = []
    for cluster in config["sample"]["clusters"]:
        summary, tasks = cluster_manifest(config, sources[cluster], regions[cluster])
        cluster_rows.append(summary)
        task_rows.extend(tasks)
        print(f"{cluster}: {summary['response_task_count']} response cells", flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "response_task_manifest.csv"
    write_manifest(manifest_path, task_rows)
    passed = all(all(row["gates"].values()) for row in cluster_rows)
    report = {
        "status": (
            "regional_response_workload_conserved_and_enumerated"
            if passed
            else "regional_response_workload_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "response_task_count": len(task_rows),
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
    print(f"tasks: {len(task_rows)}")
    print(f"report: {report_path}")
    print(f"sha256: {sha256(report_path)}")


if __name__ == "__main__":
    main()
