#!/usr/bin/env python3
"""Enumerate V19Q responses on exact positive-exposure science support."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycrates

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19p_exact_flux_obs_support as v19p

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19q_positive_exposure_response_workload.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19q_positive_exposure_response_workload"


def science_assignments(
    virtual_file: str,
    binmap: np.ndarray,
    exposure: np.ndarray,
    grid: dict[str, float],
    energy: tuple[int, int],
    threshold: float,
) -> tuple[dict[tuple[int, int], int], int, list[int], int]:
    events = pycrates.read_file(virtual_file)
    energy_values = np.asarray(events.get_column("ENERGY").values, dtype=float)
    x = np.asarray(events.get_column("X").values, dtype=float)
    y = np.asarray(events.get_column("Y").values, dtype=float)
    ccd = np.asarray(events.get_column("CCD_ID").values, dtype=int)
    selected = (energy_values >= energy[0]) & (energy_values <= energy[1])
    column = np.floor((x - float(grid["xlo"])) / float(grid["binsize"])).astype(int)
    row = np.floor((y - float(grid["ylo"])) / float(grid["binsize"])).astype(int)
    inside = (
        (row >= 0)
        & (row < binmap.shape[0])
        & (column >= 0)
        & (column < binmap.shape[1])
    )
    positive_exposure = np.zeros(len(row), dtype=bool)
    positive_exposure[inside] = exposure[row[inside], column[inside]] > threshold
    labels = np.full(len(row), -1, dtype=int)
    labels[inside] = binmap[row[inside], column[inside]].astype(int)
    spatially_admitted = inside & (labels >= 0)
    included = selected & spatially_admitted & positive_exposure
    rejected_zero_exposure = int(
        np.count_nonzero(selected & spatially_admitted & ~positive_exposure)
    )
    keys = labels[included] * 16 + ccd[included]
    unique, counts = np.unique(keys, return_counts=True)
    table = {
        (int(key // 16), int(key % 16)): int(count)
        for key, count in zip(unique, counts, strict=True)
    }
    supported_ccds = sorted(set(ccd[selected & inside & positive_exposure].tolist()))
    return table, int(np.count_nonzero(included)), supported_ccds, rejected_zero_exposure


def build_cluster(
    config: dict[str, Any],
    source: dict[str, Any],
    regions: dict[str, Any],
    inventory: dict[tuple[str, int, str], dict[str, Any]],
    expected_task_keys: set[tuple[str, int, int, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cluster = source["cluster"]
    binmap = v19p.image(v19p.region_product(regions, "binmap")).astype(int)
    combined_counts = v19p.image(v19p.frozen_product(source, "broad_counts"))
    combined_background = v19p.image(
        v19p.frozen_product(source, "broad_scaled_background")
    )
    valid_ids = {int(value) for value in regions["valid_region_ids"]}
    all_labels = binmap >= 0
    energy = (500, 7000)
    threshold = float(config["science_support_rule"]["exposure_threshold"])
    scale_by_obs_ccd = {
        (int(row["obsid"]), int(row["ccd_id"])): float(row["scale"])
        for row in source["broad_background_components"]
    }
    summed_observation_images = np.zeros_like(combined_counts, dtype=float)
    tasks = []
    science_total = 0
    weighted_background_total = 0.0
    rejected_zero_exposure_total = 0
    unexpected_ccds = set()
    observation_rows = []
    for observation in sorted(source["observations"], key=lambda row: int(row["obsid"])):
        obsid = int(observation["obsid"])
        obs_counts_record = inventory[(cluster, obsid, "flux_obs_broad_counts")]
        exposure_record = inventory[(cluster, obsid, "flux_obs_broad_exposure")]
        fov_record = inventory[(cluster, obsid, "flux_obs_support_fov")]
        science_record = inventory[(cluster, obsid, "science_event")]
        background_record = inventory[(cluster, obsid, "blanksky_event")]
        obs_counts = v19p.image(Path(obs_counts_record["path"]))
        exposure = np.nan_to_num(v19p.image(Path(exposure_record["path"])), nan=0.0)
        if obs_counts.shape != binmap.shape or exposure.shape != binmap.shape:
            raise RuntimeError(f"V19Q observation image shape mismatch: {cluster} {obsid}")
        summed_observation_images += np.nan_to_num(obs_counts, nan=0.0)
        exact_by_region = v19p.image_region_counts(obs_counts, binmap)
        fov_path = Path(fov_record["path"])
        science, assigned, science_ccds, rejected_zero = science_assignments(
            f"{science_record['path']}[sky=region({fov_path})]",
            binmap,
            exposure,
            source["grid"],
            energy,
            threshold,
        )
        background, background_assigned, background_ccds = v19p.event_assignments(
            str(background_record["path"]), binmap, source["grid"], energy
        )
        science_total += assigned
        rejected_zero_exposure_total += rejected_zero
        supported = {
            ccd for candidate_obsid, ccd in scale_by_obs_ccd if candidate_obsid == obsid
        }
        unexpected_ccds.update(set(science_ccds) - supported)
        unexpected_ccds.update(set(background_ccds) - supported)
        event_by_region: dict[int, int] = {}
        for (bin_id, _ccd_id), value in science.items():
            event_by_region[bin_id] = event_by_region.get(bin_id, 0) + value
        compared_ids = set(exact_by_region) | set(event_by_region)
        deltas = {
            bin_id: event_by_region.get(bin_id, 0) - exact_by_region.get(bin_id, 0)
            for bin_id in compared_ids
        }
        mismatched = {key: value for key, value in deltas.items() if value}
        task_count = 0
        for (bin_id, ccd_id), source_events in sorted(science.items()):
            if bin_id not in valid_ids:
                continue
            scale = scale_by_obs_ccd.get((obsid, ccd_id))
            if scale is None:
                raise RuntimeError(
                    f"source events lack frozen background scale: {cluster} {obsid} CCD {ccd_id}"
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
                    "flux_obs_support_fov_sha256": fov_record["sha256"],
                    "flux_obs_exposure_sha256": exposure_record["sha256"],
                }
            )
            task_count += 1
        for (bin_id, ccd_id), background_events in background.items():
            scale = scale_by_obs_ccd.get((obsid, ccd_id))
            if scale is None:
                if background_events:
                    unexpected_ccds.add(ccd_id)
                continue
            weighted_background_total += background_events * scale
        exact_total = sum(exact_by_region.values())
        observation_rows.append(
            {
                "obsid": obsid,
                "flux_obs_image_events_inside_all_bins": exact_total,
                "positive_exposure_events_inside_all_bins": assigned,
                "science_count_delta": assigned - exact_total,
                "zero_exposure_events_rejected_inside_bins": rejected_zero,
                "region_count_with_nonzero_delta": len(mismatched),
                "maximum_absolute_region_delta": max(
                    (abs(value) for value in mismatched.values()), default=0
                ),
                "blanksky_events_inside_all_bins": background_assigned,
                "response_task_count": task_count,
                "flux_obs_counts_sha256": obs_counts_record["sha256"],
                "flux_obs_exposure_sha256": exposure_record["sha256"],
                "flux_obs_support_fov_sha256": fov_record["sha256"],
            }
        )
    combined_finite = np.nan_to_num(combined_counts, nan=0.0)
    image_delta = summed_observation_images - combined_finite
    changed_pixels = int(np.count_nonzero(image_delta))
    exact_science = int(np.rint(np.sum(combined_finite[all_labels])))
    science_delta = science_total - exact_science
    expected_background = float(np.sum(combined_background[all_labels]))
    background_delta = weighted_background_total - expected_background
    task_keys = {
        (row["cluster"], row["bin_id"], row["obsid"], row["ccd_id"]) for row in tasks
    }
    cluster_expected_keys = {key for key in expected_task_keys if key[0] == cluster}
    missing_keys = sorted(cluster_expected_keys - task_keys)
    added_keys = sorted(task_keys - cluster_expected_keys)
    missing_regions = sorted(valid_ids - {row["bin_id"] for row in tasks})
    expected_count = int(
        config["sample"]["expected_response_task_count_by_cluster"][cluster]
    )
    gates = {
        "per_observation_images_sum_pixelwise_to_combined": changed_pixels == 0,
        "every_observation_region_conserves_exactly": all(
            row["region_count_with_nonzero_delta"] == 0 for row in observation_rows
        ),
        "total_science_count_conservation_exact": science_delta == 0,
        "scaled_background_conservation_within_1e_5": abs(background_delta) <= 1e-5,
        "task_key_set_equals_v19n": not missing_keys and not added_keys,
        "task_count_equals_frozen_expectation": len(tasks) == expected_count,
        "all_admitted_regions_have_a_response_task": not missing_regions,
        "no_unexpected_ccd_id": not unexpected_ccds,
    }
    upper_mb = len(tasks) * float(
        config["engineering_report"]["provisional_upper_storage_megabytes_per_cell"]
    )
    return (
        {
            "cluster": cluster,
            "observation_count": len(observation_rows),
            "admitted_region_count": len(valid_ids),
            "response_task_count": len(tasks),
            "expected_output_file_count": sum(row["expected_output_files"] for row in tasks),
            "provisional_upper_storage_gib": upper_mb / 1024.0,
            "positive_exposure_science_events_inside_all_bins": science_total,
            "frozen_combined_counts_inside_all_bins": exact_science,
            "science_count_delta": science_delta,
            "zero_exposure_events_rejected_inside_bins": rejected_zero_exposure_total,
            "per_observation_image_sum_changed_pixel_count": changed_pixels,
            "per_observation_image_sum_maximum_absolute_pixel_delta": float(
                np.max(np.abs(image_delta))
            ),
            "weighted_blanksky_events_inside_all_bins": weighted_background_total,
            "frozen_scaled_background_inside_all_bins": expected_background,
            "scaled_background_count_delta": background_delta,
            "missing_v19n_task_keys": [list(key) for key in missing_keys],
            "added_task_keys": [list(key) for key in added_keys],
            "missing_admitted_region_ids": missing_regions,
            "unexpected_ccd_ids": sorted(unexpected_ccds),
            "observations": observation_rows,
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
    config = v19p.load_json(config_path)
    v19p.validate_parent_hashes(config)
    source_report = v19p.load_json(ROOT / config["parents"]["source_map_report"])
    region_report = v19p.load_json(ROOT / config["parents"]["v19m_report"])
    input_inventory = v19p.load_json(ROOT / config["parents"]["input_inventory"])
    inventory = v19p.inventory_rows(input_inventory)
    expected_keys = v19p.parent_task_keys(ROOT / config["parents"]["v19n_manifest"])
    sources = {row["cluster"]: row for row in source_report["clusters"]}
    regions = {row["cluster"]: row for row in region_report["clusters"]}
    cluster_rows = []
    tasks = []
    for cluster in config["sample"]["clusters"]:
        summary, cluster_tasks = build_cluster(
            config, sources[cluster], regions[cluster], inventory, expected_keys
        )
        cluster_rows.append(summary)
        tasks.extend(cluster_tasks)
        print(
            f"{cluster}: {summary['response_task_count']} cells; "
            f"science delta {summary['science_count_delta']}; "
            f"zero-exposure rejected {summary['zero_exposure_events_rejected_inside_bins']}",
            flush=True,
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "response_task_manifest.csv"
    v19p.write_manifest(manifest_path, tasks)
    cluster_pass = all(all(row["gates"].values()) for row in cluster_rows)
    total_gate = len(tasks) == int(config["sample"]["expected_response_task_count_total"])
    passed = cluster_pass and total_gate
    report = {
        "status": (
            "positive_exposure_response_workload_conserved_and_authorized"
            if passed
            else "positive_exposure_response_workload_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": v19p.sha256(config_path),
        "runner_sha256": v19p.sha256(Path(__file__).resolve()),
        "manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": v19p.sha256(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "response_task_count": len(tasks),
        },
        "clusters": cluster_rows,
        "global_gates": {
            "all_cluster_gates_pass": cluster_pass,
            "total_task_count_equals_5082": total_gate,
        },
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
    print(f"sha256: {v19p.sha256(report_path)}")


if __name__ == "__main__":
    main()
