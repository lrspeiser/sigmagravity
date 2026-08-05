#!/usr/bin/env python3
"""Locate event/image pixel deltas after the frozen V19P gate."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pycrates

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19p_exact_flux_obs_support.json"
REPORT = ROOT / "results" / "sigma_v19p_exact_flux_obs_support" / "report.json"
OUTPUT = ROOT / "results" / "sigma_v19p_exact_flux_obs_support" / "pixel_diagnostic.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image(path: Path) -> np.ndarray:
    return np.asarray(pycrates.read_file(str(path)).get_image().values, dtype=float)


def product(inventory: dict[str, Any], cluster: str, obsid: int, role: str) -> Path:
    cluster_row = next(row for row in inventory["clusters"] if row["cluster"] == cluster)
    observation = next(
        row for row in cluster_row["observations"] if int(row["obsid"]) == obsid
    )
    item = next(row for row in observation["products"] if row["role"] == role)
    return Path(item["path"])


def column(events: Any, name: str, selected: np.ndarray) -> list[Any] | None:
    try:
        values = np.asarray(events.get_column(name).values)[selected]
    except (AttributeError, ValueError):
        return None
    result = []
    for value in values:
        if isinstance(value, np.ndarray):
            item = value.tolist()
        elif hasattr(value, "item"):
            item = value.item()
        else:
            item = value
        if isinstance(item, bytes):
            item = item.hex()
        result.append(item)
    return result


def main() -> None:
    config = load(CONFIG)
    report = load(REPORT)
    source = load(ROOT / config["parents"]["source_map_report"])
    regions = load(ROOT / config["parents"]["v19m_report"])
    inventory = load(ROOT / config["parents"]["input_inventory"])
    sources = {row["cluster"]: row for row in source["clusters"]}
    region_rows = {row["cluster"]: row for row in regions["clusters"]}
    deltas = []
    for failed_cluster in report["clusters"]:
        cluster = failed_cluster["cluster"]
        source_row = sources[cluster]
        region_row = region_rows[cluster]
        binmap_item = next(row for row in region_row["products"] if row["role"] == "binmap")
        binmap = image(ROOT / binmap_item["relative_path"]).astype(int)
        grid = source_row["grid"]
        for observation in failed_cluster["observations"]:
            if not observation["region_count_with_nonzero_delta"]:
                continue
            obsid = int(observation["obsid"])
            count_path = product(inventory, cluster, obsid, "flux_obs_broad_counts")
            exposure_path = product(inventory, cluster, obsid, "flux_obs_broad_exposure")
            event_path = product(inventory, cluster, obsid, "science_event")
            fov_path = product(inventory, cluster, obsid, "flux_obs_support_fov")
            counts = np.nan_to_num(image(count_path), nan=0.0)
            exposure = np.nan_to_num(image(exposure_path), nan=0.0)
            events = pycrates.read_file(f"{event_path}[sky=region({fov_path})]")
            energy = np.asarray(events.get_column("ENERGY").values, dtype=float)
            selected = (energy >= 500.0) & (energy <= 7000.0)
            x = np.asarray(events.get_column("X").values[selected], dtype=float)
            y = np.asarray(events.get_column("Y").values[selected], dtype=float)
            cols = np.floor((x - float(grid["xlo"])) / float(grid["binsize"])).astype(int)
            rows = np.floor((y - float(grid["ylo"])) / float(grid["binsize"])).astype(int)
            inside = (
                (rows >= 0)
                & (rows < counts.shape[0])
                & (cols >= 0)
                & (cols < counts.shape[1])
            )
            histogram = np.zeros_like(counts, dtype=np.int64)
            np.add.at(histogram, (rows[inside], cols[inside]), 1)
            difference = histogram - np.rint(counts).astype(np.int64)
            bad_rows, bad_cols = np.where((difference != 0) & (binmap >= 0))
            metadata = {
                name: column(events, name, selected)
                for name in (
                    "TIME",
                    "ENERGY",
                    "PI",
                    "CCD_ID",
                    "CHIPX",
                    "CHIPY",
                    "DETX",
                    "DETY",
                    "GRADE",
                    "STATUS",
                )
            }
            for row, col in zip(bad_rows, bad_cols, strict=True):
                matching = np.flatnonzero((rows == row) & (cols == col))
                event_records = []
                for index in matching:
                    record = {"x": float(x[index]), "y": float(y[index])}
                    for name, values in metadata.items():
                        if values is not None:
                            record[name.lower()] = values[index]
                    event_records.append(record)
                deltas.append(
                    {
                        "cluster": cluster,
                        "obsid": obsid,
                        "image_row_zero_based": int(row),
                        "image_column_zero_based": int(col),
                        "bin_id": int(binmap[row, col]),
                        "event_count": int(histogram[row, col]),
                        "flux_obs_image_count": round(counts[row, col]),
                        "event_minus_image": int(difference[row, col]),
                        "flux_obs_exposure": float(exposure[row, col]),
                        "events_in_pixel": event_records,
                    }
                )
    result = {
        "status": "post_v19p_failure_pixel_diagnostic",
        "generated_utc": datetime.now(UTC).isoformat(),
        "v19p_config_sha256": sha256(CONFIG),
        "v19p_report_sha256": sha256(REPORT),
        "delta_pixel_count_inside_regions": len(deltas),
        "deltas": deltas,
        "gate_or_threshold_changed": False,
        "spectrum_or_response_constructed": False,
        "gravity_formula_or_parameter_changed": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"sha256: {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
