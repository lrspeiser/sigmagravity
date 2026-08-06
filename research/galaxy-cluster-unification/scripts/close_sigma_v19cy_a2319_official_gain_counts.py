#!/usr/bin/env python3
"""Close the official 000100000 gain-solution accounting without applying gain."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_official_gain_count_closure.json"
)
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(config_path)
    if config.get("protocol_version") != (
        "SIGMA-V19CY-A2319-OFFICIAL-GAIN-COUNT-CLOSURE-1.0.0"
    ):
        raise RuntimeError("unexpected official gain-count closure protocol")
    if config.get("status") != (
        "frozen after the terminal pooled timeline audit and visual inspection of the official report, but before any corrected row-level selection was executed"
    ):
        raise RuntimeError("official gain-count closure protocol is not frozen")

    for name in ("timeline_report", "metadata_report", "download_provenance"):
        path = ROOT / config["parents"][name]
        expected = config["parents"][f"{name}_sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen official-count parent changed: {path}")

    timeline = load_json(ROOT / config["parents"]["timeline_report"])
    if timeline.get("gain_applied_or_interpolated") or timeline.get(
        "validation_or_holdout_accessed"
    ):
        raise RuntimeError("timeline parent violated its seal")

    authorization = config["authorization"]
    for key in (
        "read_continuous_calibration_pixel_history",
        "read_gain_history_array_columns",
        "read_event_rows_or_energies",
        "apply_or_interpolate_gain",
        "fit_spectrum_or_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed official-count boundary is open: {key}")

    provenance = load_json(ROOT / config["parents"]["download_provenance"])
    return config, provenance


def verified_source(
    relative: str,
    raw_root: Path,
    provenance_by_path: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root):
        raise RuntimeError(f"official-count source escapes raw root: {relative}")
    terminal = provenance_by_path.get(relative)
    if terminal is None:
        raise RuntimeError(f"official-count source absent from provenance: {relative}")
    if not path.is_file() or path.stat().st_size != terminal["bytes"]:
        raise RuntimeError(f"official-count source size changed: {relative}")
    if sha256(path) != terminal["sha256"]:
        raise RuntimeError(f"official-count source hash changed: {relative}")
    return path, terminal


def read_time_and_pixel(path: Path, extension: str) -> tuple[float, np.ndarray, np.ndarray]:
    with gzip.open(path, "rb") as stream, fits.open(
        stream, memmap=False, mode="readonly"
    ) as hdus:
        if extension not in hdus:
            raise RuntimeError(f"missing extension {extension}: {path.name}")
        hdu = hdus[extension]
        names = set(hdu.columns.names or [])
        if not {"TIME", "PIXEL"}.issubset(names):
            raise RuntimeError(f"TIME/PIXEL missing from {path.name}[{extension}]")
        tstart = float(hdu.header["TSTART"])
        times = np.asarray(hdu.data["TIME"], dtype=float).copy()
        pixels = np.asarray(hdu.data["PIXEL"], dtype=int).copy()
    return tstart, times, pixels


def count_pixels(pixels: np.ndarray) -> dict[str, int]:
    return {str(pixel): count for pixel, count in sorted(Counter(pixels.tolist()).items())}


def interval_overlaps(
    times: np.ndarray,
    tstart: float,
    intervals: dict[str, list[list[float]]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    relative = times - tstart
    for kind, pairs in intervals.items():
        row_indexes: set[int] = set()
        interval_counts: list[int] = []
        for start, stop in pairs:
            selected = np.flatnonzero((relative >= start) & (relative <= stop))
            interval_counts.append(int(selected.size))
            row_indexes.update(int(index) for index in selected)
        output[kind] = {
            "intervals": len(pairs),
            "per_interval_row_counts": interval_counts,
            "unique_rows_inside": len(row_indexes),
        }
    return output


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, provenance = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    provenance_by_path = {
        record["download_path"]: record for record in provenance["records"]
    }

    report_path, report_terminal = verified_source(
        config["paths"]["official_report"], raw_root, provenance_by_path
    )
    if sha256(report_path) != config["official_report"]["sha256"]:
        raise RuntimeError("official gain report differs from visually inspected source")

    history_path, history_terminal = verified_source(
        config["paths"]["intermittent_gain_history"], raw_root, provenance_by_path
    )
    tstart, times, pixels = read_time_and_pixel(
        history_path, config["source"]["extension"]
    )
    actual_per_pixel = count_pixels(pixels)
    expected_per_pixel = config["expected_counts"]["per_pixel"]
    row_count = int(times.size)
    pixel_12 = actual_per_pixel.get("12", 0)
    non_pixel_12 = row_count - pixel_12
    overlaps = interval_overlaps(
        times,
        tstart,
        config["relative_exclusion_intervals_seconds"],
    )

    checks = {
        "tstart_exact": tstart == config["source"]["expected_tstart"],
        "raw_rows_exact": row_count == config["source"]["expected_raw_rows"],
        "per_pixel_counts_exact": actual_per_pixel == expected_per_pixel,
        "pixel_12_total_exact": pixel_12
        == config["expected_counts"]["intermittent_calibration_pixel_12"],
        "non_pixel_12_total_exact": non_pixel_12
        == config["expected_counts"]["fe55_non_pixel_12"],
        "zero_saa_overlap": overlaps["saa"]["unique_rows_inside"] == 0,
        "zero_adr_overlap": overlaps["adr"]["unique_rows_inside"] == 0,
    }
    closure = all(checks.values())
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_official_gain_count_closure_audited",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "source": {
            "obsid": config["source"]["obsid"],
            "gain_history": config["paths"]["intermittent_gain_history"],
            "gain_history_bytes": history_terminal["bytes"],
            "gain_history_sha256": history_terminal["sha256"],
            "official_report": config["paths"]["official_report"],
            "official_report_bytes": report_terminal["bytes"],
            "official_report_sha256": report_terminal["sha256"],
        },
        "tstart": tstart,
        "raw_rows": row_count,
        "actual_per_pixel": actual_per_pixel,
        "expected_per_pixel": expected_per_pixel,
        "actual_counts": {
            "fe55_non_pixel_12": non_pixel_12,
            "intermittent_calibration_pixel_12": pixel_12,
        },
        "interval_overlap_audit": overlaps,
        "checks": checks,
        "official_count_closure_reproduced": closure,
        "continuous_calibration_pixel_history_accessed": False,
        "gain_history_array_column_read": False,
        "event_row_or_energy_read": False,
        "gain_applied_or_interpolated": False,
        "spectrum_or_velocity_fit_performed": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_separate_gain_reconstruction_protocol_freeze"
            if closure
            else "stop_before_gain_reconstruction_protocol"
        ),
        "authorization": {
            "freeze_gain_reconstruction_protocol": closure,
            "read_continuous_calibration_pixel_history": False,
            "read_event_rows_or_energies": False,
            "apply_or_interpolate_gain": False,
            "fit_spectrum_or_velocity": False,
            "access_validation_or_holdout_assets": False,
            "open_lensing_halo_or_gravity_targets": False,
            "derive_or_select_action": False,
        },
    }
    output = ROOT / config["paths"]["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "raw_rows",
                    "actual_counts",
                    "interval_overlap_audit",
                    "checks",
                    "official_count_closure_reproduced",
                    "decision",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
