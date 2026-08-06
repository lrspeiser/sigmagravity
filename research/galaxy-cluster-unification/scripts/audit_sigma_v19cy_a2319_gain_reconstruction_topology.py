#!/usr/bin/env python3
"""Audit A2319 gain-reconstruction topology without reading event data."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "sigma_v19cy_a2319_gain_reconstruction_topology.json"
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
        "SIGMA-V19CY-A2319-GAIN-RECONSTRUCTION-TOPOLOGY-1.0.0"
    ):
        raise RuntimeError("unexpected gain-reconstruction topology protocol")
    if config.get("status") != (
        "frozen after official gain-count closure and paper-method review, but before branch-specific scalar reconstruction or any event-row access"
    ):
        raise RuntimeError("gain-reconstruction topology protocol is not frozen")
    for name in ("official_count_report", "timeline_report", "download_provenance"):
        path = ROOT / config["parents"][name]
        if not path.is_file() or sha256(path) != config["parents"][f"{name}_sha256"]:
            raise RuntimeError(f"frozen gain-reconstruction parent changed: {path}")
    official = load_json(ROOT / config["parents"]["official_count_report"])
    if not official.get("official_count_closure_reproduced"):
        raise RuntimeError("official gain-count closure did not pass")
    authorization = config["authorization"]
    for key in (
        "read_gain_history_array_columns",
        "read_event_rows_or_energies",
        "write_or_modify_gain_history",
        "apply_or_interpolate_gain_to_events",
        "fit_calibration_or_science_spectrum",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed gain-reconstruction boundary is open: {key}")
    return config, load_json(ROOT / config["parents"]["download_provenance"])


def verified_path(
    relative: str,
    raw_root: Path,
    provenance_by_path: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root):
        raise RuntimeError(f"gain-reconstruction path escapes raw root: {relative}")
    terminal = provenance_by_path.get(relative)
    if terminal is None:
        raise RuntimeError(f"gain-reconstruction source absent from provenance: {relative}")
    if not path.is_file() or path.stat().st_size != terminal["bytes"]:
        raise RuntimeError(f"gain-reconstruction source size changed: {relative}")
    if sha256(path) != terminal["sha256"]:
        raise RuntimeError(f"gain-reconstruction source hash changed: {relative}")
    return path, terminal


def read_scalar_rows(
    path: Path,
    extension: str,
    columns: list[str],
    obsid: str,
    kind: str,
) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as stream, fits.open(
        stream, memmap=False, mode="readonly"
    ) as hdus:
        if extension not in hdus:
            raise RuntimeError(f"missing extension {extension}: {path.name}")
        table = hdus[extension].data
        names = set(table.names or [])
        if not set(columns).issubset(names):
            raise RuntimeError(f"declared scalar columns missing from {path.name}")
        arrays = {name: np.asarray(table[name]).copy() for name in columns}
    rows: list[dict[str, Any]] = []
    for index in range(len(arrays["TIME"])):
        row: dict[str, Any] = {
            "obsid": obsid,
            "kind": kind,
            "source_row": index,
        }
        for name in columns:
            value = arrays[name][index]
            row[name] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def valid_rows(rows: list[dict[str, Any]], minimum_events: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        values = (row["TIME"], row["PIXEL"], row["TEMP_FIT"], row["NEVENT"], row["CHISQ"])
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        ) and int(row["NEVENT"]) >= minimum_events:
            output.append(row)
    return output


def detect_segments(rows: list[dict[str, Any]], gap_seconds: float) -> list[dict[str, Any]]:
    times = sorted({float(row["TIME"]) for row in rows})
    if not times:
        return []
    groups: list[list[float]] = [[times[0]]]
    for time in times[1:]:
        if time - groups[-1][-1] > gap_seconds:
            groups.append([time])
        else:
            groups[-1].append(time)
    output = []
    for index, group in enumerate(groups):
        start, stop = group[0], group[-1]
        selected = [row for row in rows if start <= float(row["TIME"]) <= stop]
        output.append(
            {
                "segment": index,
                "start": start,
                "stop": stop,
                "duration_seconds": stop - start,
                "rows": len(selected),
                "obsids": sorted({row["obsid"] for row in selected}),
                "per_pixel_rows": {
                    str(pixel): sum(int(row["PIXEL"]) == pixel for row in selected)
                    for pixel in range(36)
                },
            }
        )
    return output


def segment_signature(segments: list[dict[str, Any]]) -> list[tuple[float, float, int]]:
    return [
        (round(item["start"], 6), round(item["stop"], 6), item["rows"])
        for item in segments
    ]


def choose_anchors(
    branch: dict[str, Any], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    before = [segment for segment in segments if segment["stop"] <= branch["start"]]
    after = [segment for segment in segments if segment["start"] >= branch["stop"]]
    method = branch["method"]
    if method == "cross_segment_linear_fit":
        return ([max(before, key=lambda item: item["stop"])] if before else []) + (
            [min(after, key=lambda item: item["start"])] if after else []
        )
    if method == "preceding_segment_linear_extrapolation":
        return [max(before, key=lambda item: item["stop"])] if before else []
    if method == "following_segment_linear_extrapolation":
        return [min(after, key=lambda item: item["start"])] if after else []
    raise RuntimeError(f"unknown branch method: {method}")


def fit_pixel(
    rows: list[dict[str, Any]],
    pixel: int,
    anchors: list[dict[str, Any]],
) -> dict[str, float | int | bool]:
    selected = [
        row
        for row in rows
        if int(row["PIXEL"]) == pixel
        and any(anchor["start"] <= float(row["TIME"]) <= anchor["stop"] for anchor in anchors)
    ]
    times = np.asarray([float(row["TIME"]) for row in selected], dtype=float)
    temps = np.asarray([float(row["TEMP_FIT"]) for row in selected], dtype=float)
    if times.size < 2 or np.ptp(times) <= 0:
        return {"rows": int(times.size), "finite": False}
    center = float(np.mean(times))
    slope, intercept_centered = np.polyfit(times - center, temps, 1)
    predicted = intercept_centered + slope * (times - center)
    rmse = float(np.sqrt(np.mean((temps - predicted) ** 2)))
    return {
        "rows": int(times.size),
        "finite": bool(np.isfinite([slope, intercept_centered, rmse]).all()),
        "time_center": center,
        "temperature_at_center": float(intercept_centered),
        "slope_per_second": float(slope),
        "rmse": rmse,
    }


def predict(fit: dict[str, Any], times: np.ndarray) -> np.ndarray:
    return fit["temperature_at_center"] + fit["slope_per_second"] * (
        times - fit["time_center"]
    )


def residual_summary(values: np.ndarray, times: np.ndarray) -> dict[str, float | int | bool]:
    if values.size < 2 or np.ptp(times) <= 0:
        return {"rows": int(values.size), "finite": False}
    median = float(np.median(values))
    slope = float(np.polyfit(times - np.mean(times), values, 1)[0])
    summary = {
        "rows": int(values.size),
        "finite": True,
        "median": median,
        "median_absolute_deviation": float(np.median(np.abs(values - median))),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "linear_slope_per_hour": slope * 3600.0,
    }
    summary["finite"] = bool(
        np.isfinite([value for key, value in summary.items() if key not in {"rows", "finite"}]).all()
    )
    return summary


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, provenance = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    provenance_by_path = {
        record["download_path"]: record for record in provenance["records"]
    }
    columns = config["gain_histories"]["allowed_scalar_columns"]
    histories: dict[str, list[dict[str, Any]]] = {"fe55": [], "calpixel": []}
    sources: list[dict[str, Any]] = []
    for obsid in config["gain_histories"]["obsids"]:
        kinds = [("fe55", config["gain_histories"]["fe55_template"])]
        if obsid in config["gain_histories"]["science_obsids"]:
            kinds.append(
                ("calpixel", config["gain_histories"]["continuous_calpixel_template"])
            )
        for kind, template in kinds:
            relative = template.format(obsid=obsid)
            path, terminal = verified_path(relative, raw_root, provenance_by_path)
            rows = read_scalar_rows(
                path,
                config["gain_histories"]["extension"],
                columns,
                obsid,
                kind,
            )
            histories[kind].extend(rows)
            sources.append(
                {
                    "obsid": obsid,
                    "kind": kind,
                    "download_path": relative,
                    "bytes": terminal["bytes"],
                    "sha256": terminal["sha256"],
                    "rows": len(rows),
                }
            )
    minimum_events = config["gain_histories"]["minimum_events"]
    fe55 = valid_rows(histories["fe55"], minimum_events)
    calpixel = valid_rows(histories["calpixel"], minimum_events)
    thresholds = config["segment_detection"]["robustness_gap_seconds"]
    segment_sets = {str(gap): detect_segments(fe55, gap) for gap in thresholds}
    signatures = [segment_signature(segment_sets[str(gap)]) for gap in thresholds]
    topology_stable = all(signature == signatures[0] for signature in signatures[1:])
    primary_segments = segment_sets[str(config["segment_detection"]["primary_gap_seconds"])]

    all_pixels = config["gain_histories"]["main_array_pixels"] + [
        config["gain_histories"]["calibration_pixel"]
    ]
    minimum_anchor_rows = config["segment_detection"][
        "minimum_rows_per_pixel_in_anchor_segment"
    ]
    minimum_cal_rows = config["calibration_pixel_residual"][
        "minimum_continuous_rows_per_branch"
    ]
    branch_reports: list[dict[str, Any]] = []
    for branch in config["branches"]:
        anchors = choose_anchors(branch, primary_segments)
        fits = {str(pixel): fit_pixel(fe55, pixel, anchors) for pixel in all_pixels}
        pixel12_fit = fits[str(config["gain_histories"]["calibration_pixel"])]
        continuous_rows = [
            row
            for row in calpixel
            if row["obsid"] == branch["obsid"]
            and branch["start"] <= float(row["TIME"]) <= branch["stop"]
            and int(row["PIXEL"]) == config["gain_histories"]["calibration_pixel"]
        ]
        continuous_times = np.asarray(
            [float(row["TIME"]) for row in continuous_rows], dtype=float
        )
        continuous_temps = np.asarray(
            [float(row["TEMP_FIT"]) for row in continuous_rows], dtype=float
        )
        if pixel12_fit.get("finite") and continuous_times.size:
            residuals = continuous_temps - predict(pixel12_fit, continuous_times)
            residual = residual_summary(residuals, continuous_times)
        else:
            residual = {"rows": int(continuous_times.size), "finite": False}
        expected_anchor_count = 2 if branch["method"] == "cross_segment_linear_fit" else 1
        anchors_complete = len(anchors) == expected_anchor_count and all(
            all(
                anchor["per_pixel_rows"].get(str(pixel), 0) >= minimum_anchor_rows
                for pixel in all_pixels
            )
            for anchor in anchors
        )
        fits_complete = all(item.get("finite", False) for item in fits.values())
        branch_reports.append(
            {
                **branch,
                "anchors": anchors,
                "anchor_segments_complete": anchors_complete,
                "fits": fits,
                "all_pixel_fits_finite": fits_complete,
                "calibration_pixel_residual": residual,
                "continuous_calpixel_rows_sufficient": residual["rows"] >= minimum_cal_rows,
            }
        )
    branch_gate = all(
        item["anchor_segments_complete"]
        and item["all_pixel_fits_finite"]
        and item["continuous_calpixel_rows_sufficient"]
        and item["calibration_pixel_residual"].get("finite", False)
        for item in branch_reports
    )
    passed = topology_stable and branch_gate
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_gain_reconstruction_scalar_topology_audited",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "source_files": sources,
        "raw_rows": {kind: len(rows) for kind, rows in histories.items()},
        "valid_rows": {"fe55": len(fe55), "calpixel": len(calpixel)},
        "segment_sets": segment_sets,
        "segment_topology_stable": topology_stable,
        "branches": branch_reports,
        "excluded_science_interval": config["excluded_science_interval"],
        "branch_gate_passed": branch_gate,
        "topology_gate_passed": passed,
        "gain_history_array_column_read": False,
        "event_row_or_energy_read": False,
        "gain_history_written_or_modified": False,
        "gain_applied_or_interpolated_to_events": False,
        "calibration_or_science_spectrum_fit": False,
        "cluster_velocity_fit": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_calibration_application_candidate_freeze"
            if passed
            else "stop_before_calibration_application_candidate_freeze"
        ),
        "authorization": {
            "freeze_calibration_application_candidates": passed,
            "read_gain_history_array_columns": False,
            "read_event_rows_or_energies": False,
            "write_or_modify_gain_history": False,
            "apply_or_interpolate_gain_to_events": False,
            "fit_calibration_or_science_spectrum": False,
            "fit_cluster_velocity": False,
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
                "status": result["status"],
                "raw_rows": result["raw_rows"],
                "valid_rows": result["valid_rows"],
                "segment_counts": {
                    gap: len(segments) for gap, segments in result["segment_sets"].items()
                },
                "segment_topology_stable": result["segment_topology_stable"],
                "branch_summaries": [
                    {
                        "name": branch["name"],
                        "anchor_segments": [item["segment"] for item in branch["anchors"]],
                        "anchors_complete": branch["anchor_segments_complete"],
                        "fits_finite": branch["all_pixel_fits_finite"],
                        "calpixel_residual": branch["calibration_pixel_residual"],
                    }
                    for branch in result["branches"]
                ],
                "topology_gate_passed": result["topology_gate_passed"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
