#!/usr/bin/env python3
"""Audit A2319 gain-history and GTI scalar evidence without applying gain."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_gain_timeline_evidence.json"
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
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-GAIN-TIMELINE-1.0.0":
        raise RuntimeError("unexpected A2319 gain timeline protocol")
    if config.get("status") != (
        "frozen after the metadata-only closure and archived pipeline-log audit, but before reading any gain-history or GTI row value"
    ):
        raise RuntimeError("A2319 gain timeline protocol is not frozen")
    parents = config["parents"]
    metadata_path = ROOT / parents["metadata_report"]
    provenance_path = ROOT / parents["download_provenance"]
    for path, expected in (
        (metadata_path, parents["metadata_report_sha256"]),
        (provenance_path, parents["download_provenance_sha256"]),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen gain-timeline parent changed: {path}")
    metadata = load_json(metadata_path)
    if metadata.get("table_or_image_value_read") or metadata.get("validation_or_holdout_accessed"):
        raise RuntimeError("metadata parent violated its seal")
    authorization = config["authorization"]
    for key in (
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
            raise RuntimeError(f"sealed gain-timeline boundary is open: {key}")
    return config, load_json(provenance_path)


def verified_path(
    relative: str,
    raw_root: Path,
    provenance_by_path: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root):
        raise RuntimeError(f"gain-timeline path escapes raw root: {relative}")
    terminal = provenance_by_path.get(relative)
    if terminal is None:
        raise RuntimeError(f"gain-timeline source absent from provenance: {relative}")
    if not path.is_file() or path.stat().st_size != terminal["bytes"] or sha256(path) != terminal["sha256"]:
        raise RuntimeError(f"gain-timeline source changed: {relative}")
    return path, terminal


def read_columns(path: Path, extension: str, allowed: list[str]) -> dict[str, np.ndarray]:
    with gzip.open(path, "rb") as stream, fits.open(stream, memmap=False, mode="readonly") as hdus:
        if extension not in hdus:
            raise RuntimeError(f"missing extension {extension}: {path.name}")
        table = hdus[extension].data
        names = set(table.names or [])
        missing = set(allowed) - names
        if missing:
            raise RuntimeError(f"missing columns in {path.name}[{extension}]: {sorted(missing)}")
        return {name: np.asarray(table[name]).copy() for name in allowed}


def gain_rows(
    path: Path,
    extension: str,
    columns: list[str],
    obsid: str,
    kind: str,
) -> list[dict[str, Any]]:
    arrays = read_columns(path, extension, columns)
    rows: list[dict[str, Any]] = []
    for index in range(len(arrays["TIME"])):
        item: dict[str, Any] = {"obsid": obsid, "kind": kind, "source_row": index}
        for name in columns:
            value = arrays[name][index]
            if isinstance(value, np.generic):
                value = value.item()
            item[name] = value
        rows.append(item)
    return rows


def gti_rows(path: Path, extensions: list[str], columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for extension in extensions:
        arrays = read_columns(path, extension, columns)
        for index in range(len(arrays["START"])):
            rows.append(
                {
                    "extension": extension,
                    "source_row": index,
                    "start": float(arrays["START"][index]),
                    "stop": float(arrays["STOP"][index]),
                }
            )
    return rows


def finite_required(row: dict[str, Any]) -> bool:
    return all(
        isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
        for key in ("TIME", "PIXEL", "CHISQ", "WIDTH", "NEVENT", "TEMP_FIT", "TEMP_AVE")
    )


def deduplicate_time_pixel(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[float, int], dict[str, Any]] = {}
    for row in rows:
        key = (float(row["TIME"]), int(row["PIXEL"]))
        unique.setdefault(key, row)
    return list(unique.values())


def inside_any_interval(time: float, intervals: list[dict[str, Any]]) -> bool:
    return any(item["start"] <= time <= item["stop"] for item in intervals)


def count_variants(
    rows: list[dict[str, Any]],
    kind: str,
    config: dict[str, Any],
    open_intervals: list[dict[str, Any]],
) -> dict[str, int]:
    finite = [row for row in rows if finite_required(row)]
    min_events = config["gain_histories"]["minimum_events_from_archived_rslgain"]
    finite_nevent = [row for row in finite if int(row["NEVENT"]) >= min_events]
    dedup_raw = deduplicate_time_pixel(rows)
    dedup_valid = deduplicate_time_pixel(finite_nevent)
    target_pixels = (
        set(config["gain_histories"]["science_pixels"])
        if kind == "fe55"
        else {config["gain_histories"]["calibration_pixel"]}
    )
    target = [row for row in dedup_valid if int(row["PIXEL"]) in target_pixels]
    minimum = min(item["start"] for item in open_intervals)
    maximum = max(item["stop"] for item in open_intervals)
    return {
        "raw": len(rows),
        "finite_required": len(finite),
        "finite_required_and_nevent_at_least_200": len(finite_nevent),
        "deduplicate_exact_time_pixel": len(dedup_raw),
        "deduplicate_exact_time_pixel_after_finite_and_nevent": len(dedup_valid),
        "science_pixels_after_finite_nevent_and_deduplication": (
            len(target) if kind == "fe55" else 0
        ),
        "calibration_pixel_after_finite_nevent_and_deduplication": (
            len(target) if kind == "calpixel" else 0
        ),
        "within_total_open_filter_time_span": len(
            [row for row in target if minimum <= float(row["TIME"]) <= maximum]
        ),
        "inside_an_open_filter_interval": len(
            [row for row in target if inside_any_interval(float(row["TIME"]), open_intervals)]
        ),
    }


def coverage_rows(
    fe55_rows: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    science_pixels: list[int],
) -> list[dict[str, Any]]:
    valid = deduplicate_time_pixel(
        row for row in fe55_rows if finite_required(row) and int(row["NEVENT"]) >= 200
    )
    by_pixel = {
        pixel: sorted(float(row["TIME"]) for row in valid if int(row["PIXEL"]) == pixel)
        for pixel in science_pixels
    }
    output: list[dict[str, Any]] = []
    for interval in intervals:
        pixel_support: list[dict[str, Any]] = []
        for pixel in science_pixels:
            times = by_pixel[pixel]
            preceding = [time for time in times if time <= interval["start"]]
            following = [time for time in times if time >= interval["stop"]]
            pixel_support.append(
                {
                    "pixel": pixel,
                    "preceding_time": max(preceding) if preceding else None,
                    "following_time": min(following) if following else None,
                }
            )
        output.append(
            {
                **interval,
                "pixels_with_preceding_anchor": sum(
                    item["preceding_time"] is not None for item in pixel_support
                ),
                "pixels_with_following_anchor": sum(
                    item["following_time"] is not None for item in pixel_support
                ),
                "pixel_support": pixel_support,
            }
        )
    return output


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, provenance = validate_inputs(config_path)
    raw_root = (ROOT / config["paths"]["raw_root"]).resolve()
    provenance_by_path = {record["download_path"]: record for record in provenance["records"]}
    histories: dict[str, list[dict[str, Any]]] = {"fe55": [], "calpixel": []}
    source_files: list[dict[str, Any]] = []
    for obsid in config["gain_histories"]["obsids"]:
        for kind, template in config["gain_histories"]["templates"].items():
            relative = template.format(obsid=obsid)
            path, terminal = verified_path(relative, raw_root, provenance_by_path)
            rows = gain_rows(
                path,
                config["gain_histories"]["extension"],
                config["gain_histories"]["allowed_scalar_columns"],
                obsid,
                kind,
            )
            histories[kind].extend(rows)
            source_files.append(
                {
                    "kind": kind,
                    "obsid": obsid,
                    "download_path": relative,
                    "bytes": terminal["bytes"],
                    "sha256": terminal["sha256"],
                    "rows": len(rows),
                }
            )
    gti_evidence: dict[str, dict[str, list[dict[str, Any]]]] = {}
    open_intervals: list[dict[str, Any]] = []
    for obsid in config["science_gtis"]["obsids"]:
        gti_evidence[obsid] = {}
        for role, spec in config["science_gtis"]["files"].items():
            relative = spec["template"].format(obsid=obsid)
            path, _ = verified_path(relative, raw_root, provenance_by_path)
            rows = gti_rows(path, spec["extensions"], config["science_gtis"]["allowed_columns"])
            for row in rows:
                row["obsid"] = obsid
                row["role"] = role
            gti_evidence[obsid][role] = rows
            if role == "open_filter":
                open_intervals.extend(row for row in rows if row["stop"] > row["start"])
    open_intervals = sorted(open_intervals, key=lambda item: (item["start"], item["stop"]))
    variants = {
        kind: count_variants(rows, kind, config, open_intervals)
        for kind, rows in histories.items()
    }
    official = config["official_report_comparator"]
    matches = {
        "fe55": [name for name, value in variants["fe55"].items() if value == official["fe55_solutions"]],
        "calpixel": [
            name
            for name, value in variants["calpixel"].items()
            if value == official["calibration_pixel_solutions"]
        ],
    }
    coverage = coverage_rows(
        histories["fe55"],
        open_intervals,
        config["gain_histories"]["science_pixels"],
    )
    adr_candidates = [
        {
            **row,
            "six_hour_linear_start_candidate": row["start"]
            + config["science_gtis"]["post_adr_linear_delay_seconds"],
        }
        for obsid in config["science_gtis"]["obsids"]
        for row in gti_evidence[obsid]["adr"]
        if row["extension"] == "GTIADRON"
    ]
    count_closure = bool(matches["fe55"] and matches["calpixel"])
    report = {
        "protocol_version": config["protocol_version"],
        "status": "a2319_gain_timeline_scalar_evidence_audited",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "source_files": source_files,
        "gain_rows": {kind: len(rows) for kind, rows in histories.items()},
        "gti_evidence": gti_evidence,
        "open_filter_intervals": open_intervals,
        "preregistered_count_variants": variants,
        "official_count_matches": matches,
        "official_count_closure_reproduced": count_closure,
        "coverage": coverage,
        "adr_six_hour_candidates": adr_candidates,
        "gain_history_array_column_read": False,
        "event_row_or_energy_read": False,
        "gain_applied_or_interpolated": False,
        "spectrum_or_velocity_fit_performed": False,
        "validation_or_holdout_accessed": False,
        "decision": (
            "authorize_separate_gain_interpolation_protocol_freeze"
            if count_closure
            else "stop_before_gain_application_and_require_documented_solution_selection_rule"
        ),
        "authorization": {
            "freeze_gain_interpolation_protocol": count_closure,
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
                    "gain_rows",
                    "preregistered_count_variants",
                    "official_count_matches",
                    "official_count_closure_reproduced",
                    "decision",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
