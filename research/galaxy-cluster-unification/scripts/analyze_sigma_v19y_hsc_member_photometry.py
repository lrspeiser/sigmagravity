#!/usr/bin/env python3
"""Summarize V19Y candidate coverage without selecting a counterpart."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT / "results" / "sigma_v19y_hsc_member_photometry" / "provenance.json"
)
DEFAULT_OUTPUT = DEFAULT_REPORT.parent / "coverage_analysis.json"
FILTERS = ("A_F435W", "A_F606W", "A_F814W")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p75": percentile(values, 0.75),
        "maximum": max(values) if values else None,
    }


def angular_separation_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    delta_ra = ra2 - ra1
    delta_dec = dec2 - dec1
    a = (
        math.sin(delta_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    )
    angle = 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))
    return math.degrees(angle) * 3600.0


def has_measurement(row: dict[str, str], column: str) -> bool:
    value = (row.get(column) or "").strip()
    count = (row.get(f"{column}_N") or "").strip()
    if not value or not count:
        return False
    try:
        return math.isfinite(float(value)) and float(count) > 0.0
    except ValueError:
        return False


def read_candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    path = ROOT / record["csv_path"]
    if sha256(path) != record["csv_sha256"]:
        raise RuntimeError(f"V19Y raw hash mismatch: {path}")
    if path.stat().st_size == 0:
        rows: list[dict[str, str]] = []
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if len(rows) != int(record["candidate_rows"]):
        raise RuntimeError(f"V19Y candidate count mismatch: {path}")
    return rows


def cluster_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts: list[float] = []
    all_separations: list[float] = []
    nearest_separations: list[float] = []
    filter_candidate_counts = Counter({name: 0 for name in FILTERS})
    filter_cone_counts = Counter({name: 0 for name in FILTERS})
    combination_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    all_three_candidates = 0
    exact_single_all_three_cones = 0
    repeated_image_candidates = 0

    for record in records:
        rows = read_candidates(record)
        candidate_counts.append(float(len(rows)))
        returned = record["returned_columns"]
        schema_key = "+".join(column for column in FILTERS if column in returned)
        schema_counts[schema_key or "EMPTY_CONE"] += 1
        cone_has_filter = {name: False for name in FILTERS}
        cone_all_three = False
        separations: list[float] = []
        for row in rows:
            present = tuple(name for name in FILTERS if has_measurement(row, name))
            combination_counts["+".join(present) if present else "NO_REQUESTED_BAND"] += 1
            for name in present:
                filter_candidate_counts[name] += 1
                cone_has_filter[name] = True
            if len(present) == len(FILTERS):
                all_three_candidates += 1
                cone_all_three = True
            try:
                repeated_image_candidates += float(row.get("NumImages") or 0.0) > 1.0
            except ValueError:
                pass
            separation = angular_separation_arcsec(
                float(record["ra_deg"]),
                float(record["dec_deg"]),
                float(row["MatchRA"]),
                float(row["MatchDec"]),
            )
            separations.append(separation)
            all_separations.append(separation)
        for name, present in cone_has_filter.items():
            filter_cone_counts[name] += present
        if separations:
            nearest_separations.append(min(separations))
        if len(rows) == 1 and cone_all_three:
            exact_single_all_three_cones += 1

    member_count = len(records)
    candidate_total = int(sum(candidate_counts))
    nonempty = sum(count > 0 for count in candidate_counts)
    single = sum(count == 1 for count in candidate_counts)
    multiple = sum(count > 1 for count in candidate_counts)
    return {
        "member_cones": member_count,
        "candidate_rows": candidate_total,
        "members_without_candidates": member_count - nonempty,
        "members_with_candidates": nonempty,
        "members_with_exactly_one_candidate": single,
        "members_with_multiple_candidates": multiple,
        "candidate_count_per_member": distribution(candidate_counts),
        "returned_filter_schema_cones": dict(sorted(schema_counts.items())),
        "candidate_band_combinations": dict(sorted(combination_counts.items())),
        "candidate_measurement_coverage": {
            name: {
                "candidate_rows": filter_candidate_counts[name],
                "fraction_of_candidates": filter_candidate_counts[name] / candidate_total
                if candidate_total
                else None,
                "member_cones_with_at_least_one": filter_cone_counts[name],
                "fraction_of_all_members": filter_cone_counts[name] / member_count,
            }
            for name in FILTERS
        },
        "candidates_with_all_three_bands": all_three_candidates,
        "fraction_of_candidates_with_all_three_bands": all_three_candidates
        / candidate_total
        if candidate_total
        else None,
        "members_with_exactly_one_candidate_and_all_three_bands": exact_single_all_three_cones,
        "candidates_with_num_images_gt_one": repeated_image_candidates,
        "fraction_of_candidates_with_num_images_gt_one": repeated_image_candidates
        / candidate_total
        if candidate_total
        else None,
        "all_candidate_separation_arcsec": distribution(all_separations),
        "nearest_candidate_separation_per_nonempty_cone_arcsec": distribution(
            nearest_separations
        ),
        "counterpart_selected": False,
    }


def analyze(report_path: Path, output_path: Path) -> dict[str, Any]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "all_frozen_member_cones_acquired_unmatched_and_hashed":
        raise RuntimeError("V19Y acquisition is not complete")
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for record in report["records"]:
        by_cluster.setdefault(record["cluster"], []).append(record)
    summaries = {
        cluster: cluster_summary(records)
        for cluster, records in sorted(by_cluster.items())
    }
    payload = {
        "status": "v19y_candidate_coverage_described_without_matching",
        "generated_utc": datetime.now(UTC).isoformat(),
        "acquisition_report": report_path.relative_to(ROOT).as_posix(),
        "acquisition_report_sha256": sha256(report_path),
        "analysis_runner": Path(__file__).relative_to(ROOT).as_posix(),
        "analysis_runner_sha256": sha256(Path(__file__)),
        "clusters": summaries,
        "candidate_rows": sum(row["candidate_rows"] for row in summaries.values()),
        "counterpart_selection_performed": False,
        "photometric_quality_cut_performed": False,
        "stellar_mass_inference_performed": False,
        "mass_current_constructed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = analyze(args.report, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
