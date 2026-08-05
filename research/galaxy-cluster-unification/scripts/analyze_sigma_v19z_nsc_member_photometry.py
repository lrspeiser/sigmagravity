#!/usr/bin/env python3
"""Describe V19Z NSC coverage without selecting any member counterpart."""

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
    ROOT / "results" / "sigma_v19z_nsc_member_photometry" / "provenance.json"
)
DEFAULT_HSC_REPORT = (
    ROOT / "results" / "sigma_v19y_hsc_member_photometry" / "provenance.json"
)
DEFAULT_OUTPUT = DEFAULT_REPORT.parent / "coverage_analysis.json"
BANDS = ("u", "g", "r", "i", "z", "y", "vr")
CORE_BANDS = ("g", "r", "i", "z")


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
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))) * 3600.0


def has_band(row: dict[str, str], band: str) -> bool:
    try:
        magnitude = float(row[f"{band}mag"])
        uncertainty = float(row[f"{band}err"])
        detections = float(row[f"nphot{band}"])
    except (KeyError, TypeError, ValueError):
        return False
    # NSC encodes absent magnitudes/errors as 99.99/9.99.  This distinction
    # describes catalog coverage; it is not a downstream quality selection.
    return (
        math.isfinite(magnitude)
        and math.isfinite(uncertainty)
        and magnitude < 90.0
        and uncertainty < 9.0
        and detections > 0.0
    )


def read_candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    path = ROOT / record["csv_path"]
    if sha256(path) != record["csv_sha256"]:
        raise RuntimeError(f"V19Z raw hash mismatch: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(record["candidate_rows"]):
        raise RuntimeError(f"V19Z candidate count mismatch: {path}")
    return rows


def load_published_bri(report: dict[str, Any]) -> dict[str, bool]:
    path = ROOT / report["published_bullet_photometry"]["output"]
    if sha256(path) != report["published_bullet_photometry"]["sha256"]:
        raise RuntimeError("V19Z published Bullet photometry hash changed")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["object_id"]: row["published_bri_available"] == "True"
            for row in csv.DictReader(handle)
        }


def cluster_summary(
    records: list[dict[str, Any]], published_bri: dict[str, bool]
) -> dict[str, Any]:
    candidate_counts: list[float] = []
    separations: list[float] = []
    nearest: list[float] = []
    band_candidates = Counter({band: 0 for band in BANDS})
    band_cones = Counter({band: 0 for band in BANDS})
    combinations: Counter[str] = Counter()
    all_core_candidates = 0
    single_all_core = 0
    repeated_photometry = 0
    class_star_values: list[float] = []
    proper_motion_candidates = 0
    bri_nonempty = 0
    bri_single = 0
    bri_single_all_core = 0

    for record in records:
        rows = read_candidates(record)
        candidate_counts.append(float(len(rows)))
        cone_has_band = {band: False for band in BANDS}
        cone_separations: list[float] = []
        single_has_core = False
        for row in rows:
            present = tuple(band for band in BANDS if has_band(row, band))
            combinations["+".join(present) if present else "NO_CATALOG_BAND"] += 1
            for band in present:
                band_candidates[band] += 1
                cone_has_band[band] = True
            has_core = all(band in present for band in CORE_BANDS)
            all_core_candidates += has_core
            single_has_core = has_core
            try:
                repeated_photometry += float(row["nphot"]) > 1.0
            except (TypeError, ValueError):
                pass
            try:
                value = float(row["class_star"])
                if math.isfinite(value):
                    class_star_values.append(value)
            except (TypeError, ValueError):
                pass
            try:
                pm_values = [
                    float(row["pmra"]),
                    float(row["pmdec"]),
                    float(row["pmraerr"]),
                    float(row["pmdecerr"]),
                ]
                proper_motion_candidates += all(math.isfinite(value) for value in pm_values)
            except (TypeError, ValueError):
                pass
            separation = angular_separation_arcsec(
                float(record["ra_deg"]),
                float(record["dec_deg"]),
                float(row["ra"]),
                float(row["dec"]),
            )
            separations.append(separation)
            cone_separations.append(separation)
        for band, present in cone_has_band.items():
            band_cones[band] += present
        if cone_separations:
            nearest.append(min(cone_separations))
        if len(rows) == 1 and single_has_core:
            single_all_core += 1
        has_bri = published_bri.get(record["object_id"], False)
        if has_bri and rows:
            bri_nonempty += 1
        if has_bri and len(rows) == 1:
            bri_single += 1
            bri_single_all_core += single_has_core

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
        "candidate_band_combinations": dict(sorted(combinations.items())),
        "candidate_band_coverage": {
            band: {
                "candidate_rows": band_candidates[band],
                "fraction_of_candidates": band_candidates[band] / candidate_total
                if candidate_total
                else None,
                "member_cones_with_at_least_one": band_cones[band],
                "fraction_of_all_members": band_cones[band] / member_count,
            }
            for band in BANDS
        },
        "candidates_with_all_griz": all_core_candidates,
        "fraction_of_candidates_with_all_griz": all_core_candidates / candidate_total
        if candidate_total
        else None,
        "members_with_exactly_one_candidate_and_all_griz": single_all_core,
        "candidates_with_nphot_gt_one": repeated_photometry,
        "candidates_with_finite_proper_motion_tuple": proper_motion_candidates,
        "class_star_distribution": distribution(class_star_values),
        "all_candidate_separation_arcsec": distribution(separations),
        "nearest_candidate_separation_per_nonempty_cone_arcsec": distribution(nearest),
        "members_with_published_bri_and_candidate": bri_nonempty,
        "members_with_published_bri_and_exactly_one_candidate": bri_single,
        "members_with_published_bri_exactly_one_candidate_and_all_griz": (
            bri_single_all_core
        ),
        "counterpart_selected": False,
    }


def hsc_presence_comparison(
    nsc_records: list[dict[str, Any]], hsc_records: list[dict[str, Any]]
) -> dict[str, int]:
    nsc = {
        (row["cluster"], row["object_id"]): int(row["candidate_rows"]) > 0
        for row in nsc_records
    }
    hsc = {
        (row["cluster"], row["object_id"]): int(row["candidate_rows"]) > 0
        for row in hsc_records
    }
    if set(nsc) != set(hsc):
        raise RuntimeError("V19Y/V19Z member keys differ")
    counts = Counter()
    for key, nsc_present in nsc.items():
        cluster = key[0]
        if nsc_present and hsc[key]:
            state = "both"
        elif nsc_present:
            state = "nsc_only"
        elif hsc[key]:
            state = "hsc_only"
        else:
            state = "neither"
        counts[f"{cluster}:{state}"] += 1
    return dict(sorted(counts.items()))


def analyze(report_path: Path, hsc_report_path: Path, output_path: Path) -> dict[str, Any]:
    report_path = report_path.resolve()
    hsc_report_path = hsc_report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hsc = json.loads(hsc_report_path.read_text(encoding="utf-8"))
    if report["status"] != (
        "all_frozen_nsc_member_cones_and_published_bullet_bri_acquired"
    ):
        raise RuntimeError("V19Z acquisition is not complete")
    published_bri = load_published_bri(report)
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for record in report["records"]:
        by_cluster.setdefault(record["cluster"], []).append(record)
    summaries = {
        cluster: cluster_summary(
            records, published_bri if cluster == "BULLET" else {}
        )
        for cluster, records in sorted(by_cluster.items())
    }
    payload = {
        "status": "v19z_nsc_candidate_coverage_described_without_matching",
        "generated_utc": datetime.now(UTC).isoformat(),
        "acquisition_report": report_path.relative_to(ROOT).as_posix(),
        "acquisition_report_sha256": sha256(report_path),
        "hsc_acquisition_report": hsc_report_path.relative_to(ROOT).as_posix(),
        "hsc_acquisition_report_sha256": sha256(hsc_report_path),
        "analysis_runner": Path(__file__).relative_to(ROOT).as_posix(),
        "analysis_runner_sha256": sha256(Path(__file__)),
        "clusters": summaries,
        "hsc_nsc_presence_comparison": hsc_presence_comparison(
            report["records"], hsc["records"]
        ),
        "candidate_rows": sum(row["candidate_rows"] for row in summaries.values()),
        "published_bullet_complete_bri_rows": sum(published_bri.values()),
        "counterpart_selection_performed": False,
        "photometric_quality_cut_performed": False,
        "filter_transformation_performed": False,
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
    parser.add_argument("--hsc-report", type=Path, default=DEFAULT_HSC_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = analyze(args.report, args.hsc_report, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
