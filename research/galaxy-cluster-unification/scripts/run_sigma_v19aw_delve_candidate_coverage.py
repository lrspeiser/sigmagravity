#!/usr/bin/env python3
"""Acquire the frozen DELVE DR3 field and audit all V19AU candidate matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19aw_delve_candidate_coverage.json"
USER_AGENT = "SigmaGravity-V19AW-DELVE-candidate-coverage/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with partial.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def build_adql(config: dict[str, Any]) -> str:
    source = config["source"]
    columns = ",".join(source["query_columns"])
    return (
        f"SELECT {columns} FROM {source['table']} "
        "WHERE 't'=q3c_radial_query(ra,dec,"
        f"{float(source['center_ra_deg']):.15g},{float(source['center_dec_deg']):.15g},"
        f"{float(source['field_radius_deg']):.15g}) ORDER BY {source['id_column']}"
    )


def query_url(config: dict[str, Any], adql: str) -> str:
    return config["source"]["endpoint"] + "?" + urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    )


def parse_payload(payload: bytes, expected_columns: list[str]) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != expected_columns:
        raise RuntimeError(f"DELVE response schema changed: {reader.fieldnames!r}")
    rows = list(reader)
    id_column = expected_columns[0]
    ids = [row[id_column] for row in rows]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise RuntimeError(f"DELVE {id_column} is missing or duplicated")
    for row in rows:
        for ra_column, dec_column in (
            ("ra", "dec"),
            ("alphawin_j2000", "deltawin_j2000"),
        ):
            ra = float(row[ra_column])
            dec = float(row[dec_column])
            if math.isfinite(ra) and 0 <= ra < 360 and math.isfinite(dec) and -90 <= dec <= 90:
                continue
            raise RuntimeError(f"invalid DELVE position: {row[id_column]}")
    return rows


def finite_number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_photometry(
    row: dict[str, str], bands: list[str], observable: dict[str, Any]
) -> bool:
    for band in bands:
        flux = finite_number(row[observable["flux_column_template"].format(band=band)])
        uncertainty = finite_number(
            row[observable["uncertainty_column_template"].format(band=band)]
        )
        epochs = finite_number(row[observable["epochs_column_template"].format(band=band)])
        if flux is None or uncertainty is None or epochs is None:
            return False
        if uncertainty <= 0 or epochs < int(observable["minimum_epochs_per_band"]):
            return False
    return True


def match_candidates(
    config: dict[str, Any],
    field_rows: list[dict[str, str]],
    candidates: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bands = list(config["matching"]["required_photometry_bands"])
    radius_arcsec = float(config["matching"]["radius_arcsec"])
    source = config["source"]
    id_column = source["id_column"]
    match_ra_column = source["match_ra_column"]
    match_dec_column = source["match_dec_column"]
    observable = config["coverage_observable"]
    field_sky = SkyCoord(
        [float(row[match_ra_column]) for row in field_rows],
        [float(row[match_dec_column]) for row in field_rows],
        unit="deg",
    )
    matches: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for candidate in candidates:
        sky = SkyCoord(
            float(candidate["candidate_ra_deg"]),
            float(candidate["candidate_dec_deg"]),
            unit="deg",
        )
        separations = sky.separation(field_sky).arcsec
        indices = np.where(separations <= radius_arcsec)[0]
        complete_matches = 0
        for index in sorted(
            indices, key=lambda value: (separations[value], field_rows[value][id_column])
        ):
            row = field_rows[index]
            complete = finite_photometry(row, bands, observable)
            complete_matches += complete
            matches.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_ra_deg": candidate["candidate_ra_deg"],
                    "candidate_dec_deg": candidate["candidate_dec_deg"],
                    "delve_object_id": row[id_column],
                    "separation_arcsec": float(separations[index]),
                    "complete_signed_flux_griz": complete,
                    **row,
                }
            )
        coverage.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_ra_deg": candidate["candidate_ra_deg"],
                "candidate_dec_deg": candidate["candidate_dec_deg"],
                "delve_matches_within_radius": int(indices.size),
                "complete_signed_flux_griz_matches": complete_matches,
                "has_complete_signed_flux_griz_match": complete_matches > 0,
                "matching_state": "no_match"
                if indices.size == 0
                else "unique"
                if indices.size == 1
                else "multiple",
            }
        )
    return matches, coverage


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for parent in config["parent_artifacts"]:
        path = ROOT / parent["path"]
        if sha256(path) != parent["sha256"]:
            raise RuntimeError(f"parent artifact hash changed: {parent['path']}")

    hypotheses = read_csv(ROOT / config["inputs"]["candidate_hypotheses"])
    candidates_by_id: dict[str, dict[str, str]] = {}
    for row in hypotheses:
        candidate = {
                "candidate_id": row["candidate_id"],
                "candidate_ra_deg": row["candidate_ra_deg"],
                "candidate_dec_deg": row["candidate_dec_deg"],
        }
        previous = candidates_by_id.setdefault(row["candidate_id"], candidate)
        if previous != candidate:
            raise RuntimeError(f"candidate coordinates disagree: {row['candidate_id']}")
    candidates = sorted(
        candidates_by_id.values(),
        key=lambda row: row["candidate_id"],
    )
    if len(candidates) != int(config["gates"]["exact_unique_candidates"]):
        raise RuntimeError("candidate count changed")

    adql = build_adql(config)
    url = query_url(config, adql)
    outputs = config["outputs"]
    raw_csv = ROOT / outputs["raw_field_csv"]
    raw_adql = ROOT / outputs["raw_adql"]
    raw_url = ROOT / outputs["raw_query_url"]
    expected_adql = (adql + "\n").encode()
    expected_url = (url + "\n").encode()
    if any(path.exists() for path in (raw_csv, raw_adql, raw_url)):
        if not all(path.exists() for path in (raw_csv, raw_adql, raw_url)):
            raise RuntimeError("partial DELVE field acquisition exists")
        if raw_adql.read_bytes() != expected_adql or raw_url.read_bytes() != expected_url:
            raise RuntimeError("frozen DELVE query changed")
        payload = raw_csv.read_bytes()
        reused = True
    else:
        request = urllib.request.Request(
            config["source"]["endpoint"],
            data=urllib.parse.urlencode(
                {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
            ).encode(),
            headers={"User-Agent": USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=float(config["source"]["timeout_seconds"])
        ) as response:
            if int(response.status) != 200:
                raise RuntimeError(f"DELVE returned HTTP {response.status}")
            payload = response.read()
        write_atomic(raw_csv, payload)
        write_atomic(raw_adql, expected_adql)
        write_atomic(raw_url, expected_url)
        reused = False

    field_rows = parse_payload(payload, config["source"]["query_columns"])
    if len(field_rows) != int(config["gates"]["exact_field_rows"]):
        raise RuntimeError(f"DELVE field row count changed: {len(field_rows)}")
    matches, coverage = match_candidates(config, field_rows, candidates)

    matches_path = ROOT / outputs["candidate_matches"]
    coverage_path = ROOT / outputs["candidate_coverage"]
    match_fields = [
        "candidate_id",
        "candidate_ra_deg",
        "candidate_dec_deg",
        "delve_object_id",
        "separation_arcsec",
        "complete_signed_flux_griz",
        *config["source"]["query_columns"],
    ]
    write_csv(matches_path, matches, match_fields)
    write_csv(coverage_path, coverage, list(coverage[0]))

    complete_candidates = {
        row["candidate_id"]
        for row in coverage
        if row["has_complete_signed_flux_griz_match"]
    }
    member_candidates: dict[str, set[str]] = defaultdict(set)
    for row in hypotheses:
        member_candidates[row["member_id"]].add(row["candidate_id"])
    members_with_complete = sum(
        bool(candidate_ids & complete_candidates) for candidate_ids in member_candidates.values()
    )
    matching_states = {
        state: sum(row["matching_state"] == state for row in coverage)
        for state in ("no_match", "unique", "multiple")
    }
    gate_results = {
        "exact_field_rows": len(field_rows) == int(config["gates"]["exact_field_rows"]),
        "all_candidates_evaluated": len(coverage)
        == int(config["gates"]["exact_unique_candidates"]),
        "complete_candidate_fraction": len(complete_candidates) / len(candidates)
        >= float(config["gates"]["minimum_complete_candidate_fraction"]),
        "every_member_has_complete_candidate": members_with_complete
        == int(config["gates"]["exact_members"]),
        "no_counterpart_selected": True,
    }
    passed = all(gate_results.values())
    report = {
        "protocol_version": config["protocol_version"],
        "decision": "passed" if passed else "failed_closed",
        "field": {
            "rows": len(field_rows),
            "reused_verified_payload": reused,
            "raw_csv_sha256": sha256(raw_csv),
            "raw_adql_sha256": sha256(raw_adql),
            "raw_query_url_sha256": sha256(raw_url),
        },
        "candidate_coverage": {
            "candidates": len(candidates),
            "matches": len(matches),
            "matching_states": matching_states,
            "complete_signed_flux_griz_candidates": len(complete_candidates),
            "complete_signed_flux_griz_fraction": len(complete_candidates) / len(candidates),
            "members_with_at_least_one_complete_candidate": members_with_complete,
        },
        "gate_results": gate_results,
        "candidate_selected_or_ranked": False,
        "outputs": {
            "candidate_matches": matches_path.relative_to(ROOT).as_posix(),
            "candidate_matches_sha256": sha256(matches_path),
            "candidate_coverage": coverage_path.relative_to(ROOT).as_posix(),
            "candidate_coverage_sha256": sha256(coverage_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
