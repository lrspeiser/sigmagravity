#!/usr/bin/env python3
"""Acquire every frozen HSC candidate around the V19D cluster members.

V19Y is an acquisition-only gate.  It preserves the complete cone response
for every member coordinate and intentionally performs no candidate choice,
photometric quality cut, color conversion, or stellar-mass inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19y_hsc_member_photometry.json"
USER_AGENT = "SigmaGravity-V19Y-HSC-source-audit/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def member_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_identifier(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not token or token in {".", ".."}:
        raise RuntimeError(f"unsafe member identifier: {value!r}")
    return token


def validate_config(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if not config["status"].startswith("frozen before querying"):
        raise RuntimeError("V19Y acquisition protocol is not frozen")
    authorization = config["authorization"]
    if not authorization["query_only_frozen_member_cones"]:
        raise RuntimeError("V19Y member-cone acquisition is not authorized")
    prohibited = (
        "select_hsc_counterpart",
        "apply_photometric_quality_cut",
        "derive_stellar_mass",
        "construct_mass_current",
        "read_lensing_or_halo_payload",
        "fit_gravity_parameters",
        "open_holdout",
    )
    if any(authorization[key] for key in prohibited):
        raise RuntimeError("V19Y authorizes a prohibited downstream action")

    hashes = {"config": sha256(config_path), "runner": sha256(Path(__file__))}
    frozen_runner_hash = config["implementation"]["runner_sha256"]
    if hashes["runner"] != frozen_runner_hash:
        raise RuntimeError("V19Y runner differs from its frozen hash")

    parents = config["parents"]
    for key, value in parents.items():
        if key.endswith("_sha256"):
            continue
        expected = parents.get(f"{key}_sha256")
        if expected is None:
            continue
        path = ROOT / value
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"V19Y parent hash mismatch: {value}")
        hashes[key] = actual

    query_columns = config["source"]["query_columns"]
    if len(query_columns) != len(set(query_columns)):
        raise RuntimeError("V19Y query columns are not unique")
    required_astrometry = {"MatchID", "MatchRA", "MatchDec"}
    if not required_astrometry.issubset(query_columns):
        raise RuntimeError("V19Y query omits required HSC astrometry")

    all_members: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for cluster, catalog in config["member_catalogs"].items():
        rows = member_rows(ROOT / catalog["path"])
        if len(rows) != int(catalog["expected_rows"]):
            raise RuntimeError(f"V19Y {cluster} row count changed")
        if not math.isfinite(float(catalog["query_radius_arcsec"])):
            raise RuntimeError(f"V19Y {cluster} radius is not finite")
        if float(catalog["query_radius_arcsec"]) <= 0.0:
            raise RuntimeError(f"V19Y {cluster} radius is not positive")
        for row in rows:
            if row["cluster"] != cluster:
                raise RuntimeError(f"V19Y cluster label changed in {catalog['path']}")
            key = (cluster, row["object_id"])
            if key in seen:
                raise RuntimeError(f"duplicate V19Y member key: {key}")
            seen.add(key)
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            if not (math.isfinite(ra) and 0.0 <= ra < 360.0):
                raise RuntimeError(f"invalid V19Y right ascension: {key}")
            if not (math.isfinite(dec) and -90.0 <= dec <= 90.0):
                raise RuntimeError(f"invalid V19Y declination: {key}")
            all_members.append(
                {
                    "cluster": cluster,
                    "object_id": row["object_id"],
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "query_radius_arcsec": float(catalog["query_radius_arcsec"]),
                }
            )
    if len(all_members) != int(config["gates"]["exact_member_query_count"]):
        raise RuntimeError("V19Y frozen total member count changed")
    return hashes, all_members


def build_query_url(config: dict[str, Any], member: dict[str, Any]) -> str:
    source = config["source"]
    parameters = {
        "ra": format(float(member["ra_deg"]), ".15g"),
        "dec": format(float(member["dec_deg"]), ".15g"),
        "radius": format(float(member["query_radius_arcsec"]) / 3600.0, ".15g"),
        "columns": "[" + ",".join(source["query_columns"]) + "]",
    }
    return source["endpoint"] + "?" + urllib.parse.urlencode(parameters)


def parse_response(payload: bytes, expected_columns: list[str]) -> int:
    if not payload.strip():
        # The HSC CSV endpoint returns a zero-byte body, rather than a header-only
        # CSV, for a valid cone having no catalog matches.
        return 0
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError("HSC response is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != expected_columns:
        raise RuntimeError(
            f"HSC response schema changed: {reader.fieldnames!r} != {expected_columns!r}"
        )
    count = 0
    seen_matches: set[str] = set()
    for row in reader:
        match_id = row["MatchID"].strip()
        if not match_id:
            raise RuntimeError("HSC candidate has no MatchID")
        if match_id in seen_matches:
            raise RuntimeError(f"HSC cone repeats MatchID {match_id}")
        seen_matches.add(match_id)
        ra = float(row["MatchRA"])
        dec = float(row["MatchDec"])
        if not (math.isfinite(ra) and 0.0 <= ra < 360.0):
            raise RuntimeError(f"invalid HSC MatchRA for {match_id}")
        if not (math.isfinite(dec) and -90.0 <= dec <= 90.0):
            raise RuntimeError(f"invalid HSC MatchDec for {match_id}")
        count += 1
    return count


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


def prior_records(report_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not report_path.exists():
        return {}
    report = load_json(report_path)
    return {
        (row["cluster"], row["object_id"]): row for row in report["records"]
    }


def acquire(
    config_path: Path = DEFAULT_CONFIG,
    raw_override: Path | None = None,
    output_override: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes, members = validate_config(config_path, config)
    raw_root = (
        raw_override.resolve()
        if raw_override is not None
        else (ROOT / config["outputs"]["raw_root"]).resolve()
    )
    output_root = (
        output_override.resolve()
        if output_override is not None
        else (ROOT / config["outputs"]["result_root"]).resolve()
    )
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "provenance.json"
    previous = prior_records(report_path)
    expected_columns = config["source"]["query_columns"]
    timeout = float(config["source"]["timeout_seconds"])
    records: list[dict[str, Any]] = []

    for index, member in enumerate(members, start=1):
        cluster_slug = member["cluster"].lower()
        identifier = safe_identifier(member["object_id"])
        csv_path = raw_root / cluster_slug / f"{identifier}.csv"
        url_path = raw_root / cluster_slug / f"{identifier}.url.txt"
        url = build_query_url(config, member)
        url_payload = (url + "\n").encode("utf-8")
        key = (member["cluster"], member["object_id"])
        prior = previous.get(key)
        reused = False

        if csv_path.exists() or url_path.exists():
            if not (csv_path.exists() and url_path.exists() and prior is not None):
                raise RuntimeError(f"unmanifested partial V19Y acquisition: {key}")
            if url_path.read_bytes() != url_payload:
                raise RuntimeError(f"V19Y query URL changed for {key}")
            if sha256(csv_path) != prior["csv_sha256"]:
                raise RuntimeError(f"V19Y cached CSV hash changed for {key}")
            if sha256(url_path) != prior["query_url_sha256"]:
                raise RuntimeError(f"V19Y cached URL hash changed for {key}")
            payload = csv_path.read_bytes()
            http_status = int(prior["http_status"])
            reused = True
        else:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                http_status = int(response.status)
                payload = response.read()
            if http_status != 200:
                raise RuntimeError(f"HSC returned HTTP {http_status} for {key}")
            write_atomic(csv_path, payload)
            write_atomic(url_path, url_payload)

        candidate_rows = parse_response(payload, expected_columns)
        if http_status != 200:
            raise RuntimeError(f"cached HSC status is not 200 for {key}")
        records.append(
            {
                **member,
                "query_url": url,
                "query_url_path": url_path.relative_to(ROOT).as_posix()
                if url_path.is_relative_to(ROOT)
                else str(url_path),
                "query_url_bytes": url_path.stat().st_size,
                "query_url_sha256": sha256(url_path),
                "csv_path": csv_path.relative_to(ROOT).as_posix()
                if csv_path.is_relative_to(ROOT)
                else str(csv_path),
                "csv_bytes": csv_path.stat().st_size,
                "csv_sha256": sha256(csv_path),
                "http_status": http_status,
                "candidate_rows": candidate_rows,
                "reused_verified_payload": reused,
                "counterpart_selected": False,
            }
        )
        print(
            f"[{index:03d}/{len(members):03d}] {member['cluster']} "
            f"{member['object_id']}: {candidate_rows} candidates "
            f"({'verified' if reused else 'downloaded'})",
            flush=True,
        )

    by_cluster: dict[str, dict[str, int]] = {}
    for cluster in config["member_catalogs"]:
        rows = [row for row in records if row["cluster"] == cluster]
        by_cluster[cluster] = {
            "member_queries": len(rows),
            "members_with_candidates": sum(row["candidate_rows"] > 0 for row in rows),
            "members_without_candidates": sum(row["candidate_rows"] == 0 for row in rows),
            "candidate_rows": sum(int(row["candidate_rows"]) for row in rows),
        }

    report = {
        "status": "all_frozen_member_cones_acquired_unmatched_and_hashed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": input_hashes,
        "query_count": len(records),
        "http_200_count": sum(row["http_status"] == 200 for row in records),
        "total_candidate_rows": sum(int(row["candidate_rows"]) for row in records),
        "by_cluster": by_cluster,
        "records": records,
        "all_raw_candidate_rows_retained": True,
        "counterpart_selection_performed": False,
        "photometric_quality_cut_performed": False,
        "stellar_mass_inference_performed": False,
        "mass_current_constructed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "downstream_matching_authorized": len(records)
        == int(config["gates"]["exact_member_query_count"])
        and all(row["http_status"] == 200 for row in records),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = acquire(args.config, args.raw, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "query_count": report["query_count"],
                "total_candidate_rows": report["total_candidate_rows"],
                "by_cluster": report["by_cluster"],
                "counterpart_selection_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
