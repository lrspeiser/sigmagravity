#!/usr/bin/env python3
"""Acquire frozen NSC candidates and extract the Bullet paper B/R/I table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tarfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19z_nsc_member_photometry.json"
USER_AGENT = "SigmaGravity-V19Z-NSC-source-audit/1.0"


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
        raise RuntimeError("V19Z acquisition protocol is not frozen")
    authorization = config["authorization"]
    if not authorization["query_only_frozen_member_cones"]:
        raise RuntimeError("V19Z member-cone acquisition is not authorized")
    prohibited = (
        "select_nsc_counterpart",
        "apply_photometric_quality_cut",
        "derive_stellar_mass",
        "construct_mass_current",
        "read_lensing_or_halo_payload",
        "fit_gravity_parameters",
        "open_holdout",
    )
    if any(authorization[key] for key in prohibited):
        raise RuntimeError("V19Z authorizes a prohibited downstream action")

    hashes = {"config": sha256(config_path), "runner": sha256(Path(__file__))}
    if hashes["runner"] != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19Z runner differs from its frozen hash")
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is None:
            continue
        actual = sha256(ROOT / value)
        if actual != expected:
            raise RuntimeError(f"V19Z parent hash mismatch: {value}")
        hashes[key] = actual

    columns = config["source"]["query_columns"]
    if len(columns) != len(set(columns)):
        raise RuntimeError("V19Z query columns are not unique")
    if not {"id", "ra", "dec"}.issubset(columns):
        raise RuntimeError("V19Z query omits required NSC identity or astrometry")

    all_members: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for cluster, catalog in config["member_catalogs"].items():
        rows = member_rows(ROOT / catalog["path"])
        if len(rows) != int(catalog["expected_rows"]):
            raise RuntimeError(f"V19Z {cluster} member count changed")
        for row in rows:
            if row["cluster"] != cluster:
                raise RuntimeError(f"V19Z cluster label changed in {catalog['path']}")
            key = (cluster, row["object_id"])
            if key in seen:
                raise RuntimeError(f"duplicate V19Z member key: {key}")
            seen.add(key)
            ra = float(row["ra_deg"])
            dec = float(row["dec_deg"])
            radius = float(catalog["query_radius_arcsec"])
            if not (math.isfinite(ra) and 0.0 <= ra < 360.0):
                raise RuntimeError(f"invalid V19Z right ascension: {key}")
            if not (math.isfinite(dec) and -90.0 <= dec <= 90.0):
                raise RuntimeError(f"invalid V19Z declination: {key}")
            if not (math.isfinite(radius) and radius > 0.0):
                raise RuntimeError(f"invalid V19Z query radius: {key}")
            all_members.append(
                {
                    "cluster": cluster,
                    "object_id": row["object_id"],
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "query_radius_arcsec": radius,
                }
            )
    if len(all_members) != int(config["gates"]["exact_member_query_count"]):
        raise RuntimeError("V19Z frozen total member count changed")
    return hashes, all_members


def clean_tex_cell(value: str) -> str:
    return value.strip().rstrip("\\").strip()


def numeric_or_blank(value: str) -> float | None:
    token = value.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", token):
        return None
    number = float(token)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite published photometry: {value!r}")
    return number


def extract_bullet_photometry(
    config: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    source = config["published_bullet_photometry"]
    archive_path = ROOT / source["archive"]
    with tarfile.open(archive_path, mode="r:*") as archive:
        handle = archive.extractfile(source["archive_member"])
        if handle is None:
            raise RuntimeError("V19Z Bullet TeX member is missing")
        text = handle.read().decode("latin-1")

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not re.match(r"^\s*\d{2}\s*&", line):
            continue
        cells = [clean_tex_cell(cell) for cell in line.split("&")]
        if len(cells) != 11:
            raise RuntimeError(f"unexpected Bullet photometry row: {line}")
        b_mag = numeric_or_blank(cells[3])
        b_minus_r = numeric_or_blank(cells[4])
        b_minus_i = numeric_or_blank(cells[5])
        available = all(value is not None for value in (b_mag, b_minus_r, b_minus_i))
        r_mag = b_mag - b_minus_r if available else None
        i_mag = b_mag - b_minus_i if available else None
        rows.append(
            {
                "cluster": "BULLET",
                "object_id": cells[0],
                "b_bessel_mag": b_mag,
                "b_minus_r_bessel_mag": b_minus_r,
                "b_minus_i_bessel_mag": b_minus_i,
                "r_bessel_mag": r_mag,
                "i_bessel_mag": i_mag,
                "published_bri_available": available,
                "source_arxiv_id": source["source_arxiv_id"],
                "source_table": source["source_table"],
            }
        )
    if len(rows) != int(source["expected_rows"]):
        raise RuntimeError("V19Z Bullet published-photometry row count changed")
    available_count = sum(row["published_bri_available"] for row in rows)
    if available_count != int(source["expected_complete_bri_rows"]):
        raise RuntimeError("V19Z Bullet complete B/R/I count changed")

    member_ids = {
        row["object_id"]
        for row in member_rows(ROOT / config["member_catalogs"]["BULLET"]["path"])
    }
    if {row["object_id"] for row in rows} != member_ids:
        raise RuntimeError("V19Z Bullet photometry/member identifiers differ")

    columns = [
        "cluster",
        "object_id",
        "b_bessel_mag",
        "b_minus_r_bessel_mag",
        "b_minus_i_bessel_mag",
        "r_bessel_mag",
        "i_bessel_mag",
        "published_bri_available",
        "source_arxiv_id",
        "source_table",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rows": len(rows),
        "complete_bri_rows": available_count,
        "missing_bri_rows": len(rows) - available_count,
        "output": output_path.relative_to(ROOT).as_posix()
        if output_path.is_relative_to(ROOT)
        else str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
        "counterpart_selection_performed": False,
    }


def build_adql(config: dict[str, Any], member: dict[str, Any]) -> str:
    source = config["source"]
    columns = ",".join(source["query_columns"])
    radius_deg = float(member["query_radius_arcsec"]) / 3600.0
    return (
        f"SELECT {columns} FROM {source['table']} "
        "WHERE 't'=q3c_radial_query(ra,dec,"
        f"{float(member['ra_deg']):.15g},{float(member['dec_deg']):.15g},"
        f"{radius_deg:.15g}) ORDER BY id"
    )


def build_query_url(config: dict[str, Any], adql: str) -> str:
    source = config["source"]
    parameters = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": adql,
    }
    return source["endpoint"] + "?" + urllib.parse.urlencode(parameters)


def parse_response(payload: bytes, expected_columns: list[str]) -> int:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError("NSC response is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != expected_columns:
        raise RuntimeError(f"NSC response schema changed: {reader.fieldnames!r}")
    seen_ids: set[str] = set()
    count = 0
    for row in reader:
        identifier = row["id"].strip()
        if not identifier or identifier in seen_ids:
            raise RuntimeError(f"missing or repeated NSC identifier: {identifier!r}")
        seen_ids.add(identifier)
        ra = float(row["ra"])
        dec = float(row["dec"])
        if not (math.isfinite(ra) and 0.0 <= ra < 360.0):
            raise RuntimeError(f"invalid NSC right ascension: {identifier}")
        if not (math.isfinite(dec) and -90.0 <= dec <= 90.0):
            raise RuntimeError(f"invalid NSC declination: {identifier}")
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


def previous_records(report_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
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
    derived_override: Path | None = None,
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
    derived_path = (
        derived_override.resolve()
        if derived_override is not None
        else (ROOT / config["outputs"]["bullet_published_bri"]).resolve()
    )
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "provenance.json"
    previous = previous_records(report_path)
    expected_columns = config["source"]["query_columns"]
    timeout = float(config["source"]["timeout_seconds"])
    published = extract_bullet_photometry(config, derived_path)
    records: list[dict[str, Any]] = []

    for index, member in enumerate(members, start=1):
        cluster_slug = member["cluster"].lower()
        identifier = safe_identifier(member["object_id"])
        csv_path = raw_root / cluster_slug / f"{identifier}.csv"
        url_path = raw_root / cluster_slug / f"{identifier}.url.txt"
        adql_path = raw_root / cluster_slug / f"{identifier}.adql.txt"
        adql = build_adql(config, member)
        url = build_query_url(config, adql)
        url_payload = (url + "\n").encode("utf-8")
        adql_payload = (adql + "\n").encode("utf-8")
        key = (member["cluster"], member["object_id"])
        prior = previous.get(key)
        reused = False

        existing = (csv_path.exists(), url_path.exists(), adql_path.exists())
        if any(existing):
            if not all(existing) or prior is None:
                raise RuntimeError(f"unmanifested partial V19Z acquisition: {key}")
            if url_path.read_bytes() != url_payload or adql_path.read_bytes() != adql_payload:
                raise RuntimeError(f"V19Z frozen query changed for {key}")
            for path, field in (
                (csv_path, "csv_sha256"),
                (url_path, "query_url_sha256"),
                (adql_path, "adql_sha256"),
            ):
                if sha256(path) != prior[field]:
                    raise RuntimeError(f"V19Z cached payload changed for {key}")
            payload = csv_path.read_bytes()
            http_status = int(prior["http_status"])
            reused = True
        else:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                http_status = int(response.status)
                payload = response.read()
            if http_status != 200:
                raise RuntimeError(f"NSC returned HTTP {http_status} for {key}")
            write_atomic(csv_path, payload)
            write_atomic(url_path, url_payload)
            write_atomic(adql_path, adql_payload)

        candidate_rows = parse_response(payload, expected_columns)
        if http_status != 200:
            raise RuntimeError(f"cached NSC status is not 200 for {key}")
        records.append(
            {
                **member,
                "adql": adql,
                "adql_path": adql_path.relative_to(ROOT).as_posix()
                if adql_path.is_relative_to(ROOT)
                else str(adql_path),
                "adql_bytes": adql_path.stat().st_size,
                "adql_sha256": sha256(adql_path),
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
        "status": "all_frozen_nsc_member_cones_and_published_bullet_bri_acquired",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": input_hashes,
        "published_bullet_photometry": published,
        "query_count": len(records),
        "http_200_count": sum(row["http_status"] == 200 for row in records),
        "total_candidate_rows": sum(int(row["candidate_rows"]) for row in records),
        "by_cluster": by_cluster,
        "records": records,
        "all_raw_candidate_rows_retained": True,
        "counterpart_selection_performed": False,
        "photometric_quality_cut_performed": False,
        "filter_transformation_performed": False,
        "stellar_mass_inference_performed": False,
        "mass_current_constructed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
        "downstream_matching_design_authorized": len(records)
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
    parser.add_argument("--derived", type=Path)
    args = parser.parse_args()
    report = acquire(args.config, args.raw, args.output, args.derived)
    print(
        json.dumps(
            {
                "status": report["status"],
                "query_count": report["query_count"],
                "total_candidate_rows": report["total_candidate_rows"],
                "published_bullet_photometry": report["published_bullet_photometry"],
                "by_cluster": report["by_cluster"],
                "counterpart_selection_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
