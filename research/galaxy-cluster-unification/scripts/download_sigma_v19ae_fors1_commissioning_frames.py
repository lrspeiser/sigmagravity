#!/usr/bin/env python3
"""Acquire frozen ESO FORS1 commissioning science and calibration frames.

V19AE is lossless acquisition only.  It validates exact archive metadata and
downloads the original compressed files without opening or interpreting any
FITS pixel or header payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ae_fors1_commissioning_frames.json"
USER_AGENT = "SigmaGravity-V19AE-FORS1-acquisition/1.0"

QUERY_COLUMNS = [
    "dp_id",
    "date_obs",
    "ra",
    "dec",
    "exposure",
    "filter_path",
    "dp_cat",
    "dp_type",
    "dp_tech",
    "prog_id",
    "ob_id",
    "ob_name",
    "object",
    "origfile",
    "tpl_id",
    "tpl_name",
    "access_estsize",
    "access_url",
    "datalink_url",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_dataset_filename(dp_id: str) -> str:
    """Encode an ESO identifier as one portable filename component."""
    encoded = re.sub(r"[^A-Za-z0-9_.-]+", "_", dp_id)
    if not encoded or encoded in {".", ".."}:
        raise RuntimeError(f"unsafe ESO dataset identifier: {dp_id!r}")
    return encoded


def build_queries() -> dict[str, str]:
    columns = ",".join(QUERY_COLUMNS)
    return {
        "science": (
            f"SELECT {columns} FROM dbo.raw "
            "WHERE ra BETWEEN 104.50 AND 104.75 "
            "AND dec BETWEEN -56.05 AND -55.85 "
            "AND instrument='FORS1' "
            "AND date_obs BETWEEN '1998-12-01' AND '1999-01-01' "
            "AND dp_cat='SCIENCE' AND dp_tech='IMAGE' "
            "AND filter_path IN ('B_BESS','R_BESS','I_BESS') "
            "ORDER BY dp_id"
        ),
        "bias": (
            f"SELECT {columns} FROM dbo.raw "
            "WHERE instrument='FORS1' "
            "AND date_obs BETWEEN '1998-12-12' AND '1998-12-16' "
            "AND dp_cat='CALIB' AND dp_tech='IMAGE' AND dp_type='BIAS' "
            "AND ob_name='ALL-BIAS_fl1x1_10' "
            "ORDER BY dp_id"
        ),
        "flat": (
            f"SELECT {columns} FROM dbo.raw "
            "WHERE instrument='FORS1' "
            "AND date_obs BETWEEN '1998-12-22' AND '1998-12-27' "
            "AND dp_cat='CALIB' AND dp_tech='IMAGE' AND dp_type='CALIB' "
            "AND tpl_id='FORS1_img_cal_Twili' "
            "AND filter_path IN ('B_BESS','R_BESS','I_BESS') "
            "ORDER BY dp_id"
        ),
    }


def parse_metadata(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != QUERY_COLUMNS:
        raise RuntimeError(f"V19AE archive schema changed: {reader.fieldnames}")
    rows = list(reader)
    ids = [row["dp_id"] for row in rows]
    if not all(value.startswith("OFORS.") for value in ids):
        raise RuntimeError("V19AE archive response contains a non-FORS identifier")
    if len(ids) != len(set(ids)):
        raise RuntimeError("V19AE archive response repeats a dataset identifier")
    if ids != sorted(ids):
        raise RuntimeError("V19AE archive response is not deterministically ordered")
    return rows


def request_bytes(
    url: str,
    *,
    data: bytes | None,
    timeout: float,
    attempts: int,
) -> tuple[int, bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                headers = {key.lower(): value for key, value in response.headers.items()}
                return int(response.status), response.read(), headers
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(float(attempt))
    raise RuntimeError(f"V19AE request failed after {attempts} attempts") from last_error


def query_archive(
    config: dict[str, Any], query: str
) -> tuple[int, bytes, bytes]:
    source = config["source"]
    form = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        }
    ).encode("utf-8")
    status, payload, _ = request_bytes(
        source["tap_endpoint"],
        data=form,
        timeout=float(source["timeout_seconds"]),
        attempts=int(source["maximum_attempts"]),
    )
    return status, payload, form


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


def download_file(
    url: str,
    path: Path,
    *,
    timeout: float,
    attempts: int,
) -> tuple[int, int | None, str | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if partial.exists():
            partial.unlink()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                expected = response.headers.get("Content-Length")
                disposition = response.headers.get("Content-Disposition")
                with partial.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            size = partial.stat().st_size
            expected_size = int(expected) if expected else None
            if status != 200:
                raise RuntimeError(f"download returned HTTP {status}")
            if expected_size is not None and size != expected_size:
                raise RuntimeError(
                    f"download length mismatch: expected {expected_size}, got {size}"
                )
            partial.replace(path)
            return status, expected_size, disposition
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt == attempts:
                break
            time.sleep(float(attempt))
    raise RuntimeError(f"V19AE download failed after {attempts} attempts: {url}") from last_error


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if config["status"] != "frozen_before_any_fits_payload_download_or_inspection":
        raise RuntimeError("V19AE protocol is not frozen")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AE runner hash mismatch")
    hashes = {"config": sha256(config_path), "runner": sha256(runner)}
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"V19AE parent hash mismatch: {artifact['path']}")
        hashes[artifact["path"]] = actual
    authorization = config["authorization"]
    prohibited = (
        "open_or_parse_fits_payload",
        "select_science_exposure_from_pixels",
        "fit_photometry_or_counterparts",
        "infer_stellar_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics",
        "open_holdout",
    )
    if any(authorization[name] for name in prohibited):
        raise RuntimeError("V19AE authorizes a prohibited downstream action")
    expected = config["expected_dataset_ids"]
    combined = [value for values in expected.values() for value in values]
    if len(combined) != len(set(combined)):
        raise RuntimeError("V19AE expected dataset IDs overlap between roles")
    return hashes


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes = validate_config(config_path, config)
    queries = build_queries()
    if set(queries) != set(config["expected_dataset_ids"]):
        raise RuntimeError("V19AE query roles differ from frozen expected roles")
    raw_root = ROOT / config["outputs"]["raw_root"]
    metadata_root = raw_root / "metadata"
    file_root = raw_root / "files"
    report_path = ROOT / config["outputs"]["report"]
    prior: dict[str, Any] = {}
    if report_path.exists():
        prior_report = load_json(report_path)
        prior = {row["dp_id"]: row for row in prior_report["files"]}

    all_rows: list[dict[str, str]] = []
    metadata_records: list[dict[str, Any]] = []
    for role in sorted(queries):
        query = queries[role]
        status, payload, form = query_archive(config, query)
        if status != 200:
            raise RuntimeError(f"V19AE metadata query {role} returned HTTP {status}")
        rows = parse_metadata(payload)
        returned = [row["dp_id"] for row in rows]
        expected = sorted(config["expected_dataset_ids"][role])
        if returned != expected:
            raise RuntimeError(
                f"V19AE {role} dataset inventory changed: {returned!r} != {expected!r}"
            )
        csv_path = metadata_root / f"{role}.csv"
        query_path = metadata_root / f"{role}.adql"
        form_path = metadata_root / f"{role}.form.txt"
        write_atomic(csv_path, payload)
        write_atomic(query_path, (query + "\n").encode("utf-8"))
        write_atomic(form_path, form)
        metadata_records.append(
            {
                "role": role,
                "rows": len(rows),
                "csv_path": csv_path.relative_to(ROOT).as_posix(),
                "csv_sha256": sha256(csv_path),
                "query_path": query_path.relative_to(ROOT).as_posix(),
                "query_sha256": sha256(query_path),
                "form_path": form_path.relative_to(ROOT).as_posix(),
                "form_sha256": sha256(form_path),
                "http_status": status,
            }
        )
        for row in rows:
            all_rows.append({"role": role, **row})

    file_records: list[dict[str, Any]] = []
    timeout = float(config["source"]["download_timeout_seconds"])
    attempts = int(config["source"]["maximum_attempts"])
    for index, row in enumerate(all_rows, start=1):
        dp_id = row["dp_id"]
        path = file_root / f"{safe_dataset_filename(dp_id)}.fits.Z"
        reused = False
        if path.exists():
            record = prior.get(dp_id)
            if record is None:
                raise RuntimeError(f"unmanifested V19AE file already exists: {path}")
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"cached V19AE file hash changed: {dp_id}")
            status = int(record["http_status"])
            content_length = record["content_length_header"]
            disposition = record["content_disposition"]
            reused = True
        else:
            status, content_length, disposition = download_file(
                row["access_url"], path, timeout=timeout, attempts=attempts
            )
        record = {
            **row,
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "http_status": status,
            "content_length_header": content_length,
            "content_disposition": disposition,
            "reused_verified_payload": reused,
        }
        file_records.append(record)
        print(
            f"[{index:02d}/{len(all_rows):02d}] {row['role']} {dp_id}: "
            f"{record['bytes']} bytes ({'verified' if reused else 'downloaded'})",
            flush=True,
        )

    counts = {
        role: sum(row["role"] == role for row in file_records) for role in queries
    }
    filters: dict[str, int] = {}
    for row in file_records:
        key = row["filter_path"] or "NONE"
        filters[key] = filters.get(key, 0) + 1
    gates = {
        "exact_frozen_dataset_ids": all(
            sorted(row["dp_id"] for row in file_records if row["role"] == role)
            == sorted(config["expected_dataset_ids"][role])
            for role in queries
        ),
        "all_metadata_queries_http_200": all(
            row["http_status"] == 200 for row in metadata_records
        ),
        "all_downloads_http_200": all(row["http_status"] == 200 for row in file_records),
        "all_download_lengths_match_headers_when_present": all(
            row["content_length_header"] is None
            or row["bytes"] == row["content_length_header"]
            for row in file_records
        ),
        "all_payloads_hashed": all(len(row["sha256"]) == 64 for row in file_records),
        "no_fits_payload_opened_or_parsed": True,
        "no_photometry_counterpart_mass_lensing_or_gravity": True,
    }
    gates["all_acquisition_gates_pass"] = all(gates.values())
    report = {
        "report_version": "SIGMA-V19AE-FORS1-COMMISSIONING-FRAMES-1.0.0",
        "status": "completed_lossless_unopened_fors1_acquisition"
        if gates["all_acquisition_gates_pass"]
        else "failed_acquisition_gate",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "input_hashes": input_hashes,
        "metadata": metadata_records,
        "counts": counts,
        "filters": dict(sorted(filters.items())),
        "total_files": len(file_records),
        "total_bytes": sum(row["bytes"] for row in file_records),
        "files": file_records,
        "gates": gates,
        "claim_boundary": config["claim_boundary"],
        "fits_payload_opened_or_parsed": False,
        "science_exposure_selected_from_pixels": False,
        "photometry_or_counterpart_fitted": False,
        "stellar_mass_or_current_inferred": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "total_files": report["total_files"],
                "total_bytes": report["total_bytes"],
                "gates": report["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
