#!/usr/bin/env python3
"""Acquire all NSC per-exposure measurements for retained V19AA candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ac_nsc_measurement_photometry.json"
USER_AGENT = "SigmaGravity-V19AC-NSC-measurement-acquisition/1.0"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    return value


def validate_config(config: dict[str, Any]) -> list[str]:
    if config["status"] != "frozen_before_querying_per_measurement_photometry_for_all_candidates":
        raise RuntimeError("V19AC protocol is not frozen")
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"parent artifact hash mismatch: {path}")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("frozen V19AC runner hash mismatch")
    authorization = config["authorization"]
    prohibited = (
        "select_aperture_or_exposure",
        "score_photometric_identity",
        "select_counterpart",
        "infer_stellar_mass",
        "construct_mass_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics",
        "open_holdout",
    )
    if any(authorization[name] for name in prohibited):
        raise RuntimeError("V19AC authorizes a prohibited downstream action")
    rows = read_csv_rows(ROOT / config["inputs"]["unified_candidates"])
    column = config["inputs"]["object_id_column"]
    object_ids = sorted({row[column].strip() for row in rows if row[column].strip()})
    if len(object_ids) != int(config["source"]["expected_unique_object_ids"]):
        raise RuntimeError(f"V19AC object-ID count changed: {len(object_ids)}")
    if any(not SAFE_ID.fullmatch(value) for value in object_ids):
        raise RuntimeError("unsafe NSC object ID")
    return object_ids


def make_batches(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def build_query(object_ids: list[str]) -> str:
    if not object_ids or any(not SAFE_ID.fullmatch(value) for value in object_ids):
        raise ValueError("query requires safe nonempty object IDs")
    values = ",".join(f"'{value}'" for value in object_ids)
    return (
        "SELECT "
        "m.objectid AS objectid,m.measid AS measid,m.exposure AS exposure,"
        "e.instrument AS instrument,m.filter AS filter,m.mjd AS mjd,"
        "m.mag_auto AS mag_auto,m.magerr_auto AS magerr_auto,"
        "m.mag_aper1 AS mag_aper1,m.magerr_aper1 AS magerr_aper1,"
        "m.mag_aper2 AS mag_aper2,m.magerr_aper2 AS magerr_aper2,"
        "m.mag_aper4 AS mag_aper4,m.magerr_aper4 AS magerr_aper4,"
        "m.mag_aper8 AS mag_aper8,m.magerr_aper8 AS magerr_aper8,"
        "m.flags AS flags,m.class_star AS class_star,m.kron_radius AS kron_radius,"
        "m.asemi AS asemi,m.bsemi AS bsemi,e.fwhm AS exposure_fwhm,"
        "e.zptermerr AS zptermerr,e.zptermsig AS zptermsig "
        "FROM nsc_dr2.meas AS m JOIN nsc_dr2.exposure AS e ON m.exposure=e.exposure "
        f"WHERE m.objectid IN ({values}) "
        "ORDER BY m.objectid,m.filter,m.mjd,m.measid"
    )


def request_batch(config: dict[str, Any], query: str) -> tuple[int, bytes, bytes]:
    source = config["source"]
    form = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": source["format"], "QUERY": query}
    ).encode("utf-8")
    request = urllib.request.Request(
        source["endpoint"],
        data=form,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )
    attempts = int(source["maximum_attempts"])
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=float(source["timeout_seconds"])) as response:
                return int(response.status), response.read(), form
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def parse_payload(payload: bytes, expected_columns: list[str]) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != expected_columns:
        raise RuntimeError(f"V19AC response schema mismatch: {reader.fieldnames}")
    return list(reader)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    object_ids = validate_config(config)
    source = config["source"]
    columns = list(source["query_columns"])
    batches = make_batches(object_ids, int(source["batch_size"]))
    raw_root = ROOT / config["outputs"]["raw_root"]
    raw_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        query = build_query(batch)
        status, payload, form = request_batch(config, query)
        if status != 200:
            raise RuntimeError(f"V19AC HTTP status {status} for batch {index}")
        rows = parse_payload(payload, columns)
        returned = {row["objectid"] for row in rows}
        if not returned.issubset(batch):
            raise RuntimeError(f"V19AC batch {index} returned unrequested object IDs")
        csv_path = raw_root / f"batch_{index:03d}.csv"
        query_path = raw_root / f"batch_{index:03d}.adql"
        form_path = raw_root / f"batch_{index:03d}.form.txt"
        csv_path.write_bytes(payload)
        query_path.write_text(query + "\n", encoding="utf-8")
        form_path.write_bytes(form)
        records.append(
            {
                "batch": index,
                "requested_object_ids": batch,
                "returned_object_ids": sorted(returned),
                "measurement_rows": len(rows),
                "csv_path": csv_path.relative_to(ROOT).as_posix(),
                "csv_sha256": sha256(csv_path),
                "query_path": query_path.relative_to(ROOT).as_posix(),
                "query_sha256": sha256(query_path),
                "form_path": form_path.relative_to(ROOT).as_posix(),
                "form_sha256": sha256(form_path),
                "http_status": status,
            }
        )
        all_rows.extend(rows)

    combined_path = ROOT / config["outputs"]["combined_measurements"]
    write_csv(combined_path, all_rows, columns)
    returned_all = {row["objectid"] for row in all_rows}
    instruments: dict[str, int] = {}
    filters: dict[str, int] = {}
    for row in all_rows:
        instruments[row["instrument"]] = instruments.get(row["instrument"], 0) + 1
        filters[row["filter"]] = filters.get(row["filter"], 0) + 1
    gates = {
        "exact_unique_object_ids": len(object_ids) == int(config["gates"]["exact_unique_object_ids"]),
        "http_200_every_batch": all(record["http_status"] == 200 for record in records),
        "exact_schema_every_batch": True,
        "no_unrequested_object_ids": returned_all.issubset(object_ids),
        "every_requested_object_has_at_least_one_measurement": returned_all == set(object_ids),
        "all_rows_retained_without_quality_or_aperture_selection": True,
        "every_raw_payload_query_and_request_form_hashed": True,
    }
    gates["all_acquisition_gates_pass"] = all(gates.values())
    report_path = ROOT / config["outputs"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "SIGMA-V19AC-NSC-MEASUREMENT-PHOTOMETRY-1.0.0",
        "status": "completed_lossless_measurement_acquisition",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "implementation": config["implementation"],
        "parent_hashes": {artifact["path"]: artifact["sha256"] for artifact in config["parent_artifacts"]},
        "requested_object_ids": len(object_ids),
        "returned_object_ids": len(returned_all),
        "measurement_rows": len(all_rows),
        "instruments": dict(sorted(instruments.items())),
        "filters": dict(sorted(filters.items())),
        "records": records,
        "gates": gates,
        "outputs": {
            "combined_measurements": combined_path.relative_to(ROOT).as_posix(),
            "combined_measurements_sha256": sha256(combined_path),
        },
        "claim_boundary": config["claim_boundary"],
        "aperture_or_exposure_selected": False,
        "photometric_identity_scored": False,
        "counterpart_selected": False,
        "stellar_mass_inferred": False,
        "mass_current_constructed": False,
        "lensing_or_halo_payload_opened": False,
        "gravity_formula_or_parameter_changed": False,
    }
    report_path.write_text(json.dumps(strict_json(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(strict_json(run(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
