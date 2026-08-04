#!/usr/bin/env python3
"""Download and hash the frozen Gaia DR3 astrometric reference cones."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a_gaia_astrometry.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a_gaia_astrometry_acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adql(config: dict, cluster: dict) -> str:
    archive = config["archive"]
    columns = ",".join(archive["columns"])
    return (
        f"SELECT {columns} FROM {archive['table']} "
        "WHERE 1=CONTAINS("
        "POINT('ICRS',ra,dec),"
        f"CIRCLE('ICRS',{cluster['center_ra_deg']},{cluster['center_dec_deg']},"
        f"{archive['cone_radius_deg']})) "
        "ORDER BY source_id"
    )


def validate_csv(content: bytes, expected_columns: list[str]) -> int:
    text = content.decode("utf-8-sig")
    rows = csv.DictReader(io.StringIO(text))
    if rows.fieldnames != expected_columns:
        raise RuntimeError(
            f"unexpected Gaia columns: {rows.fieldnames}; expected {expected_columns}"
        )
    count = 0
    previous = -1
    for row in rows:
        source_id = int(row["source_id"])
        if source_id <= previous:
            raise RuntimeError("Gaia rows are not strictly source_id ordered")
        previous = source_id
        ra = float(row["ra"])
        dec = float(row["dec"])
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            raise RuntimeError(f"invalid Gaia coordinate for source {source_id}")
        count += 1
    if count == 0:
        raise RuntimeError("Gaia cone query returned no rows")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_status = (
        "frozen metadata correction after the wrong-field Gaia cone exposed the preregistered "
        "PLCK center transcription error, before querying the corrected PLCK cone, matching "
        "any PLCK X-ray source, or constructing any temperature map; the unchanged AS295 "
        "match outcomes are already known"
    )
    if config["status"] != expected_status:
        raise RuntimeError("Gaia astrometry protocol is not at its frozen acquisition state")

    raw = ROOT / config["archive"]["raw_directory"]
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name, cluster in config["clusters"].items():
        query = adql(config, cluster)
        response = requests.post(
            config["archive"]["tap_sync_url"],
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": config["archive"]["format"],
                "QUERY": query,
            },
            timeout=(30, 300),
        )
        response.raise_for_status()
        count = validate_csv(response.content, config["archive"]["columns"])
        path = raw / f"{name}_gaia_dr3.csv"
        if path.exists() and path.read_bytes() != response.content:
            raise RuntimeError(
                f"existing immutable Gaia response differs from current TAP response: {path}"
            )
        if not path.exists():
            path.write_bytes(response.content)
        records.append(
            {
                "cluster": name,
                "query": query,
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "relative_path": path.relative_to(ROOT).as_posix(),
                "rows": count,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        print(f"{name}: {count} Gaia DR3 sources", flush=True)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "frozen_Gaia_DR3_reference_cones_downloaded_and_hashed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "records": records,
        "files": len(records),
        "rows": sum(row["rows"] for row in records),
        "bytes": sum(row["bytes"] for row in records),
        "xray_source_crossmatch_run": False,
        "astrometric_offset_fit": False,
        "registered_event_image_inspected": False,
        "lensing_target_opened": False,
        "temperature_map_constructed": False,
    }
    report_path = output / "provenance.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
