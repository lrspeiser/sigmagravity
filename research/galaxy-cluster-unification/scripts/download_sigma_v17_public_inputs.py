#!/usr/bin/env python3
"""Acquire and validate open v17 dynamical-stress inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17_public_data_acquisition.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17_public_data_acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def download(url: str, path: Path, expected_bytes: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected_bytes:
        return True
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            response.raw.decode_content = False
            shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{path.name}: expected {expected_bytes} bytes, got {partial.stat().st_size}"
        )
    partial.replace(path)
    return False


def selected_member_summary(path: Path) -> dict[str, int]:
    rows = data_rows(path)
    z_values = [float(line[31:39]) for line in rows]
    return {
        "selected_members": len(rows),
        "spectroscopic_members": sum(value > 0.0 for value in z_values),
        "photometric_members": sum(value <= 0.0 for value in z_values),
    }


def write_selected_spectroscopic_members(source: Path, destination: Path) -> dict[str, float]:
    records = []
    for line in data_rows(source):
        redshift = float(line[31:39])
        if redshift <= 0.0:
            continue
        records.append(
            {
                "catalog_id": int(line[0:4]),
                "ra_deg": float(line[5:16]),
                "dec_deg": float(line[17:28]),
                "instrument_code": line[29],
                "spectroscopic_redshift": redshift,
                "f160w_kron_mag": float(line[40:45]),
            }
        )
    median_redshift = float(np.median([row["spectroscopic_redshift"] for row in records]))
    speed_of_light_km_s = 299792.458
    for row in records:
        row["rest_frame_velocity_km_s"] = (
            speed_of_light_km_s
            * (row["spectroscopic_redshift"] - median_redshift)
            / (1.0 + median_redshift)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    velocities = np.array([row["rest_frame_velocity_km_s"] for row in records])
    return {
        "members": len(records),
        "median_redshift": median_redshift,
        "velocity_median_km_s": float(np.median(velocities)),
        "velocity_standard_deviation_km_s": float(np.std(velocities, ddof=1)),
    }


def full_spectroscopy_summary(path: Path) -> dict[str, int]:
    rows = data_rows(path)
    qualities = [int(line[40:43]) for line in rows]
    return {
        "spectroscopic_rows": len(rows),
        "secure_quality_3": qualities.count(3),
        "single_line_quality_9": qualities.count(9),
        "likely_quality_2": qualities.count(2),
        "uncertain_quality_1": qualities.count(1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen before downloading the PLCKG287 member spectroscopy":
        raise RuntimeError("the v17 acquisition protocol is not frozen")

    raw = ROOT / config["raw_directory"]
    records = []
    for item in config["files"]:
        path = raw / item["filename"]
        reused = download(
            f"{config['base_url']}/{item['filename']}",
            path,
            int(item["expected_bytes"]),
        )
        expected_records = item["expected_records"]
        if expected_records is not None and len(data_rows(path)) != int(expected_records):
            raise RuntimeError(f"{path.name}: record count changed")
        records.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "reused": reused,
            }
        )

    member_summary = selected_member_summary(raw / "tabled1.dat")
    spectroscopy_summary = full_spectroscopy_summary(raw / "tablee1.dat")
    if member_summary != {
        "selected_members": 153,
        "spectroscopic_members": 129,
        "photometric_members": 24,
    }:
        raise RuntimeError("selected-member composition changed")
    if spectroscopy_summary["spectroscopic_rows"] != 639:
        raise RuntimeError("full spectroscopy row count changed")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    velocity_path = output / "plckg287_selected_spectroscopic_members.csv"
    velocity_summary = write_selected_spectroscopic_members(
        raw / "tabled1.dat",
        velocity_path,
    )
    report = {
        "status": "open_PLCKG287_member_spectroscopy_downloaded_and_validated",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "files": records,
        "selected_member_summary": member_summary,
        "full_spectroscopy_summary": spectroscopy_summary,
        "velocity_table": {
            "relative_path": velocity_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(velocity_path),
            **velocity_summary,
            "interpretation": "commissioning observable only; no local kernel, mass weight, or gravity score has been selected",
        },
        "lensing_target_opened": False,
        "formula_selection_authorized": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
