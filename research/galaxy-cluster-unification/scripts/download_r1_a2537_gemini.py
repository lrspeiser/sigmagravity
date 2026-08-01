#!/usr/bin/env python3
"""Download only the A2537 files authorized by the frozen feasibility gate."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a2537_gemini_feasibility_protocol.json"
FEASIBILITY_PATH = ROOT / "results/r1_a2537_gemini_feasibility/report.json"
RAW = ROOT / "data/raw/r1_a2537_gemini"
PROVENANCE_PATH = RAW / "provenance.json"

META_FIELDS = [
    "name", "object", "program_id", "observation_id", "data_label", "ut_datetime",
    "observation_class", "observation_type", "mode", "exposure_time", "qa_state",
    "detector_binning", "detector_readspeed_setting", "detector_gain_setting",
    "detector_roi_setting", "disperser", "central_wavelength", "focal_plane_mask",
    "cass_rotator_pa", "file_size", "data_size", "file_md5", "data_md5",
]


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def scalar(value):
    if getattr(value, "mask", False) is True:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def download_group(names: list[str], rows: dict, category: str) -> list[dict]:
    missing = sorted(set(names) - set(rows))
    if missing:
        raise RuntimeError(f"Frozen {category} files disappeared from archive metadata: {missing}")
    records = []
    for name in names:
        row = rows[name]
        target = RAW / name
        if not target.exists():
            result = Observations.get_file(name, download_dir=str(RAW), timeout=120000)
            if result and Path(result).exists():
                target = Path(result)
        if not target.exists() or target.stat().st_size == 0:
            raise RuntimeError(f"Download did not produce {name}")
        local_md5 = digest(target, "md5")
        archive_md5s = {str(row["file_md5"]).upper(), str(row["data_md5"]).upper()}
        if local_md5 not in archive_md5s:
            raise RuntimeError(f"Archive MD5 mismatch for {name}")
        records.append({
            "category": category,
            "archive_name": name,
            "local_path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "local_size_bytes": target.stat().st_size,
            "local_md5": local_md5,
            "local_sha256": digest(target, "sha256"),
            "matched_archive_md5_kind": "file_md5" if local_md5 == str(row["file_md5"]).upper() else "data_md5",
            "metadata": {field: scalar(row[field]) for field in META_FIELDS},
        })
    return records


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feasibility = json.loads(FEASIBILITY_PATH.read_text(encoding="utf-8"))
    if not feasibility["authorization"]["download_exact_frozen_raw_files"]:
        raise RuntimeError("A2537 metadata feasibility gate did not authorize acquisition")

    RAW.mkdir(parents=True, exist_ok=True)
    science = config["science_selection"]["science_filenames"]
    flats = config["calibration_selection"]["exact_flat_download"]
    arcs = config["calibration_selection"]["exact_arc_download"]
    biases = config["calibration_selection"]["exact_bias_download"]
    bpm = [config["calibration_selection"]["required_bpm"]]

    program_table = Observations.query_criteria(program_id=config["program_id"], raw_reduced="RAW")
    program_rows = {str(row["name"]): row for row in program_table}
    bias_rows = {}
    for bias_date in (date(2008, 9, 21), date(2008, 9, 22)):
        bias_table = Observations.query_criteria(
            utc_date=(bias_date, bias_date),
            instrument="GMOS-S",
            observation_type="BIAS",
            raw_reduced="RAW",
        )
        bias_rows.update({str(row["name"]): row for row in bias_table})
    bpm_table = Observations.query_criteria("BPM", "2x2", instrument="GMOS-S")
    bpm_rows = {str(row["name"]): row for row in bpm_table}

    records = [
        *download_group(science, program_rows, "science"),
        *download_group(flats, program_rows, "flat"),
        *download_group(arcs, program_rows, "arc"),
        *download_group(biases, bias_rows, "bias"),
        *download_group(bpm, bpm_rows, "bpm"),
    ]
    provenance = {
        "provenance_version": config["protocol_version"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "archive_api": "astroquery.gemini.Observations over https://archive.gemini.edu/",
        "selection_frozen_before_download": True,
        "science_pixels_inspected": False,
        "records": records,
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "downloaded_files": len(records),
        "downloaded_bytes": sum(record["local_size_bytes"] for record in records),
        "categories": {category: sum(record["category"] == category for record in records) for category in ("science", "flat", "arc", "bias", "bpm")},
        "provenance": str(PROVENANCE_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
