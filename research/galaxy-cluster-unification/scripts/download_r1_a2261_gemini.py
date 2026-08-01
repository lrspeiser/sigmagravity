#!/usr/bin/env python3
"""Download the frozen A2261 target-associated GMOS files and biases."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a2261_gemini_acquisition_protocol.json"
RAW = ROOT / "data/raw/r1_a2261_gemini"


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
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def acquire(expected: list[str], rows: dict, provenance_name: str, selection: str) -> list[dict]:
    missing = sorted(set(expected) - set(rows))
    if missing:
        raise RuntimeError(f"Frozen archive files missing from metadata: {missing}")
    records = []
    metadata_fields = [
        "name", "object", "program_id", "observation_id", "data_label", "ut_datetime",
        "observation_class", "observation_type", "mode", "exposure_time", "cass_rotator_pa",
        "qa_state", "detector_binning", "detector_readspeed_setting", "detector_gain_setting",
        "detector_roi_setting", "disperser", "central_wavelength", "focal_plane_mask",
        "filter_name", "file_size", "data_size", "file_md5", "data_md5",
    ]
    for name in expected:
        row = rows[name]
        path = RAW / name
        if not path.exists():
            result = Observations.get_file(name, download_dir=str(RAW), timeout=120000)
            if result and Path(result).exists():
                path = Path(result)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Download did not produce {name}")
        local_md5 = digest(path, "md5")
        archive_md5s = {str(row["file_md5"]).upper(), str(row["data_md5"]).upper()}
        if local_md5 not in archive_md5s:
            raise RuntimeError(f"Archive MD5 mismatch for {name}")
        records.append(
            {
                "archive_name": name,
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "local_size_bytes": path.stat().st_size,
                "local_sha256": digest(path, "sha256"),
                "local_md5": local_md5,
                "metadata": {field: scalar(row[field]) for field in metadata_fields},
            }
        )
    provenance = {
        "provenance_version": json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["protocol_version"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "archive_api": "astroquery.gemini.Observations over https://archive.gemini.edu/",
        "selection": selection,
        "records": records,
    }
    (RAW / provenance_name).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    program_archive = Observations.query_criteria(program_id=config["program_id"], raw_reduced="RAW")
    program_rows = {str(row["name"]): row for row in program_archive}
    program = acquire(
        config["exact_target_associated_download"],
        program_rows,
        "provenance.json",
        "Exact pre-frozen A2261 target-associated acquisition, science, flat, and arc files",
    )

    bias_archive = Observations.query_criteria(
        utc_date=(date(2008, 3, 15), date(2008, 3, 16)),
        instrument="GMOS-N",
        observation_type="BIAS",
        raw_reduced="RAW",
    )
    bias_rows = {str(row["name"]): row for row in bias_archive}
    required = {
        "qa_state": "Pass",
        "detector_binning": "2x2",
        "detector_readspeed_setting": "slow",
        "detector_gain_setting": "low",
        "detector_roi_setting": "Full Frame",
    }
    for name in config["calibration_gate"]["exact_bias_download"]:
        for field, expected in required.items():
            if str(bias_rows[name][field]) != expected:
                raise RuntimeError(f"Frozen bias metadata mismatch for {name}: {field}")
    biases = acquire(
        config["calibration_gate"]["exact_bias_download"],
        bias_rows,
        "bias_provenance.json",
        config["calibration_gate"]["bias_selection"],
    )
    print(json.dumps({"target_associated_files": len(program), "bias_files": len(biases), "output": str(RAW)}, indent=2))


if __name__ == "__main__":
    main()
