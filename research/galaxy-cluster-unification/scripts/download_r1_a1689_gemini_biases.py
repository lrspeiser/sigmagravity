#!/usr/bin/env python3
"""Download the ten frozen same-date A1689 GMOS-N bias frames."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a1689_gemini_acquisition_protocol.json"
RAW = ROOT / "data/raw/r1_a1689_gemini"
PROVENANCE_PATH = RAW / "bias_provenance.json"


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


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = config["calibration_gate"]["exact_bias_download"]
    RAW.mkdir(parents=True, exist_ok=True)
    archive = Observations.query_criteria(
        utc_date=(date(2009, 6, 15), date(2009, 6, 21)),
        instrument="GMOS-N",
        observation_type="BIAS",
        raw_reduced="RAW",
    )
    rows = {str(row["name"]): row for row in archive}
    if sorted(set(expected) - set(rows)):
        raise RuntimeError("One or more frozen bias files are absent from the archive query")

    records = []
    metadata_fields = [
        "name", "program_id", "ut_datetime", "observation_class", "observation_type",
        "exposure_time", "qa_state", "detector_binning", "detector_readspeed_setting",
        "detector_gain_setting", "detector_roi_setting", "file_size", "data_size",
        "file_md5", "data_md5",
    ]
    for name in expected:
        row = rows[name]
        required = {
            "qa_state": "Pass",
            "detector_binning": "2x2",
            "detector_readspeed_setting": "slow",
            "detector_gain_setting": "low",
            "detector_roi_setting": "Full Frame",
        }
        for field, value in required.items():
            if str(row[field]) != value:
                raise RuntimeError(f"Frozen bias metadata mismatch for {name}: {field}")
        path = RAW / name
        if not path.exists():
            result = Observations.get_file(name, download_dir=str(RAW), timeout=120000)
            if result and Path(result).exists():
                path = Path(result)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing local bias {name}")
        local_md5 = digest(path, "md5")
        archive_file_md5 = str(row["file_md5"]).upper()
        archive_data_md5 = str(row["data_md5"]).upper()
        if local_md5 not in {archive_file_md5, archive_data_md5}:
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
        "provenance_version": config["protocol_version"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "five same-date 2x2 slow/low full-frame QA-pass biases for each science night",
        "records": records,
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bias_files": len(records), "output": str(RAW)}, indent=2))


if __name__ == "__main__":
    main()
