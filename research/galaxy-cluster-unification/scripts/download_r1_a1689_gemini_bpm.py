#!/usr/bin/env python3
"""Download the pre-registered GMOS-N EEV BPM and record archive provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_gmos_reduction_covariance_protocol.json"
RAW = ROOT / "data/raw/r1_a1689_gemini"
PROVENANCE = RAW / "bpm_provenance.json"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bpm_config = config["raw_inputs"]["bad_pixel_mask"]
    expected = bpm_config["selected_filename"]
    table = Observations.query_criteria("BPM", "2x2", instrument="GMOS-N")
    candidates = []
    selected_row = None
    for row in table:
        record = {
            "name": str(row["name"]),
            "ut_datetime": str(row["ut_datetime"]),
            "instrument": str(row["instrument"]),
            "detector_binning": str(row["detector_binning"]),
            "file_size": int(row["file_size"]),
            "data_size": int(row["data_size"]),
            "archive_file_md5": str(row["file_md5"]).upper(),
            "archive_data_md5": str(row["data_md5"]).upper(),
        }
        candidates.append(record)
        if record["name"] == expected:
            selected_row = record
    if selected_row is None:
        raise RuntimeError(f"Frozen BPM is absent from current archive query: {expected}")

    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / expected
    if not target.exists():
        Observations.get_file(expected, download_dir=str(RAW), timeout=120000)
        if not target.exists():
            raise RuntimeError(f"Archive client did not create the expected BPM path: {target}")
    local_md5 = digest(target, "md5")
    if local_md5 not in {selected_row["archive_file_md5"], selected_row["archive_data_md5"]}:
        raise RuntimeError("Downloaded BPM matches neither archive file nor data MD5")
    provenance = {
        "provenance_version": "R1B1-A1689-GMOS-reduction-covariance-0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive_api": "astroquery.gemini.Observations query BPM/2x2/GMOS-N",
        "selection_frozen_before_preprocessing": True,
        "science_or_kinematic_values_inspected": False,
        "query_candidates": candidates,
        "selected": {
            **selected_row,
            "local_path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "local_size_bytes": target.stat().st_size,
            "local_md5": local_md5,
            "local_sha256": digest(target, "sha256"),
            "matched_archive_md5_kind": "file_md5" if local_md5 == selected_row["archive_file_md5"] else "data_md5",
        },
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
