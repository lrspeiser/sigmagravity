#!/usr/bin/env python3
"""Download the frozen A1689 GN-2008B-Q-5 program-associated raw manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from astroquery.gemini import Observations


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_a1689_gemini_acquisition_protocol.json"
DEFAULT_OUTPUT = ROOT / "data/raw/r1_a1689_gemini"


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


def find_download(output: Path, name: str) -> Path | None:
    matches = [path for path in output.glob(f"{name}*") if path.is_file()]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous local files for {name}: {matches}")
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = (ROOT / "data/raw").resolve()
    if allowed not in output.parents:
        raise RuntimeError(f"Output must remain inside {allowed}")
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = config["exact_program_associated_download"]
    archive = Observations.query_criteria(program_id=config["program_id"], raw_reduced="RAW")
    rows = {str(row["name"]): row for row in archive}
    missing_metadata = sorted(set(expected) - set(rows))
    if missing_metadata:
        raise RuntimeError(f"Frozen archive files missing from current metadata query: {missing_metadata}")

    records = []
    metadata_fields = [
        "name", "object", "program_id", "observation_id", "data_label", "ut_datetime",
        "observation_class", "observation_type", "mode", "exposure_time", "cass_rotator_pa",
        "qa_state", "detector_binning", "detector_readspeed_setting", "detector_gain_setting",
        "detector_roi_setting", "disperser", "central_wavelength", "focal_plane_mask", "filter_name",
        "file_size", "data_size", "file_md5", "data_md5",
    ]
    for name in expected:
        path = find_download(output, name)
        if path is None:
            result = Observations.get_file(name, download_dir=str(output), timeout=120000)
            if result:
                candidate = Path(result)
                if candidate.exists():
                    path = candidate.resolve()
            if path is None:
                path = find_download(output, name)
        if path is None or not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Download did not produce a local file for {name}")

        row = rows[name]
        local_md5 = digest(path, "md5")
        archive_file_md5 = str(row["file_md5"]).upper()
        archive_data_md5 = str(row["data_md5"]).upper()
        md5_kind = None
        if local_md5 == archive_file_md5:
            md5_kind = "archive_file_md5"
        elif local_md5 == archive_data_md5:
            md5_kind = "archive_uncompressed_data_md5"
        else:
            raise RuntimeError(f"Archive MD5 mismatch for {path}")
        records.append(
            {
                "archive_name": name,
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "local_size_bytes": path.stat().st_size,
                "local_sha256": digest(path, "sha256"),
                "local_md5": local_md5,
                "matched_archive_md5_kind": md5_kind,
                "metadata": {field: scalar(row[field]) for field in metadata_fields},
            }
        )

    provenance = {
        "provenance_version": config["protocol_version"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "archive_api": "astroquery.gemini.Observations over https://archive.gemini.edu/",
        "program_id": config["program_id"],
        "records": records,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
