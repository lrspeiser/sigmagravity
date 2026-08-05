#!/usr/bin/env python3
"""Validate v19E ancestry, then run the shared Chandra archive downloader."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19e_chandra_acquisition.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19e_chandra_acquisition"
SHARED_DOWNLOADER = ROOT / "scripts" / "download_sigma_v17a_chandra_inputs.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != (
        "frozen before downloading any analysis-grade event or calibration product"
    ):
        raise RuntimeError("v19E Chandra acquisition protocol is not frozen")
    for key in ("member_extraction_config", "member_extraction_report"):
        path = ROOT / config["parents"][key]
        if sha256(path) != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
    if set(config["clusters"]) != {"BULLET", "ABELL2146"}:
        raise RuntimeError("v19E must retain the selected development pair")
    if not any("lensing" in item.lower() for item in config["exclusions"]):
        raise RuntimeError("v19E does not explicitly exclude lensing payloads")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    validate(config_path)
    subprocess.run(
        [
            sys.executable,
            str(SHARED_DOWNLOADER),
            "--config",
            str(config_path),
            "--output",
            str(args.output.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
