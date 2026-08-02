#!/usr/bin/env python3
"""Acquire the official Iorio online tables whose signed URL needs the OUP UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0710b_iorio_online_tables_acquisition.json"
OUTPUT = ROOT / "results" / "p0710b_iorio_online_tables_acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unlock = json.loads(
        (ROOT / "results/p0633_external_validation/unlock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    acquisition = json.loads(
        (ROOT / "results/p0710_external_target_acquisition/report.json").read_text(
            encoding="utf-8"
        )
    )
    if unlock["status"] != "authorized_for_exactly_one_external_parse":
        raise RuntimeError("P0709 unlock is missing")
    if acquisition["status"] != "pass" or not acquisition["P0633_sample_now_spent"]:
        raise RuntimeError("P0710 external acquisition is incomplete")
    destination = ROOT / config["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != config["expected_bytes"]:
        with (
            urllib.request.urlopen(config["resolved_url"], timeout=120) as response,
            tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle,
        ):
            temporary = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        try:
            if temporary.stat().st_size != config["expected_bytes"]:
                raise RuntimeError("official supplement byte count changed")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    with zipfile.ZipFile(destination) as archive:
        members = sorted(archive.namelist())
        if members != sorted(config["expected_members"]):
            raise RuntimeError(f"unexpected supplement members: {members}")
        member_hashes = {}
        for member in members:
            digest = hashlib.sha256(archive.read(member)).hexdigest()
            member_hashes[member] = digest
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0710B-IORIO-ONLINE-TABLES-ACQUISITION-RESULT-1.0.0",
        "status": "pass",
        "selection_changed_after_unlock": False,
        "source_resolution_changed_after_unlock": True,
        "article_url": config["article_url"],
        "canonical_cdn_path": config["canonical_cdn_path"],
        "destination": config["destination"],
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "members": members,
        "member_sha256": member_hashes,
        "P0633_sample_spent": True,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
