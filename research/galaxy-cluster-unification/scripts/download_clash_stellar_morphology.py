#!/usr/bin/env python3
"""Download and hash the frozen four-cluster CLASH F160W package."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_bytes: int) -> None:
    if destination.exists() and destination.stat().st_size == expected_bytes:
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "sigmagravity-data-audit/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"byte count mismatch for {url}: {partial.stat().st_size} != {expected_bytes}"
        )
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/clash_stellar_morphology_acquisition_protocol.json",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_f160w_download_or_pixel_inspection":
        raise RuntimeError("acquisition protocol is not frozen")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for system in protocol["systems"]:
        for kind in ("science", "weight"):
            item = system[kind]
            destination = output / item["filename"]
            jobs.append((system, kind, item, destination))

    def acquire(job):
        system, kind, item, destination = job
        print(f"download {system['label']} {kind}", flush=True)
        download(item["url"], destination, int(item["content_length"]))
        return {
            "label": system["label"],
            "kind": kind,
            "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
            "url": item["url"],
            "declared_etag": item["etag"],
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(acquire, jobs))

    provenance = {
        "report_version": "CLASH-STELLAR-MORPHOLOGY-PROVENANCE-0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(config_path),
        "pixel_data_inspected_by_downloader": False,
        "files": records,
    }
    provenance_path = ROOT / protocol["outputs"]["provenance"]
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
