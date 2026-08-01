#!/usr/bin/env python3
"""Download the frozen RELICS galaxy catalogs for gravity-arc tomography."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/gravity_arc_tomography_acquisition.json"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_catalog_download_or_map_correlation":
        raise RuntimeError("acquisition protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for system in protocol["systems"]:
        destination = output / system["catalog_filename"]
        if not destination.exists() or destination.stat().st_size != system["expected_bytes"]:
            request = urllib.request.Request(
                system["catalog_url"], headers={"User-Agent": "gravity-arc-tomography/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if len(payload) != int(system["expected_bytes"]):
                raise RuntimeError(
                    f"{system['label']}: expected {system['expected_bytes']} bytes, got {len(payload)}"
                )
            destination.write_bytes(payload)
        records.append(
            {
                "label": system["label"],
                "url": system["catalog_url"],
                "path": str(destination.relative_to(ROOT)).replace("\\", "/"),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "expected_etag": system["etag"],
            }
        )
        print(f"verified {system['label']}: {destination.name}", flush=True)
    provenance = {
        "protocol_version": protocol["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(config_path),
        "files": records,
    }
    provenance_path = ROOT / protocol["outputs"]["provenance"]
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
