#!/usr/bin/env python3
"""Download the exact frozen P0566 HST and Chandra products."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0566_a383_ms2137_transfer_acquisition_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def download(product, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(product["content_length"])
    if destination.exists() and destination.stat().st_size != expected:
        destination.unlink()
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        with requests.get(product["url"], stream=True, timeout=(60, 300)) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with partial.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle, length=8 * 1024 * 1024)
        partial.replace(destination)
    size = destination.stat().st_size
    if size != expected:
        raise RuntimeError(
            f"Length mismatch for {destination}: expected {expected}, found {size}"
        )
    return {
        "size_bytes": size,
        "sha256": sha256(destination),
    }


def main():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_after_metadata_feasibility"):
        raise RuntimeError("P0566 acquisition was not frozen before pixel download")
    feasibility = json.loads(
        (ROOT / protocol["inputs"]["feasibility_report"]).read_text(encoding="utf-8")
    )
    if not feasibility["gate_audit"]["feasibility_passed"]:
        raise RuntimeError("P0566 feasibility gate did not authorize acquisition")

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    resolved_output = output.resolve()
    records = []
    for system in protocol["systems"]:
        label = system["label"]
        for product in system["hst"]:
            destination = output / "hst" / label / product["filename"]
            if resolved_output not in destination.resolve().parents:
                raise RuntimeError(f"Refusing output outside acquisition root: {destination}")
            audit = download(product, destination)
            records.append(
                {
                    "system_label": label,
                    "kind": product["kind"],
                    "obsid": None,
                    "url": product["url"],
                    "local_path": destination.relative_to(ROOT).as_posix(),
                    **audit,
                }
            )
            print(label, product["kind"], destination.name, audit["size_bytes"], flush=True)
        for product in system["chandra"]:
            destination = output / "chandra" / label / str(product["obsid"]) / product["filename"]
            if resolved_output not in destination.resolve().parents:
                raise RuntimeError(f"Refusing output outside acquisition root: {destination}")
            audit = download(product, destination)
            records.append(
                {
                    "system_label": label,
                    "kind": "chandra_evt2",
                    "obsid": int(product["obsid"]),
                    "url": product["url"],
                    "local_path": destination.relative_to(ROOT).as_posix(),
                    **audit,
                }
            )
            print(label, "chandra_evt2", destination.name, audit["size_bytes"], flush=True)

    provenance = {
        "provenance_version": "P0566-A383-MS2137-TRANSFER-ACQUISITION-RESULTS-0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": CONFIG.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(CONFIG),
        "feasibility_report_path": protocol["inputs"]["feasibility_report"],
        "feasibility_report_sha256": sha256(
            ROOT / protocol["inputs"]["feasibility_report"]
        ),
        "records": records,
        "science_arrays_opened": False,
    }
    provenance_path = ROOT / protocol["outputs"]["provenance"]
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {provenance_path} with {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
