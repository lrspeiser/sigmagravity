#!/usr/bin/env python3
"""Download the frozen ten-cluster gravity-arc confirmation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "gravity-arc-fresh-sample/0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def download(url: str, destination: Path, expected_bytes: int | None = None) -> None:
    if destination.exists() and (expected_bytes is None or destination.stat().st_size == expected_bytes):
        return
    payload = fetch(url)
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise RuntimeError(f"{url}: expected {expected_bytes} bytes, received {len(payload)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(payload)
    partial.replace(destination)


def range_names(model: dict) -> list[str]:
    """Construct the archive's frozen map000--mapNNN naming convention."""
    token = f"_{model['method']}_{model['version']}_kappa.fits"
    if token not in model["best_filename"]:
        raise ValueError(f"cannot derive range names from {model['best_filename']}")
    return [
        model["best_filename"].replace(
            token,
            f"_{model['method']}-map{index:03d}_{model['version']}_kappa.fits",
        )
        for index in range(int(model["range_count"]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/gravity_arc_fresh_sample_protocol.json"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_download_or_fresh_map_spatial_inspection":
        raise RuntimeError("fresh-sample protocol is not frozen")

    acquisition = protocol["acquisition"]
    output = ROOT / acquisition["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path, int | None, dict]] = []
    for system in protocol["systems"]:
        catalog_url = acquisition["catalog_url_template"].format(**system)
        jobs.append(
            (
                catalog_url,
                output / "catalogs" / system["catalog_filename"],
                int(system["catalog_bytes"]),
                {
                    "system": system["label"],
                    "kind": "catalog",
                    "method": "",
                    "sample_index": "",
                },
            )
        )
        for model in system["models"]:
            directory = output / "models" / system["slug"] / model["method"]
            jobs.append(
                (
                    model["base_url"] + model["best_filename"],
                    directory / model["best_filename"],
                    None,
                    {
                        "system": system["label"],
                        "kind": "best_kappa",
                        "method": model["method"],
                        "sample_index": "",
                    },
                )
            )
            if model["method"] in acquisition["download_range_for_methods"]:
                names = range_names(model)
                if len(names) != int(model["range_count"]):
                    raise RuntimeError(
                        f"{system['label']} {model['method']}: expected "
                        f"{model['range_count']} range maps, found {len(names)}"
                    )
                for index, name in enumerate(names):
                    jobs.append(
                        (
                            model["base_url"] + "range/" + name,
                            directory / "range" / name,
                            None,
                            {
                                "system": system["label"],
                                "kind": "range_kappa",
                                "method": model["method"],
                                "sample_index": index,
                            },
                        )
                    )

    print(f"verifying or downloading {len(jobs)} frozen files", flush=True)
    with ThreadPoolExecutor(max_workers=int(acquisition["maximum_workers"])) as pool:
        futures = {
            pool.submit(download, url, path, expected): (url, path, metadata)
            for url, path, expected, metadata in jobs
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)

    records = []
    for url, path, _expected, metadata in jobs:
        records.append(
            {
                **metadata,
                "url": url,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path = ROOT / acquisition["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    provenance = {
        "protocol_version": protocol["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(config_path),
        "files": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(manifest_path),
    }
    provenance_path = ROOT / acquisition["provenance"]
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
