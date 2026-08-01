#!/usr/bin/env python3
"""Download open RELICS baryons and opaque sealed lensing containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "p0640_relics_baryon_and_sealed_lensing_acquisition.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0640_relics_input_acquisition"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path, *, attempts: int = 30) -> None:
    """Tolerate short Windows scanner locks after closing a large download."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.25 * (attempt + 1), 2.0))


def hst_jobs(config: dict) -> list[dict]:
    base = config["hst_archive"]["base_url"]
    raw = ROOT / config["raw_directory"]
    jobs = []
    for target in config["targets"]:
        slug = target["slug"]
        catalog = f"hlsp_relics_hst_acs-wfc3ir_{slug}_multi_v1_cat.txt"
        image = f"hlsp_relics_hst_wfc3ir-60mas_{slug}_f160w_v1_drz.fits"
        segmentation = f"hlsp_relics_hst_acs-wfc3ir_{slug}_multi_v1_segm.fits"
        specs = [
            ("member_catalog", catalog, target["hst"]["catalog_bytes"], "catalogs"),
            ("f160w_image", image, target["hst"]["f160w_image_bytes"], "images/60mas-resolution"),
            (
                "segmentation",
                segmentation,
                target["hst"]["segmentation_bytes"],
                "catalogs",
            ),
        ]
        for role, filename, expected, remote_subdirectory in specs:
            jobs.append(
                {
                    "system": target["id"],
                    "kind": "open_baryon",
                    "role": role,
                    "url": f"{base}/{slug}/{remote_subdirectory}/{filename}",
                    "path": raw / target["id"] / "hst" / filename,
                    "expected_bytes": int(expected),
                }
            )
        for obsid, filename, expected, exposure in target["chandra"]:
            jobs.append(
                {
                    "system": target["id"],
                    "kind": "open_baryon",
                    "role": "chandra_center_image",
                    "obsid": int(obsid),
                    "exposure_s": float(exposure),
                    "url": (
                        f"{config['chandra_archive']['base_url']}/{str(obsid)[-1]}/"
                        f"{obsid}/primary/{filename}"
                    ),
                    "path": raw / target["id"] / "chandra" / filename,
                    "expected_bytes": int(expected),
                }
            )
    return jobs


def all_jobs(config: dict) -> list[dict]:
    jobs = hst_jobs(config)
    sealed = ROOT / config["sealed_directory"]
    for item in config["sealed_constraint_containers"]:
        jobs.append(
            {
                "system": ",".join(item["systems"]),
                "kind": "sealed_constraint_container",
                "role": item["id"],
                "url": item["url"],
                "path": sealed / item["filename"],
                "expected_bytes": int(item["bytes"]),
                "handling": item["handling"],
            }
        )
    reference = ROOT / config["reference_directory"]
    for item in config["open_reference_metadata"]:
        jobs.append(
            {
                "system": "PLCKG287",
                "kind": "open_reference_metadata",
                "role": item["id"],
                "url": item["url"],
                "path": reference / item["filename"],
                "expected_bytes": int(item["bytes"]),
            }
        )
    return jobs


def download(job: dict, *, attempts: int = 4) -> dict:
    path = job["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = int(job["expected_bytes"])
    headers = {}
    if path.exists() and path.stat().st_size == expected:
        return {**job, "response_headers": headers, "reused": True}
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists() and partial.stat().st_size == expected:
        replace_with_retry(partial, path)
        return {**job, "response_headers": headers, "reused": True}
    for attempt in range(attempts):
        try:
            with requests.get(job["url"], stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                headers = {
                    "etag": response.headers.get("ETag", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
                with partial.open("wb") as handle:
                    response.raw.decode_content = False
                    shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
            if partial.stat().st_size != expected:
                raise RuntimeError(
                    f"{path.name}: expected {expected} bytes, received {partial.stat().st_size}"
                )
            replace_with_retry(partial, path)
            return {**job, "response_headers": headers, "reused": False}
        except (OSError, RuntimeError, requests.RequestException):
            if partial.exists():
                try:
                    partial.unlink()
                except PermissionError:
                    if partial.stat().st_size == expected:
                        replace_with_retry(partial, path)
                        return {**job, "response_headers": headers, "reused": False}
                    raise
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "baryonic_inputs_open_lensing_constraints_opaque_before_candidate_lock":
        raise RuntimeError("P0640 acquisition config is not frozen for blind acquisition")
    jobs = all_jobs(config)
    completed = []
    with ThreadPoolExecutor(max_workers=int(config["maximum_workers"])) as pool:
        futures = {pool.submit(download, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            print(f"completed {index}/{len(jobs)}", flush=True)
    records = []
    for job in sorted(completed, key=lambda item: (item["kind"], item["system"], item["role"], str(item["path"]))):
        path = job["path"]
        records.append(
            {
                "system": job["system"],
                "kind": job["kind"],
                "role": job["role"],
                "obsid": job.get("obsid"),
                "exposure_s": job.get("exposure_s"),
                "url": job["url"],
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "etag": job["response_headers"].get("etag", ""),
                "last_modified": job["response_headers"].get("last_modified", ""),
                "reused": bool(job["reused"]),
            }
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "status": "downloaded_and_hashed_without_opening_sealed_payloads",
        "files": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "sealed_state": config["sealed_state"],
        "records": records,
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "files", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
