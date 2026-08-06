#!/usr/bin/env python3
"""Download and hash the frozen V19CY A2319 development acquisition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_development_acquisition.json"
DEFAULT_REPORT = (
    ROOT
    / "results"
    / "sigma_v19cy_direct_icm_velocity_evidence"
    / "development_acquisition_inventory_report.json"
)
USER_AGENT = "SigmaGravity-V19CY-A2319-Downloader/1.0"
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_inputs(
    config_path: Path, report_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_json(config_path)
    report = load_json(report_path)
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-DEVELOPMENT-ACQUISITION-1.0.0":
        raise RuntimeError("unexpected A2319 acquisition protocol")
    if report.get("status") != (
        "a2319_scientifically_complete_development_acquisition_frozen_before_payload_download"
    ):
        raise RuntimeError("A2319 inventory is not the frozen pre-download result")
    if sha256(config_path) != report["config_sha256"]:
        raise RuntimeError("A2319 acquisition config changed after inventory")
    manifest = ROOT / report["manifest"]["path"]
    if not manifest.is_file() or sha256(manifest) != report["manifest"]["sha256"]:
        raise RuntimeError("A2319 acquisition manifest changed after inventory")
    authorization = config["authorization"]
    if not authorization["inventory_and_download_all_listed_development_assets"]:
        raise RuntimeError("development acquisition is not authorized")
    for key in (
        "download_or_open_validation_assets",
        "download_or_open_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameter",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed acquisition boundary is open: {key}")
    return config, report, manifest


def read_jobs(manifest: Path, raw_root: Path) -> list[dict[str, Any]]:
    resolved_root = raw_root.resolve()
    jobs: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            destination = (resolved_root / row["download_path"]).resolve()
            if not destination.is_relative_to(resolved_root):
                raise RuntimeError(f"acquisition destination escapes raw root: {row['download_path']}")
            jobs.append(
                {
                    "asset_group": row["asset_group"],
                    "role": row["role"],
                    "obsid": row["obsid"],
                    "relative_path": row["relative_path"],
                    "download_path": row["download_path"],
                    "url": row["url"],
                    "expected_bytes": int(row["bytes"]),
                    "expected_etag": row["etag"],
                    "path": destination,
                }
            )
    if not jobs:
        raise RuntimeError("frozen acquisition manifest is empty")
    paths = [job["path"] for job in jobs]
    if len(paths) != len(set(paths)):
        raise RuntimeError("frozen acquisition manifest has duplicate destinations")
    return jobs


def bytes_still_needed(jobs: list[dict[str, Any]]) -> int:
    needed = 0
    for job in jobs:
        path = job["path"]
        partial = path.with_suffix(path.suffix + ".part")
        if path.is_file() and path.stat().st_size == job["expected_bytes"]:
            continue
        present = partial.stat().st_size if partial.is_file() else 0
        needed += max(0, job["expected_bytes"] - min(present, job["expected_bytes"]))
    return needed


def ensure_free_space(raw_root: Path, needed: int) -> dict[str, int]:
    raw_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(raw_root)
    reserve = max(5 * 1024**3, int(needed * 0.1))
    if usage.free < needed + reserve:
        raise RuntimeError(
            f"insufficient free space: need {needed + reserve} bytes including reserve, "
            f"have {usage.free}"
        )
    return {"needed_bytes": needed, "reserve_bytes": reserve, "free_bytes": usage.free}


def _stream_response(response: requests.Response, partial: Path, mode: str) -> None:
    with partial.open(mode) as stream:
        response.raw.decode_content = False
        shutil.copyfileobj(response.raw, stream, length=BLOCK_BYTES)


def download_one(job: dict[str, Any], attempts: int = 5) -> dict[str, Any]:
    path: Path = job["path"]
    expected = int(job["expected_bytes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == expected:
        return {
            **job,
            "bytes": expected,
            "sha256": sha256(path),
            "reused": True,
            "resumed": False,
        }
    if path.exists():
        raise RuntimeError(f"existing file has unexpected size: {path}")
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset > expected:
                raise RuntimeError(f"partial file exceeds frozen size: {partial}")
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(
                job["url"],
                headers=headers,
                stream=True,
                timeout=(30, 600),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code == 206:
                    mode = "ab"
                    resumed = True
                elif offset and response.status_code == 200:
                    mode = "wb"
                    resumed = False
                else:
                    mode = "wb"
                    resumed = False
                _stream_response(response, partial, mode)
            actual = partial.stat().st_size
            if actual != expected:
                raise RuntimeError(f"{path.name}: expected {expected} bytes, got {actual}")
            os.replace(partial, path)
            return {
                **job,
                "bytes": actual,
                "sha256": sha256(path),
                "reused": False,
                "resumed": resumed,
            }
        except (OSError, RuntimeError, requests.RequestException):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def write_provenance(
    output: Path,
    config_path: Path,
    inventory_report: dict[str, Any],
    records: list[dict[str, Any]],
    disk_preflight: dict[str, int],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, dict[str, int]] = {}
    for record in records:
        item = by_group.setdefault(record["asset_group"], {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(record["bytes"])
    compact_records = [
        {
            "asset_group": row["asset_group"],
            "role": row["role"],
            "obsid": row["obsid"],
            "relative_path": row["relative_path"],
            "download_path": row["download_path"],
            "url": row["url"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "reused": row["reused"],
            "resumed": row["resumed"],
        }
        for row in sorted(records, key=lambda item: item["download_path"])
    ]
    report = {
        "protocol_version": "SIGMA-V19CY-A2319-DEVELOPMENT-DOWNLOAD-1.0.0",
        "status": "all_frozen_a2319_development_payloads_downloaded_size_verified_and_sha256_hashed",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "frozen_manifest_sha256": inventory_report["manifest"]["sha256"],
        "files": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "by_asset_group": dict(sorted(by_group.items())),
        "disk_preflight": disk_preflight,
        "records": compact_records,
        "validation_or_holdout_asset_accessed": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "scientific_velocity_fit_performed": False,
        "validation_and_holdout_outcome_seals_preserved": True,
    }
    path = output / "development_download_provenance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v19cy_direct_icm_velocity_evidence",
    )
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    report_path = args.inventory_report.resolve()
    config, inventory_report, manifest = validate_frozen_inputs(config_path, report_path)
    raw_root = (ROOT / config["raw_directory"]).resolve()
    jobs = read_jobs(manifest, raw_root)
    disk_preflight = ensure_free_space(raw_root, bytes_still_needed(jobs))
    workers = int(args.workers or config["maximum_workers"])
    if workers < 1 or workers > 8:
        raise RuntimeError("workers must be between 1 and 8")
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            completed.append(record)
            state = "reused" if record["reused"] else ("resumed" if record["resumed"] else "downloaded")
            print(
                f"completed {index}/{len(jobs)} {state} {record['download_path']} "
                f"({record['bytes']} bytes)",
                flush=True,
            )
    report = write_provenance(
        args.output.resolve(),
        config_path,
        inventory_report,
        completed,
        disk_preflight,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "files", "bytes", "by_asset_group")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
