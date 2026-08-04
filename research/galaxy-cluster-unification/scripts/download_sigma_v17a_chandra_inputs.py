#!/usr/bin/env python3
"""Download and hash analysis-grade Chandra inputs for Sigma v17A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v17a_chandra_acquisition.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v17a_chandra_acquisition"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source: Path, destination: Path, attempts: int = 30) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.25 * (attempt + 1), 2.0))


def directory_links(url: str) -> list[str]:
    response = requests.get(url, timeout=(30, 90))
    response.raise_for_status()
    parser = LinkParser()
    parser.feed(response.text)
    return [value for value in parser.links if value not in {"../", "/"}]


def obsid_url(base: str, obsid: int) -> str:
    return f"{base}/{str(obsid)[-1]}/{obsid}/"


def role_for(filename: str) -> str:
    for role in (
        "evt1",
        "evt2",
        "bpix1",
        "fov1",
        "asol1",
        "osol1",
        "aqual1",
        "flt1",
        "msk1",
        "mtl1",
        "stat1",
        "bias0",
        "pbk0",
        "eph1",
    ):
        if f"_{role}." in filename:
            return role
    return "metadata"


def selected_jobs(config: dict, cluster: str, obsid: int) -> list[dict]:
    base = obsid_url(config["archive_base_url"], obsid)
    output = ROOT / config["raw_directory"] / cluster / str(obsid)
    products = config["included_products"]
    jobs = []
    locations = {
        "": (products["root"], "root"),
        "primary/": (products["primary_suffixes"], "primary"),
        "secondary/": (products["secondary_suffixes"], "secondary"),
        "secondary/aspect/": (
            products["secondary_aspect_suffixes"],
            "secondary/aspect",
        ),
        "secondary/ephem/": (
            products["secondary_ephem_suffixes"],
            "secondary/ephem",
        ),
    }
    for subdirectory, (patterns, relative_directory) in locations.items():
        url = urljoin(base, subdirectory)
        for filename in directory_links(url):
            if filename.endswith("/"):
                continue
            selected = (
                filename in patterns
                if subdirectory == ""
                else any(filename.endswith(pattern) for pattern in patterns)
            )
            if not selected:
                continue
            jobs.append(
                {
                    "cluster": cluster,
                    "obsid": obsid,
                    "role": role_for(filename),
                    "url": urljoin(url, filename),
                    "path": output / relative_directory / filename,
                }
            )
    return jobs


def download(job: dict, attempts: int = 4) -> dict:
    path = job["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(attempts):
        try:
            with requests.get(job["url"], stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                expected_header = response.headers.get("Content-Length")
                expected = int(expected_header) if expected_header else None
                if path.exists() and (expected is None or path.stat().st_size == expected):
                    return {
                        **job,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                        "reused": True,
                    }
                with partial.open("wb") as handle:
                    response.raw.decode_content = False
                    shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
            if expected is not None and partial.stat().st_size != expected:
                raise RuntimeError(
                    f"{path.name}: expected {expected} bytes, got {partial.stat().st_size}"
                )
            replace_with_retry(partial, path)
            return {
                **job,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "reused": False,
            }
        except (OSError, RuntimeError, requests.RequestException):
            if partial.exists():
                partial.unlink()
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def validate_roles(records: list[dict], requirements: dict[str, int]) -> dict:
    counts = {}
    for row in records:
        counts[row["role"]] = counts.get(row["role"], 0) + 1
    missing = {
        role: minimum - counts.get(role, 0)
        for role, minimum in requirements.items()
        if counts.get(role, 0) < minimum
    }
    if missing:
        raise RuntimeError(f"required Chandra product roles missing: {missing}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != (
        "frozen before downloading any analysis-grade event or calibration product"
    ):
        raise RuntimeError("the v17A Chandra acquisition protocol is not frozen")

    jobs = []
    for cluster, cluster_config in config["clusters"].items():
        for obsid in cluster_config["obsids"]:
            jobs.extend(selected_jobs(config, cluster, int(obsid)))
    if not jobs:
        raise RuntimeError("archive inventory produced no selected files")

    completed = []
    with ThreadPoolExecutor(max_workers=int(config["maximum_workers"])) as pool:
        futures = {pool.submit(download, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed.append(result)
            print(f"completed {index}/{len(jobs)} {result['path'].name}", flush=True)

    records = []
    per_obsid = []
    requirements = {
        role: int(minimum) for role, minimum in config["required_roles_per_obsid"].items()
    }
    for cluster, cluster_config in config["clusters"].items():
        for obsid in cluster_config["obsids"]:
            selected = [
                row for row in completed if row["cluster"] == cluster and row["obsid"] == int(obsid)
            ]
            role_counts = validate_roles(selected, requirements)
            per_obsid.append(
                {
                    "cluster": cluster,
                    "obsid": int(obsid),
                    "files": len(selected),
                    "bytes": sum(row["bytes"] for row in selected),
                    "role_counts": role_counts,
                }
            )
            for row in selected:
                records.append(
                    {
                        "cluster": cluster,
                        "obsid": int(obsid),
                        "role": row["role"],
                        "url": row["url"],
                        "relative_path": row["path"].relative_to(ROOT).as_posix(),
                        "bytes": row["bytes"],
                        "sha256": row["sha256"],
                        "reused": row["reused"],
                    }
                )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "analysis_grade_Chandra_archive_products_downloaded_and_hashed",
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "files": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "per_obsid": per_obsid,
        "records": sorted(
            records,
            key=lambda row: (row["cluster"], row["obsid"], row["relative_path"]),
        ),
        "lensing_target_opened": False,
        "temperature_map_constructed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "files", "bytes", "per_obsid")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
