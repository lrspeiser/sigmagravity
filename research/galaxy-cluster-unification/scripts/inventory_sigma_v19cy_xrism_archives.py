#!/usr/bin/env python3
"""Inventory the frozen V19CY XRISM archive roots without downloading data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_direct_icm_velocity_evidence.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19cy_direct_icm_velocity_evidence"
USER_AGENT = "SigmaGravity-V19CY-Archive-Inventory/1.0"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != "SIGMA-V19CY-DIRECT-ICM-VELOCITY-EVIDENCE-1.0.0":
        raise RuntimeError("unexpected V19CY protocol")
    authorization = config["authorization"]
    if not authorization["inventory_named_public_archives_now"]:
        raise RuntimeError("V19CY archive inventory is not authorized")
    for key in (
        "open_validation_outcomes_before_development_freeze",
        "open_holdout_outcomes_before_validation_pass",
        "open_lensing_halo_or_gravity_targets",
        "derive_or_select_action",
        "change_gravity_formula_or_parameter",
    ):
        if authorization[key]:
            raise RuntimeError(f"V19CY authorization boundary is open: {key}")
    parents = config["parents"]
    for key in ("v19cx_config", "v19cx_report", "pre_deep_direction_checkpoint"):
        path = ROOT / parents[key]
        if not path.is_file() or sha256(path) != parents[f"{key}_sha256"]:
            raise RuntimeError(f"V19CY parent changed: {key}")


def fetch_directory(url: str, timeout: float) -> list[str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get_content_type() != "text/html":
            raise RuntimeError(f"archive directory did not return HTML: {url}")
        text = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(text)
    return parser.links


def head_file(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is None:
            raise RuntimeError(f"archive file lacks Content-Length: {url}")
        return {
            "bytes": int(length),
            "last_modified": response.headers.get("Last-Modified", ""),
            "etag": response.headers.get("ETag", ""),
            "content_type": response.headers.get_content_type(),
        }


def inventory_root(root_url: str, timeout: float) -> list[dict[str, Any]]:
    if not root_url.endswith("/"):
        raise RuntimeError(f"archive root is not a directory URL: {root_url}")
    parsed_root = urlparse(root_url)
    root_path = parsed_root.path
    pending = [root_url]
    seen_directories: set[str] = set()
    files: list[dict[str, Any]] = []
    while pending:
        directory = pending.pop()
        if directory in seen_directories:
            continue
        seen_directories.add(directory)
        for href in fetch_directory(directory, timeout):
            if href.startswith("?") or href in {"../", "/"}:
                continue
            target = urljoin(directory, href)
            parsed = urlparse(target)
            if parsed.scheme != parsed_root.scheme or parsed.netloc != parsed_root.netloc:
                continue
            if not parsed.path.startswith(root_path) or parsed.path == root_path:
                continue
            if target.endswith("/"):
                pending.append(target)
                continue
            metadata = head_file(target, timeout)
            files.append(
                {
                    "relative_path": parsed.path[len(root_path) :],
                    "url": target,
                    **metadata,
                }
            )
    files.sort(key=lambda row: row["relative_path"])
    return files


def build_manifest(config: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, split in config["evidence_split"].items():
        for observation in split["observations"]:
            for item in inventory_root(observation["archive_url"], timeout):
                rows.append(
                    {
                        "role": role,
                        "cluster": split["cluster"],
                        "obsid": observation["obsid"],
                        **item,
                    }
                )
    rows.sort(key=lambda row: (row["role"], row["cluster"], row["obsid"], row["relative_path"]))
    return rows


def write_outputs(config_path: Path, output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "archive_manifest.csv"
    fields = ["role", "cluster", "obsid", "relative_path", "url", "bytes", "last_modified", "etag", "content_type"]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_observation: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_role[row["role"]]["files"] += 1
        by_role[row["role"]]["bytes"] += int(row["bytes"])
        key = f"{row['cluster']}:{row['obsid']}"
        item = by_observation.setdefault(key, {"cluster": row["cluster"], "obsid": row["obsid"], "files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += int(row["bytes"])
    report = {
        "protocol_version": "SIGMA-V19CY-XRISM-ARCHIVE-INVENTORY-1.0.0",
        "status": "named_public_xrism_archives_inventoried_without_scientific_outcome_access",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_path.resolve().relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "manifest": {
            "path": manifest.resolve().relative_to(ROOT).as_posix(),
            "rows": len(rows),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
        "remote_totals": {
            "files": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "by_role": dict(sorted(by_role.items())),
            "by_observation": [by_observation[key] for key in sorted(by_observation)],
        },
        "file_bodies_downloaded": False,
        "scientific_velocity_outcomes_opened": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "validation_and_holdout_outcome_seals_preserved": True,
    }
    (output / "archive_inventory_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    rows = build_manifest(config, args.timeout)
    report = write_outputs(args.config, args.output, rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
