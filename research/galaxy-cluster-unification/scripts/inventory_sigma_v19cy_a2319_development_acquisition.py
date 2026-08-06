#!/usr/bin/env python3
"""Freeze the exact A2319 development acquisition payload before download."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_development_acquisition.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19cy_direct_icm_velocity_evidence"
USER_AGENT = "SigmaGravity-V19CY-A2319-Acquisition-Inventory/1.0"


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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any]) -> None:
    expected = "SIGMA-V19CY-A2319-DEVELOPMENT-ACQUISITION-1.0.0"
    if config.get("protocol_version") != expected:
        raise RuntimeError("unexpected A2319 acquisition protocol")
    if not config.get("status", "").startswith("frozen after"):
        raise RuntimeError("A2319 acquisition protocol is not frozen")
    parent = config["parent"]
    for key in ("config", "archive_manifest", "archive_inventory_report"):
        path = ROOT / parent[key]
        if not path.is_file() or sha256(path) != parent[f"{key}_sha256"]:
            raise RuntimeError(f"A2319 acquisition parent changed: {key}")
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


def fetch_links(url: str, timeout: float) -> list[str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        if response.headers.get_content_type() != "text/html":
            raise RuntimeError(f"directory did not return HTML: {url}")
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset, errors="replace")
    parser = LinkParser()
    parser.feed(body)
    return parser.links


def head_file(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is None:
            raise RuntimeError(f"remote file lacks Content-Length: {url}")
        return {
            "bytes": int(length),
            "last_modified": response.headers.get("Last-Modified", ""),
            "etag": response.headers.get("ETag", ""),
            "content_type": response.headers.get_content_type(),
        }


def inventory_root(root_url: str, timeout: float) -> list[dict[str, Any]]:
    parsed_root = urlparse(root_url)
    if not root_url.endswith("/"):
        raise RuntimeError(f"archive root is not a directory: {root_url}")
    root_path = parsed_root.path
    pending = [root_url]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    while pending:
        directory = pending.pop()
        if directory in seen:
            continue
        seen.add(directory)
        for href in fetch_links(directory, timeout):
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
            rows.append(
                {
                    "relative_path": parsed.path[len(root_path) :],
                    "url": target,
                    **head_file(target, timeout),
                }
            )
    return sorted(rows, key=lambda row: row["relative_path"])


def classify_chandra_role(filename: str) -> str:
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


def selected_chandra_rows(config: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    spec = config["chandra"]
    products = spec["included_products"]
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
    rows: list[dict[str, Any]] = []
    for obsid in spec["obsids"]:
        root = f"{spec['archive_base_url']}/{str(obsid)[-1]}/{obsid}/"
        for subdirectory, (patterns, destination) in locations.items():
            directory = urljoin(root, subdirectory)
            for href in fetch_links(directory, timeout):
                if href.endswith("/") or href.startswith("?") or href == "../":
                    continue
                selected = href in patterns if not subdirectory else any(
                    href.endswith(pattern) for pattern in patterns
                )
                if not selected:
                    continue
                url = urljoin(directory, href)
                relative = f"{destination}/{href}"
                rows.append(
                    {
                        "asset_group": "chandra_ssm",
                        "role": classify_chandra_role(href),
                        "obsid": str(obsid),
                        "relative_path": relative,
                        "download_path": f"chandra/{obsid}/{relative}",
                        "url": url,
                        **head_file(url, timeout),
                    }
                )
    requirements = {key: int(value) for key, value in spec["required_roles_per_obsid"].items()}
    for obsid in spec["obsids"]:
        counts = Counter(row["role"] for row in rows if row["obsid"] == str(obsid))
        missing = {role: minimum - counts[role] for role, minimum in requirements.items() if counts[role] < minimum}
        if missing:
            raise RuntimeError(f"Chandra ObsID {obsid} lacks required roles: {missing}")
    return rows


def selected_xrism_science_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    xrism = config["xrism"]
    allowed_obsids = set(xrism["science_obsids"])
    prefixes = tuple(xrism["science_manifest_include_prefixes"])
    parent_manifest = ROOT / config["parent"]["archive_manifest"]
    rows: list[dict[str, Any]] = []
    with parent_manifest.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            if source["role"] != "development" or source["obsid"] not in allowed_obsids:
                continue
            if not source["relative_path"].startswith(prefixes):
                continue
            rows.append(
                {
                    "asset_group": "xrism_science",
                    "role": PurePosixPath(source["relative_path"]).parts[0],
                    "obsid": source["obsid"],
                    "relative_path": source["relative_path"],
                    "download_path": f"xrism/{source['obsid']}/{source['relative_path']}",
                    "url": source["url"],
                    "bytes": int(source["bytes"]),
                    "last_modified": source["last_modified"],
                    "etag": source["etag"],
                    "content_type": source["content_type"],
                }
            )
    found_obsids = {row["obsid"] for row in rows}
    if found_obsids != allowed_obsids:
        raise RuntimeError(f"science manifest is missing ObsIDs: {sorted(allowed_obsids - found_obsids)}")
    return rows


def selected_predecessor_rows(config: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    spec = config["xrism"]["calibration_predecessor"]
    inventory = inventory_root(spec["archive_url"], timeout)
    by_path = {row["relative_path"]: row for row in inventory}
    expected = list(spec["include_exact_paths"])
    missing = [path for path in expected if path not in by_path]
    if missing:
        raise RuntimeError(f"calibration predecessor paths are missing: {missing}")
    rows: list[dict[str, Any]] = []
    for path in expected:
        source = by_path[path]
        rows.append(
            {
                "asset_group": "xrism_calibration_predecessor",
                "role": "gain_dependency",
                "obsid": spec["obsid"],
                "relative_path": path,
                "download_path": f"xrism/{spec['obsid']}/{path}",
                "url": source["url"],
                "bytes": source["bytes"],
                "last_modified": source["last_modified"],
                "etag": source["etag"],
                "content_type": source["content_type"],
            }
        )
    return rows


def external_rows(config: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    report_url = config["method_authorities"]["official_gain_report"]
    report_name = PurePosixPath(urlparse(report_url).path).name
    rows.append(
        {
            "asset_group": "official_gain_report",
            "role": "energy_scale_quality_report",
            "obsid": "000100000",
            "relative_path": report_name,
            "download_path": f"documentation/{report_name}",
            "url": report_url,
            **head_file(report_url, timeout),
        }
    )
    for archive in config["calibration"]["archives"]:
        url = archive["url"]
        name = PurePosixPath(urlparse(url).path).name
        rows.append(
            {
                "asset_group": "caldb",
                "role": archive["role"],
                "obsid": "",
                "relative_path": name,
                "download_path": f"caldb/{name}",
                "url": url,
                **head_file(url, timeout),
            }
        )
    return rows


def build_manifest(config: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    rows = [
        *selected_xrism_science_rows(config),
        *selected_predecessor_rows(config, timeout),
        *selected_chandra_rows(config, timeout),
        *external_rows(config, timeout),
    ]
    paths = [row["download_path"] for row in rows]
    if len(paths) != len(set(paths)):
        duplicates = [path for path, count in Counter(paths).items() if count > 1]
        raise RuntimeError(f"duplicate acquisition destinations: {duplicates}")
    return sorted(rows, key=lambda row: (row["asset_group"], row["obsid"], row["relative_path"]))


def write_outputs(config_path: Path, output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "development_acquisition_manifest.csv"
    fields = [
        "asset_group",
        "role",
        "obsid",
        "relative_path",
        "download_path",
        "url",
        "bytes",
        "last_modified",
        "etag",
        "content_type",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_observation: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_group[row["asset_group"]]["files"] += 1
        by_group[row["asset_group"]]["bytes"] += int(row["bytes"])
        if row["obsid"]:
            key = f"{row['asset_group']}:{row['obsid']}"
            item = by_observation.setdefault(
                key,
                {"asset_group": row["asset_group"], "obsid": row["obsid"], "files": 0, "bytes": 0},
            )
            item["files"] += 1
            item["bytes"] += int(row["bytes"])
    report = {
        "protocol_version": "SIGMA-V19CY-A2319-DEVELOPMENT-ACQUISITION-INVENTORY-1.0.0",
        "status": "a2319_scientifically_complete_development_acquisition_frozen_before_payload_download",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": display_path(config_path),
        "config_sha256": sha256(config_path),
        "manifest": {
            "path": display_path(manifest),
            "rows": len(rows),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
        "remote_totals": {
            "files": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "by_asset_group": dict(sorted(by_group.items())),
            "by_observation": [by_observation[key] for key in sorted(by_observation)],
        },
        "payload_file_bodies_downloaded": False,
        "validation_or_holdout_asset_accessed": False,
        "lensing_halo_or_gravity_payload_opened": False,
        "validation_and_holdout_outcome_seals_preserved": True,
    }
    report_path = output / "development_acquisition_inventory_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)
    rows = build_manifest(config, args.timeout)
    report = write_outputs(config_path, args.output.resolve(), rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
