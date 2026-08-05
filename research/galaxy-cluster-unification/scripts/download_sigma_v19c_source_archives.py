#!/usr/bin/env python3
"""Download the frozen, source-only Sigma v19C paper archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19c_source_archive_acquisition.json"
USER_AGENT = "sigmagravity-source-audit/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    if not config["status"].startswith("frozen before downloading"):
        raise RuntimeError("v19C acquisition protocol is not frozen")
    authorization = config["authorization"]
    if not authorization["download_exact_source_assets"]:
        raise RuntimeError("source archive acquisition is not authorized")
    forbidden_actions = (
        "download_lensing_paper_source_archives",
        "download_multiple_image_coordinates",
        "read_lens_models_or_inferred_halo_products",
        "construct_causal_source",
        "fit_gravity_parameters",
        "open_holdout",
    )
    if any(authorization[key] for key in forbidden_actions):
        raise RuntimeError("v19C acquisition authorizes a prohibited action")

    hashes = {"config": sha256(config_path)}
    for key in ("replacement_screen_config", "replacement_screen_report"):
        parent = ROOT / config["parents"][key]
        actual = sha256(parent)
        if actual != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"frozen {key} changed")
        hashes[key] = actual

    screen = load_json(ROOT / config["parents"]["replacement_screen_report"])
    if sorted(screen["selected_development_pair"]) != sorted(config["selected_clusters"]):
        raise RuntimeError("acquisition clusters differ from v19B source-gate survivors")
    if not screen["gate_results"]["source_archive_acquisition_authorized"]:
        raise RuntimeError("v19B did not authorize source archive acquisition")
    if not screen["gate_results"]["all_replacement_lensing_targets_remained_sealed"]:
        raise RuntimeError("v19B replacement targets were not sealed")

    forbidden_ids = set(config["explicitly_forbidden_arxiv_ids"])
    output_root = (ROOT / config["output_root"]).resolve()
    selected = set(config["selected_clusters"])
    seen: set[Path] = set()
    for asset in config["assets"]:
        if asset["cluster"] not in selected:
            raise RuntimeError("asset belongs to an unselected cluster")
        if asset["arxiv_id"] in forbidden_ids:
            raise RuntimeError("forbidden lensing paper entered source acquisition")
        if asset["contains_lensing_target_payload"]:
            raise RuntimeError("asset declares a lensing target payload")
        target = (output_root / asset["filename"]).resolve()
        if not target.is_relative_to(output_root):
            raise RuntimeError("asset path escapes the frozen output root")
        if target in seen:
            raise RuntimeError("duplicate acquisition destination")
        seen.add(target)
    return hashes


def archive_members(path: Path) -> list[str]:
    with tarfile.open(path, mode="r:*") as archive:
        return sorted(member.name for member in archive.getmembers() if member.isfile())


def validate_download(path: Path, minimum_bytes: int) -> list[str]:
    if path.stat().st_size < minimum_bytes:
        raise RuntimeError(f"download is implausibly small: {path}")
    prefix = path.read_bytes()[:256].lower()
    if b"<html" in prefix or b"<!doctype" in prefix:
        raise RuntimeError(f"download is HTML rather than an archive: {path}")
    members = archive_members(path)
    if not members:
        raise RuntimeError(f"archive has no files: {path}")
    return members


def download(url: str, destination: Path, minimum_bytes: int) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    if partial.exists():
        partial.unlink()
    try:
        with requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            timeout=(60, 300),
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        members = validate_download(partial, minimum_bytes)
        partial.replace(destination)
        return members
    finally:
        if partial.exists():
            partial.unlink()


def existing_record_by_filename(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return {row["filename"]: row for row in manifest["assets"]}


def acquire(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    input_hashes = validate_config(config_path, config)
    output_root = (ROOT / config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "provenance.json"
    existing = existing_record_by_filename(manifest_path)
    minimum_bytes = int(config["minimum_archive_bytes"])

    records: list[dict[str, Any]] = []
    for asset in config["assets"]:
        destination = (output_root / asset["filename"]).resolve()
        prior = existing.get(asset["filename"])
        reused = False
        if destination.exists():
            if prior is None:
                raise RuntimeError(f"unmanifested existing archive: {destination}")
            members = validate_download(destination, minimum_bytes)
            if (
                prior["sha256"] != sha256(destination)
                or int(prior["bytes"]) != destination.stat().st_size
                or prior["url"] != asset["url"]
            ):
                raise RuntimeError(f"existing archive fails frozen provenance: {destination}")
            reused = True
        else:
            members = download(asset["url"], destination, minimum_bytes)
        records.append(
            {
                **asset,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "archive_file_count": len(members),
                "archive_members": members,
                "reused_verified_archive": reused,
            }
        )
        print(
            f"{'verified' if reused else 'downloaded'} {asset['arxiv_id']} "
            f"{destination.stat().st_size} bytes",
            flush=True,
        )

    manifest = {
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_hashes": input_hashes,
        "selection_frozen_before_download": True,
        "selected_clusters": config["selected_clusters"],
        "assets": records,
        "asset_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "all_replacement_lensing_targets_remained_sealed": True,
        "lensing_or_halo_payload_downloaded": False,
        "gravity_parameters_fit": 0,
        "source_construction_performed": False,
        "holdout_opened": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    manifest = acquire(args.config)
    print(
        json.dumps(
            {
                "asset_count": manifest["asset_count"],
                "total_bytes": manifest["total_bytes"],
                "lensing_or_halo_payload_downloaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
