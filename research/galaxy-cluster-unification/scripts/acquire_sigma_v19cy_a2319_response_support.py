#!/usr/bin/env python3
"""Acquire and structurally audit frozen A2319 Resolve response support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19cy_a2319_response_support_acquisition.json"
USER_AGENT = "SigmaGravity-V19CY-A2319-Response-Support/1.0"
BLOCK_BYTES = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "SIGMA-V19CY-A2319-RESPONSE-SUPPORT-ACQUISITION-1.0.1":
        raise RuntimeError("unexpected response-support protocol")
    parent = ROOT / config["parent"]["diagnosis_report"]
    if not parent.is_file() or sha256(parent) != config["parent"]["diagnosis_report_sha256"]:
        raise RuntimeError("response-free diagnosis parent changed")
    diagnosis = json.loads(parent.read_text(encoding="utf-8"))
    if diagnosis.get("diagnosis", {}).get("supported_classification") != config["parent"]["required_status"]:
        raise RuntimeError("response-free diagnosis did not authorize response-aware acquisition")
    if not diagnosis.get("diagnosis", {}).get("authorize_response_aware_development_protocol"):
        raise RuntimeError("response-free diagnosis did not authorize response-aware development")
    files = config["files"]
    if len(files) != config["expected_files"]:
        raise RuntimeError("response-support file count changed")
    if sum(item["bytes"] for item in files) != config["expected_bytes"]:
        raise RuntimeError("response-support byte total changed")
    if len({item["filename"] for item in files}) != len(files):
        raise RuntimeError("duplicate response-support destination")
    authorization = config["authorization"]
    if not authorization["head_and_download_declared_official_support_files"]:
        raise RuntimeError("response-support acquisition is not authorized")
    for key in (
        "read_or_fit_A2319_science_energy_distribution",
        "generate_A2319_response_or_background",
        "fit_A2319_spectrum_or_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        if authorization[key]:
            raise RuntimeError(f"sealed response-support boundary is open: {key}")
    return config


def remote_metadata(item: dict[str, Any]) -> dict[str, Any]:
    response = requests.head(
        item["url"],
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        timeout=60,
        allow_redirects=True,
    )
    response.raise_for_status()
    result = {
        "bytes": int(response.headers.get("Content-Length", "-1")),
        "last_modified": response.headers.get("Last-Modified", ""),
        "etag": response.headers.get("ETag", ""),
    }
    expected = {key: item[key] for key in result}
    if result != expected:
        raise RuntimeError(f"remote metadata changed for {item['filename']}: {result!r}")
    return result


def acquire_one(item: dict[str, Any], destination: Path, attempts: int = 5) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item["bytes"])
    if destination.is_file():
        if destination.stat().st_size != expected:
            raise RuntimeError(f"existing response-support size is wrong: {destination}")
        return {"reused": True, "resumed": False, "bytes": expected, "sha256": sha256(destination)}
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset > expected:
                raise RuntimeError(f"partial response-support file exceeds frozen size: {partial}")
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(item["url"], headers=headers, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as stream:
                    response.raw.decode_content = False
                    shutil.copyfileobj(response.raw, stream, length=BLOCK_BYTES)
            if partial.stat().st_size != expected:
                raise RuntimeError(
                    f"{item['filename']}: expected {expected} bytes, got {partial.stat().st_size}"
                )
            os.replace(partial, destination)
            return {
                "reused": False,
                "resumed": resumed,
                "bytes": expected,
                "sha256": sha256(destination),
            }
        except (OSError, RuntimeError, requests.RequestException):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fits_structure(path: Path, rule: dict[str, Any]) -> dict[str, Any]:
    with fits.open(path, memmap=True) as hdus:
        names = [hdu.name for hdu in hdus]
        for required in rule.get("required_extensions", []):
            if required not in names:
                raise RuntimeError(f"{path.name} lacks required extension {required}")
        result: dict[str, Any] = {"extensions": names}
        if "extension" in rule:
            hdu = hdus[rule["extension"]]
            columns = list(hdu.columns.names or [])
            missing = sorted(set(rule["required_columns"]) - set(columns))
            if missing:
                raise RuntimeError(f"{path.name} lacks required columns: {missing}")
            result.update(
                {
                    "audited_extension": rule["extension"],
                    "rows": int(hdu.header.get("NAXIS2", 0)),
                    "required_columns_present": list(rule["required_columns"]),
                }
            )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_and_validate_config(config_path)
    raw_root = (ROOT / config["raw_root"]).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(raw_root)
    needed = sum(
        item["bytes"]
        for item in config["files"]
        if not (raw_root / item["filename"]).is_file()
    )
    if usage.free < needed + config["minimum_free_space_reserve_bytes"]:
        raise RuntimeError("insufficient free space for response-support acquisition and reserve")

    records = []
    rules = config["fits_structure"]
    for item in config["files"]:
        metadata = remote_metadata(item)
        destination = (raw_root / item["filename"]).resolve()
        if not destination.is_relative_to(raw_root):
            raise RuntimeError("response-support destination escapes raw root")
        acquired = acquire_one(item, destination)
        structure: dict[str, Any]
        if item["role"] in rules:
            structure = fits_structure(destination, rules[item["role"]])
        else:
            first_line = destination.read_text(encoding="utf-8").splitlines()[0]
            normalized_prefix = " ".join(first_line.split())[:16]
            if not normalized_prefix.startswith(config["acceptance"]["require_model_prefix"]):
                raise RuntimeError("NXB empirical model prefix changed")
            structure = {
                "literal_text_prefix": first_line[:17],
                "whitespace_normalized_text_prefix": normalized_prefix,
            }
        records.append(
            {
                "role": item["role"],
                "filename": item["filename"],
                "url": item["url"],
                "remote_metadata": metadata,
                **acquired,
                "structure": structure,
            }
        )

    report = {
        "protocol_version": "SIGMA-V19CY-A2319-RESPONSE-SUPPORT-ACQUISITION-RESULT-1.0.1",
        "status": "official_provisional_resolve_nxb_support_acquired_hashed_and_structurally_verified",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(config_path),
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "records": records,
        "validation_or_holdout_accessed": False,
        "science_energy_distribution_read_or_fit": False,
        "response_or_background_generated": False,
        "scientific_velocity_fit_performed": False,
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / config["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
