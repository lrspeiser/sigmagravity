#!/usr/bin/env python3
"""Acquire only the files frozen in the J1402 A1/J2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "r1_j1402_acquisition_replay_jacobian_protocol.json"
RAW_ROOT = (ROOT / "data" / "raw" / "r1_j1402").resolve()
USER_AGENT = "SigmaGravity-J1402-audit/0.1 (public archival data acquisition)"
CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class Download:
    group: str
    identity: str
    url: str
    relative_path: str
    expected_bytes: int | None = None
    expected_git_blob_sha1: str | None = None
    expected_prefix_hex: str | None = None


def load_protocol() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def protocol_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def _kcwi_ids(kcwi: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, group in [
        ("science_ids", "kcwi_science"),
        ("bias_ids", "kcwi_bias"),
        ("continuum_bar_ids", "kcwi_continuum_bar"),
        ("arc_ids", "kcwi_arc"),
        ("flat_ids", "kcwi_flat"),
    ]:
        rows.extend((group, koaid) for koaid in kcwi[key])
    rows.extend(("kcwi_standard_star", item["koaid"]) for item in kcwi["standard_star_ids"])
    return rows


def build_downloads(protocol: dict) -> list[Download]:
    acquisition = protocol["acquisition"]
    github = acquisition["Dinos_GitHub"]
    downloads = [
        Download(
            group="dinos_github",
            identity=item["path"],
            url=github["raw_url_template"].format(path=item["path"]),
            relative_path=f"dinos_repo/{item['path']}",
            expected_bytes=int(item["bytes"]),
            expected_git_blob_sha1=item["git_blob_sha1"],
        )
        for item in github["files"]
    ]

    full = acquisition["Dinos_full_output"]
    query = urllib.parse.urlencode(
        {"id": full["file_id"], "export": "download", "confirm": "t"}
    )
    downloads.append(
        Download(
            group="dinos_full_output",
            identity=full["file_id"],
            url=f"https://drive.usercontent.google.com/download?{query}",
            relative_path=f"dinos_output/{full['file_name']}",
            expected_prefix_hex="89484446",
        )
    )

    kcwi = acquisition["KCWI"]
    for group, koaid in _kcwi_ids(kcwi):
        filehand = kcwi["file_handle_template"].format(koaid=koaid)
        url = (
            "https://koa.ipac.caltech.edu/cgi-bin/getKOA/nph-getKOA"
            f"?instrument=KC&filehand={urllib.parse.quote(filehand, safe='/')}"
        )
        downloads.append(
            Download(
                group=group,
                identity=koaid,
                url=url,
                relative_path=f"kcwi/{group}/{koaid}",
                expected_prefix_hex="53494d504c45",
            )
        )
    return downloads


def resolve_destination(item: Download) -> Path:
    destination = (RAW_ROOT / item.relative_path).resolve()
    if destination == RAW_ROOT or RAW_ROOT not in destination.parents:
        raise ValueError(f"unsafe destination outside locked raw root: {destination}")
    return destination


def file_hashes(path: Path) -> tuple[int, str, str]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1()
    git_sha1.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return size, sha256.hexdigest(), git_sha1.hexdigest()


def verify_file(item: Download, path: Path) -> dict:
    size, sha256, git_sha1 = file_hashes(path)
    if item.expected_bytes is not None and size != item.expected_bytes:
        raise ValueError(
            f"{item.identity}: expected {item.expected_bytes} bytes, received {size}"
        )
    if item.expected_git_blob_sha1 is not None and git_sha1 != item.expected_git_blob_sha1:
        raise ValueError(
            f"{item.identity}: Git blob mismatch {git_sha1} != {item.expected_git_blob_sha1}"
        )
    if item.expected_prefix_hex is not None:
        expected = bytes.fromhex(item.expected_prefix_hex)
        with path.open("rb") as handle:
            prefix = handle.read(len(expected))
        if prefix != expected:
            raise ValueError(
                f"{item.identity}: file signature mismatch {prefix.hex()} != {expected.hex()}"
            )
    return {
        "bytes": size,
        "sha256": sha256,
        "git_blob_sha1": git_sha1 if item.expected_git_blob_sha1 else None,
    }


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "protocol": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": protocol_sha256(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "receipts": [],
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != protocol_sha256():
        raise ValueError("protocol changed after the first acquisition receipt")
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["receipt_count"] = len(manifest["receipts"])
    manifest["verified_bytes"] = sum(item["bytes"] for item in manifest["receipts"])
    manifest["complete"] = manifest["receipt_count"] == manifest["planned_file_count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def existing_receipt(manifest: dict, item: Download) -> dict | None:
    return next(
        (
            receipt
            for receipt in manifest["receipts"]
            if receipt["group"] == item.group and receipt["identity"] == item.identity
        ),
        None,
    )


def stream_download(item: Download, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(item.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
        headers = {key.lower(): value for key, value in response.headers.items()}
        while chunk := response.read(CHUNK_BYTES):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    verification = verify_file(item, partial)
    header_length = headers.get("content-length")
    if header_length is not None and int(header_length) != verification["bytes"]:
        raise ValueError(
            f"{item.identity}: HTTP Content-Length {header_length} != {verification['bytes']}"
        )
    os.replace(partial, destination)
    return {
        **verification,
        "content_type": headers.get("content-type"),
        "content_disposition": headers.get("content-disposition"),
        "last_modified": headers.get("last-modified"),
        "etag": headers.get("etag"),
    }


def acquire(item: Download, manifest: dict, manifest_path: Path) -> None:
    destination = resolve_destination(item)
    receipt = existing_receipt(manifest, item)
    if receipt is not None:
        if not destination.exists():
            raise FileNotFoundError(f"manifested file is missing: {destination}")
        verification = verify_file(item, destination)
        if verification["sha256"] != receipt["sha256"]:
            raise ValueError(f"manifest checksum mismatch for existing {item.identity}")
        print(f"SKIP verified {item.group}: {item.identity}", flush=True)
        return
    if destination.exists():
        raise FileExistsError(
            f"unmanifested destination already exists; refusing overwrite: {destination}"
        )

    print(f"GET {item.group}: {item.identity}", flush=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = stream_download(item, destination)
            break
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            partial = destination.with_suffix(destination.suffix + ".part")
            if partial.exists():
                partial.unlink()
            if attempt == 3:
                raise
            print(f"RETRY {attempt}/3 {item.identity}: {exc}", flush=True)
            time.sleep(2**attempt)
    else:
        raise RuntimeError(str(last_error))

    manifest["receipts"].append(
        {
            "group": item.group,
            "identity": item.identity,
            "relative_path": str(destination.relative_to(ROOT)).replace("\\", "/"),
            "source_url": item.url,
            "received_utc": datetime.now(timezone.utc).isoformat(),
            **response,
        }
    )
    write_manifest(manifest_path, manifest)
    print(
        f"OK {item.identity}: {response['bytes']} bytes sha256={response['sha256']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="*",
        help="optional exact group names; default acquires every locked group",
    )
    parser.add_argument("--plan", action="store_true", help="print the locked plan only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    if protocol["status"] != "frozen_before_any_J1402_science_array_download_or_inspection":
        raise ValueError("the expected frozen pre-download protocol status is absent")
    downloads = build_downloads(protocol)
    known_groups = sorted({item.group for item in downloads})
    if args.groups:
        unknown = sorted(set(args.groups) - set(known_groups))
        if unknown:
            raise ValueError(f"unknown groups: {unknown}; expected one of {known_groups}")
        downloads = [item for item in downloads if item.group in args.groups]
    if args.plan:
        print(json.dumps([asdict(item) for item in downloads], indent=2))
        return

    manifest_path = RAW_ROOT / "acquisition_manifest.json"
    manifest = load_manifest(manifest_path)
    full_plan = build_downloads(protocol)
    manifest["planned_file_count"] = len(full_plan)
    manifest["planned_groups"] = sorted({item.group for item in full_plan})
    write_manifest(manifest_path, manifest)
    for item in downloads:
        acquire(item, manifest, manifest_path)


if __name__ == "__main__":
    main()
