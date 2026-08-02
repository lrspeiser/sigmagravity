#!/usr/bin/env python3
"""Acquire the once-unlocked P0633 target and fixed-comparator products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "p0633_external_validation" / "unlock_manifest.json"
OUTPUT = ROOT / "results" / "p0710_external_target_acquisition"
GALAXY_DIR = ROOT / "data" / "raw" / "p0633_little_things_kinematics"
COMPARATOR_DIR = ROOT / "data" / "raw" / "p0633_relics_lensing_comparators"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(item: dict) -> dict:
    destination = Path(item["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("expected_bytes")
    if destination.exists() and (expected is None or destination.stat().st_size == expected):
        return {
            **item,
            "status": "reused",
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "etag": None,
            "last_modified": None,
        }
    with urllib.request.urlopen(item["url"], timeout=120) as response:
        headers = response.headers
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    try:
        size = temporary.stat().st_size
        if expected is not None and size != expected:
            raise RuntimeError(
                f"byte count mismatch for {item['url']}: expected {expected}, got {size}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **item,
        "status": "downloaded",
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
    }


def comparator_items(manifest: dict, include_sensitivity: bool) -> list[dict]:
    specification = manifest["compact_halo_comparator"]
    rows = []
    for system in specification["systems"]:
        methods = [(specification["primary_method"], specification["primary_version"])]
        if include_sensitivity:
            methods.append(
                (
                    specification["sensitivity_method"],
                    specification["sensitivity_versions"][system["id"]],
                )
            )
        for method, version in methods:
            version_path = (
                f"{version}/"
                if method == "glafic" or (system["id"] == "AS295" and version == "v2")
                else ""
            )
            base = specification["base_url_template"].format(
                slug=system["slug"], method=method, version_path=version_path
            )
            for product in specification["products"]:
                filename = specification["filename_template"].format(
                    slug=system["slug"],
                    method=method,
                    version=version,
                    product=product,
                )
                rows.append(
                    {
                        "domain": "cluster_comparator",
                        "system": system["id"],
                        "role": f"{method}_{version}_{product}",
                        "url": f"{base}{filename}",
                        "destination": str(
                            COMPARATOR_DIR / system["id"] / method / version / filename
                        ),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    if manifest["status"] != "authorized_for_exactly_one_external_parse":
        raise RuntimeError("P0709 does not authorize external acquisition")
    if manifest["outcomes_opened_at_manifest_creation"]:
        raise RuntimeError("unlock manifest has an invalid initial state")

    items = []
    for item in manifest["galaxy_moment_products"]:
        items.append(
            {
                "domain": "galaxy",
                "system": item["system"],
                "role": item["product"],
                "url": item["url"],
                "expected_bytes": item["expected_bytes"],
                "destination": str(GALAXY_DIR / item["system"] / item["filename"]),
            }
        )
    circular = manifest["published_circular_speed_source"]
    items.append(
        {
            "domain": "galaxy_publication",
            "system": "Iorio2017",
            "role": "published_kinematic_source",
            "url": circular["url"],
            "expected_bytes": circular["expected_bytes"],
            "destination": str(GALAXY_DIR / circular["filename"]),
        }
    )
    items.extend(comparator_items(manifest, not args.primary_only))

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(download, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"{result['status']:10s} {result['domain']:20s} "
                    f"{result['system']:10s} {result['role']}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - retain every failed future in the report
                failures.append({**item, "error": repr(error)})
                print(f"FAILED {item['url']}: {error}", flush=True)

    results.sort(key=lambda row: (row["domain"], row["system"], row["role"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "system",
        "role",
        "url",
        "destination",
        "status",
        "bytes",
        "sha256",
        "etag",
        "last_modified",
    ]
    with (OUTPUT / "provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    report = {
        "report_version": "P0710-EXTERNAL-TARGET-ACQUISITION-1.0.0",
        "status": "pass" if not failures and len(results) == len(items) else "fail",
        "unlock_manifest_sha256": sha256(manifest_path),
        "requested_products": len(items),
        "received_products": len(results),
        "failed_products": failures,
        "total_bytes": sum(int(row["bytes"]) for row in results),
        "primary_only": bool(args.primary_only),
        "target_outcomes_now_open": True,
        "P0633_sample_now_spent": True,
        "formula_changes_after_this_run_are_validation": False,
        "provenance_sha256": sha256(OUTPUT / "provenance.csv"),
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0710 external target acquisition

- Status: **{report['status'].upper()}**.
- Products received: **{len(results)} / {len(items)}**.
- Downloaded bytes: **{report['total_bytes']:,}**.
- P0633 target outcomes are now open and the sample is **spent**.
- Any formula change from here is exploratory rather than P0633 validation.
"""
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
