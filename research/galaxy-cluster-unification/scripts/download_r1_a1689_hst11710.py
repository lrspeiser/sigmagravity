#!/usr/bin/env python3
"""Download the frozen public HST/ACS program-11710 acquisition set."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_hst11710_acquisition_protocol.json"
MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"


def invoke(request: dict) -> dict:
    response = requests.post(MAST_INVOKE, data={"request": json.dumps(request)}, timeout=120)
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "COMPLETE":
        raise RuntimeError(f"MAST query failed: {result}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def download(uri: str, target: Path, expected_size: int) -> dict:
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == expected_size:
        return {"filename": target.name, "bytes": expected_size, "sha256": sha256(target), "downloaded": False}
    if target.exists():
        raise RuntimeError(f"Existing file has wrong size: {target}")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(MAST_DOWNLOAD, params={"uri": uri}, headers=headers, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    stream.write(block)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {target.name}: {partial.stat().st_size} != {expected_size}")
    partial.replace(target)
    return {"filename": target.name, "bytes": expected_size, "sha256": sha256(target), "downloaded": True}


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    parents = cfg["selection_basis"]["parent_observation_ids"]
    observation_request = {
        "service": "Mast.Caom.Filtered.Position",
        "params": {
            "columns": "obsid,obs_collection,proposal_id,target_name,instrument_name,filters,t_exptime,t_min,t_max,calib_level,dataRights,s_ra,s_dec,obs_id",
            "filters": [
                {"paramName": "obs_collection", "values": ["HST"]},
                {"paramName": "proposal_id", "values": [cfg["selection_basis"]["proposal_id"]]},
                {"paramName": "instrument_name", "values": [cfg["selection_basis"]["instrument"]]},
                {"paramName": "filters", "values": [cfg["selection_basis"]["filter"]]},
            ],
            "position": "197.873, -1.3410833333333333, 0.05",
        },
        "format": "json",
        "pagesize": 200,
        "page": 1,
    }
    observations = invoke(observation_request)
    (ROOT / cfg["outputs"]["observation_query"]).write_text(
        json.dumps({"request": observation_request, "response": observations}, indent=2) + "\n",
        encoding="utf-8",
    )
    product_request = {
        "service": "Mast.Caom.Products",
        "params": {"obsid": ",".join(str(value) for value in parents)},
        "format": "json",
        "pagesize": 5000,
        "page": 1,
    }
    products = invoke(product_request)
    (ROOT / cfg["outputs"]["product_query"]).write_text(
        json.dumps({"request": product_request, "response": products}, indent=2) + "\n",
        encoding="utf-8",
    )

    selected_by_name: dict[str, dict] = {}
    selections = cfg["frozen_product_selection"]
    for definition in (selections["individual_exposures"], selections["visit_mosaics"], selections["association_tables"]):
        pattern = re.compile(definition["legacy_filename_regex"], re.IGNORECASE)
        matches = {
            str(row["productFilename"]): row
            for row in products["data"]
            if str(row.get("productSubGroupDescription")) == definition["subgroup"]
            and pattern.fullmatch(str(row.get("productFilename", "")))
        }
        if len(matches) != definition["expected_unique_files"]:
            raise RuntimeError(f"Frozen {definition['subgroup']} count changed: {len(matches)}")
        if sum(int(row["size"]) for row in matches.values()) != definition["expected_total_archive_bytes"]:
            raise RuntimeError(f"Frozen {definition['subgroup']} archive byte total changed")
        selected_by_name.update(matches)
    if len(selected_by_name) != cfg["expected_archive_totals"]["files"]:
        raise RuntimeError("Frozen total selected file count changed")
    if sum(int(row["size"]) for row in selected_by_name.values()) != cfg["expected_archive_totals"]["bytes"]:
        raise RuntimeError("Frozen selected archive byte total changed")

    paper_targets = [
        (cfg["paper_products"]["pdf_url"], ROOT / cfg["outputs"]["paper_pdf"]),
        (cfg["paper_products"]["source_url"], ROOT / cfg["outputs"]["paper_source"]),
    ]
    for url, target in paper_targets:
        if not target.exists():
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            target.write_bytes(response.content)

    rows = sorted(selected_by_name.values(), key=lambda row: str(row["productFilename"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                download,
                str(row["dataURI"]),
                output / str(row["productFilename"]),
                int(row["size"]),
            ): row
            for row in rows
        }
        downloaded = []
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            row = futures[future]
            result.update({
                "data_uri": row["dataURI"],
                "product_subgroup": row["productSubGroupDescription"],
                "calib_level": int(row["calib_level"]),
            })
            downloaded.append(result)
            print(json.dumps({"completed_files": completed, "total_files": len(rows), "filename": result["filename"]}), flush=True)

    manifest = {
        "provenance_version": cfg["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_pixel_inspection": True,
        "archive": "MAST",
        "proposal_id": cfg["selection_basis"]["proposal_id"],
        "files": sorted(downloaded, key=lambda row: row["filename"]),
        "selected_file_count": len(downloaded),
        "selected_archive_bytes": sum(item["bytes"] for item in downloaded),
        "paper": [
            {"path": str(target.relative_to(ROOT)).replace("\\", "/"), "url": url, "bytes": target.stat().st_size, "sha256": sha256(target)}
            for url, target in paper_targets
        ],
        "authorization": cfg["authorization"],
    }
    (ROOT / cfg["outputs"]["manifest"]).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": cfg["outputs"]["manifest"], "selected_file_count": len(downloaded), "selected_archive_bytes": manifest["selected_archive_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
