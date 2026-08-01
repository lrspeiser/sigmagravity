#!/usr/bin/env python3
"""Download the frozen public HST/ACS program-9289 acquisition set."""

from __future__ import annotations

import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

from download_r1_a1689_hst11710 import download, invoke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_a1689_hst9289_acquisition_protocol.json"


def selected_groups(rows: list[dict]) -> dict[str, dict[str, dict]]:
    groups = {
        "FLC": {
            str(row["productFilename"]): row for row in rows
            if str(row.get("productSubGroupDescription")) == "FLC"
            and str(row.get("productFilename", "")).startswith("j8e")
            and str(row.get("productFilename", "")).endswith("_flc.fits")
        },
        "DRC": {
            str(row["productFilename"]): row for row in rows
            if str(row.get("productSubGroupDescription")) == "DRC"
            and str(row.get("productFilename", "")).startswith("hst_9289_")
            and str(row.get("productFilename", "")).endswith("_drc.fits")
            and len(str(row["productFilename"]).split("_")[-2]) == 6
        },
        "ASN": {
            str(row["productFilename"]): row for row in rows
            if str(row.get("productSubGroupDescription")) == "ASN"
            and str(row.get("productFilename", "")).startswith("j8e")
            and str(row.get("productFilename", "")).endswith("_asn.fits")
        },
    }
    return groups


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    request = {
        "service": "Mast.Caom.Products",
        "params": {"obsid": ",".join(str(value) for value in cfg["selection_basis"]["aggregate_parent_observation_ids"])},
        "format": "json",
        "pagesize": 5000,
        "page": 1,
    }
    products = invoke(request)
    (ROOT / cfg["outputs"]["product_query"]).write_text(
        json.dumps({"request": request, "response": products}, indent=2) + "\n", encoding="utf-8"
    )
    groups = selected_groups(products["data"])
    definitions = cfg["frozen_product_selection"]
    for subgroup, key in (("FLC", "individual_exposures"), ("DRC", "aggregate_filter_visit_mosaics"), ("ASN", "association_tables")):
        definition = definitions[key]
        if len(groups[subgroup]) != definition["expected_unique_files"]:
            raise RuntimeError(f"Frozen {subgroup} count changed: {len(groups[subgroup])}")
        if sum(int(row["size"]) for row in groups[subgroup].values()) != definition["expected_total_archive_bytes"]:
            raise RuntimeError(f"Frozen {subgroup} archive byte total changed")
    rows = [row for group in groups.values() for row in group.values()]
    if len(rows) != cfg["expected_archive_totals"]["files"] or sum(int(row["size"]) for row in rows) != cfg["expected_archive_totals"]["bytes"]:
        raise RuntimeError("Frozen program-9289 total changed")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download, str(row["dataURI"]), output / str(row["productFilename"]), int(row["size"])): row
            for row in rows
        }
        completed_rows = []
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            row = futures[future]
            result.update({"data_uri": row["dataURI"], "product_subgroup": row["productSubGroupDescription"], "calib_level": int(row["calib_level"])})
            completed_rows.append(result)
            print(json.dumps({"completed_files": completed, "total_files": len(rows), "filename": result["filename"]}), flush=True)
    manifest = {
        "provenance_version": cfg["protocol_version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_pixel_inspection": True,
        "archive": "MAST",
        "proposal_id": cfg["selection_basis"]["proposal_id"],
        "files": sorted(completed_rows, key=lambda row: row["filename"]),
        "selected_file_count": len(completed_rows),
        "selected_archive_bytes": sum(item["bytes"] for item in completed_rows),
        "authorization": cfg["authorization"],
    }
    (ROOT / cfg["outputs"]["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": cfg["outputs"]["manifest"], "selected_file_count": len(completed_rows), "selected_archive_bytes": manifest["selected_archive_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
