#!/usr/bin/env python3
"""Run the frozen metadata-only P0566 transfer feasibility gate."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0566_a383_ms2137_transfer_feasibility_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def json_value(value):
    if value is None:
        return None
    if hasattr(value, "mask") and bool(value.mask):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (float, int, bool, str)):
        return value
    return str(value)


def angular_separation_deg(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def hst_url(protocol, system, product):
    root = protocol["metadata_queries"]["hst"]["official_root"]
    slug = system["archive_slug"]
    filename = f"hlsp_clash_hst_wfc3ir_{slug}_f160w_v1_{product}.fits"
    return (
        f"{root}/{slug}/data/hst/scale_65mas/{filename}",
        filename,
    )


def query_chandra(protocol, system):
    endpoint = protocol["metadata_queries"]["chandra"]["official_tap"]
    radius = float(protocol["metadata_queries"]["chandra"]["cone_radius_deg"])
    ra = float(system["center_ra_deg"])
    dec = float(system["center_dec_deg"])
    ra_pad = radius / max(math.cos(math.radians(dec)), 0.5) + 0.01
    query = f"""SELECT obsid,target_name,ra,dec,instrument,grating,status,
start_date,public_avail_date,exposure_time,exposure_mode,event_count
FROM cxc.observation
WHERE ra BETWEEN {ra - ra_pad:.12f} AND {ra + ra_pad:.12f}
AND dec BETWEEN {dec - radius:.12f} AND {dec + radius:.12f}"""
    response = requests.post(
        endpoint,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query},
        timeout=60,
    )
    response.raise_for_status()
    table = Table.read(BytesIO(response.content), format="votable")
    rows = []
    for item in table:
        row = {name: json_value(item[name]) for name in table.colnames}
        row["angular_separation_deg"] = angular_separation_deg(
            ra,
            dec,
            float(row["ra"]),
            float(row["dec"]),
        )
        grating = str(row.get("grating") or "").strip().lower()
        instrument = str(row.get("instrument") or "").strip().upper()
        status = str(row.get("status") or "").strip().lower()
        row["passes_frozen_selection"] = bool(
            row["angular_separation_deg"] <= radius
            and instrument.startswith("ACIS")
            and grating in {"", "none"}
            and status == "archived"
        )
        rows.append(row)
    return query, sorted(rows, key=lambda item: int(item["obsid"]))


def main():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not protocol["status"].startswith(
        "frozen_before_remote_archive_metadata_query"
    ):
        raise RuntimeError("P0566 was not frozen before its metadata queries")

    lens = pd.read_csv(ROOT / protocol["metadata_queries"]["local_lensing"])
    accept_lines = (
        ROOT / protocol["metadata_queries"]["local_accept"]
    ).read_text(encoding="utf-8", errors="replace").splitlines()
    baryon_lines = (
        ROOT / protocol["metadata_queries"]["local_baryons"]
    ).read_text(encoding="utf-8", errors="replace").splitlines()
    gates = protocol["advance_gates"]
    inventory = []
    all_checks = []

    for system in protocol["systems"]:
        label = system["label"]
        selected = lens[
            lens.system.eq(label)
            & lens.alternative_metric_likelihood_ready.astype(bool)
        ].copy()
        family_sizes = (
            selected.groupby("source_family").size().sort_index().astype(int).tolist()
        )
        accept_shells = sum(
            line.startswith(system["accept_name"] + " ") for line in accept_lines
        )
        baryon_matches = [
            line for line in baryon_lines if line.rstrip().endswith("|" + label)
        ]

        hst_rows = []
        for product in protocol["metadata_queries"]["hst"]["products"]:
            url, filename = hst_url(protocol, system, product)
            response = requests.head(url, allow_redirects=True, timeout=60)
            length = int(response.headers.get("Content-Length", 0))
            hst_rows.append(
                {
                    "product": product,
                    "filename": filename,
                    "url": url,
                    "status_code": int(response.status_code),
                    "content_length": length,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
            )

        query, chandra_candidates = query_chandra(protocol, system)
        chandra_selected = [
            row for row in chandra_candidates if row["passes_frozen_selection"]
        ]
        exposure = sum(float(row["exposure_time"]) for row in chandra_selected)
        checks = {
            "likelihood_ready_images": len(selected)
            >= int(gates["minimum_likelihood_ready_images_each"]),
            "source_families": selected.source_family.nunique()
            >= int(gates["minimum_source_families_each"]),
            "images_per_family": min(family_sizes)
            >= int(gates["minimum_images_per_family"]),
            "expected_catalog_counts_unchanged": (
                len(selected) == int(system["expected_likelihood_ready_images"])
                and selected.source_family.nunique()
                == int(system["expected_source_families"])
                and family_sizes == list(map(int, system["expected_family_sizes"]))
            ),
            "accept_shells": accept_shells
            >= int(gates["minimum_accept_shells_each"]),
            "tian_baryon_anchor": len(baryon_matches) == 1,
            "hst_products": (
                len(hst_rows) == int(gates["required_hst_products_each"])
                and all(row["status_code"] == 200 for row in hst_rows)
                and all(
                    row["content_length"]
                    >= int(gates["minimum_hst_content_length_bytes"])
                    for row in hst_rows
                )
            ),
            "public_chandra_observations": len(chandra_selected)
            >= int(gates["minimum_public_chandra_observations_each"]),
            "public_chandra_exposure": exposure
            >= float(gates["minimum_public_chandra_exposure_ks_each"]),
        }
        all_checks.extend(checks.values())
        inventory.append(
            {
                "system_label": label,
                "lensing": {
                    "likelihood_ready_images": int(len(selected)),
                    "source_families": int(selected.source_family.nunique()),
                    "family_sizes": family_sizes,
                    "image_ids": selected.image_id.astype(str).tolist(),
                },
                "accept_shells": int(accept_shells),
                "tian_baryon_row": baryon_matches[0] if baryon_matches else None,
                "hst": hst_rows,
                "chandra": {
                    "tap_query": query,
                    "candidates": chandra_candidates,
                    "selected_obsids": [int(row["obsid"]) for row in chandra_selected],
                    "selected_exposure_ks": exposure,
                },
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
        )

    exact_targets = len(inventory) == int(gates["exact_selected_systems"])
    passed = bool(exact_targets and all(all_checks))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "inventory_version": "P0566-A383-MS2137-METADATA-INVENTORY-0.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(CONFIG),
        "systems": inventory,
    }
    (output / protocol["outputs"]["metadata_inventory"]).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "report_version": "P0566-A383-MS2137-TRANSFER-FEASIBILITY-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {
            "path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(CONFIG),
        },
        "system_summaries": [
            {
                "system_label": row["system_label"],
                "lensing_images": row["lensing"]["likelihood_ready_images"],
                "source_families": row["lensing"]["source_families"],
                "accept_shells": row["accept_shells"],
                "hst_content_lengths": [
                    item["content_length"] for item in row["hst"]
                ],
                "chandra_obsids": row["chandra"]["selected_obsids"],
                "chandra_exposure_ks": row["chandra"]["selected_exposure_ks"],
                "passed": row["passed"],
            }
            for row in inventory
        ],
        "gate_audit": {
            "exact_selected_systems": exact_targets,
            "all_system_checks": bool(all(all_checks)),
            "feasibility_passed": passed,
        },
        "decision": (
            "authorize_frozen_pixel_acquisition_protocol"
            if passed
            else "stop_transfer_due_to_data_shortfall"
        ),
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# P0566 A383 + MS2137 transfer feasibility", ""]
    for row in report["system_summaries"]:
        lines.append(
            f"- {row['system_label']}: {row['lensing_images']} images / "
            f"{row['source_families']} families, {row['accept_shells']} ACCEPT shells, "
            f"Chandra ObsIDs {row['chandra_obsids']} totaling "
            f"{row['chandra_exposure_ks']:.3f} ks; passed={row['passed']}."
        )
    lines.extend(
        [
            "",
            f"Feasibility passed: **{passed}**. This is archive coverage only; no "
            "science pixel or gravity outcome was inspected.",
        ]
    )
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
