#!/usr/bin/env python3
"""Download only the five E325 products frozen by the J1 protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/r1_e325_acquisition_jacobian_protocol.json"
UPSTREAM_PATH = ROOT / "results/r1_e325_feasibility/report.json"
DOWNLOADER = ROOT / "scripts/download_parallel_http_file.py"
PROVENANCE_PATH = ROOT / "data/raw/r1_e325_science/provenance.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def remote_headers(url: str) -> dict[str, object]:
    request = Request(url, method="HEAD", headers={"User-Agent": "sigmagravity-observable-audit/0.1"})
    with urlopen(request, timeout=180) as response:
        return {
            "status": response.status,
            "content_length": int(response.headers["Content-Length"]),
            "accept_ranges": response.headers.get("Accept-Ranges", ""),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_disposition": response.headers.get("Content-Disposition"),
            "resolved_url": response.geturl(),
        }


def fits_header_inventory(path: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    with fits.open(
        path,
        mode="readonly",
        memmap=True,
        do_not_scale_image_data=True,
        lazy_load_hdus=False,
    ) as hdul:
        for index, hdu in enumerate(hdul):
            header = hdu.header
            axes = [int(header.get(f"NAXIS{axis}", 0)) for axis in range(1, int(header.get("NAXIS", 0)) + 1)]
            inventory.append(
                {
                    "index": index,
                    "name": str(hdu.name),
                    "xtension": str(header.get("XTENSION", "PRIMARY")),
                    "bitpix": int(header.get("BITPIX", 0)),
                    "axes_fits_order": axes,
                    "bunit": str(header.get("BUNIT", "")),
                    "filter": str(header.get("FILTER", "")),
                    "exptime": float(header["EXPTIME"]) if "EXPTIME" in header else None,
                    "ctype": [str(header.get(f"CTYPE{axis}", "")) for axis in range(1, int(header.get("NAXIS", 0)) + 1)],
                }
            )
    return inventory


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    if not upstream["authorization"]["freeze_acquisition_and_image_level_jacobian_protocol"]:
        raise RuntimeError("E325 feasibility audit did not authorize the frozen acquisition protocol")
    if not config["authorization"]["download_only_the_five_frozen_products"]:
        raise RuntimeError("J1 protocol does not authorize acquisition")
    products = config["acquisition"]["products"]
    if len(products) != 5:
        raise RuntimeError(f"Frozen J1 product count changed: {len(products)}")

    records: list[dict[str, object]] = []
    for number, product in enumerate(products, start=1):
        archive_dir = "hst" if product["archive"] == "HST_MAST" else "eso"
        target = ROOT / config["acquisition"]["output_root"] / archive_dir / product["product_filename"]
        headers = remote_headers(product["url"])
        if headers["content_length"] != int(product["expected_bytes"]):
            raise RuntimeError(
                f"Remote size changed for {product['product_filename']}: "
                f"{headers['content_length']} != {product['expected_bytes']}"
            )
        if str(headers["accept_ranges"]).lower() != "bytes":
            raise RuntimeError(f"Remote does not support immutable range receipt: {product['product_filename']}")
        print(f"E325 download {number}/{len(products)} {product['product_filename']}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(DOWNLOADER),
                product["url"],
                str(target),
                "--workers",
                "8",
                "--chunk-mib",
                "32",
            ],
            check=True,
        )
        if target.stat().st_size != int(product["expected_bytes"]):
            raise RuntimeError(f"Local size mismatch after download: {target}")
        records.append(
            {
                "archive": product["archive"],
                "proposal_id": product["proposal_id"],
                "archive_identifier": product.get("obsid", product.get("product_id")),
                "product_filename": product["product_filename"],
                "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                "url": product["url"],
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "remote_headers": headers,
                "fits_headers": fits_header_inventory(target),
                "role": product["role"],
            }
        )

    provenance = {
        "provenance_version": config["protocol_version"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_download": True,
        "science_arrays_seen_before_protocol_freeze": False,
        "science_arrays_inspected_during_receipt": False,
        "fits_headers_inspected_after_hash_receipt": True,
        "protocol": {
            "path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(CONFIG_PATH),
        },
        "upstream_report": {
            "path": str(UPSTREAM_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(UPSTREAM_PATH),
        },
        "expected_products": 5,
        "received_products": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "records": records,
    }
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "received_products": provenance["received_products"],
                "total_bytes": provenance["total_bytes"],
                "provenance": str(PROVENANCE_PATH.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
