#!/usr/bin/env python3
"""Download the exact RX J2129 XMM observation authorized by R1B3-P1."""

from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import astropy.units as u
import requests
from astropy.coordinates import SkyCoord
from astroquery.heasarc import Heasarc


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/r1_rxj2129_strict_observable_next_stage.json"
GATE = ROOT / "results/r1_rxj2129_strict_observable_feasibility/report.json"
RAW = ROOT / "data/raw/r1_rxj2129_xmm"
PROVENANCE = RAW / "provenance.json"
S3_BUCKET_URL = "https://nasa-heasarc.s3.amazonaws.com"
S3_PREFIX = "xmm/data/rev0/0093030201/"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def list_s3_objects(prefix: str) -> list[dict]:
    objects = []
    continuation = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation:
            params["continuation-token"] = continuation
        response = requests.get(f"{S3_BUCKET_URL}/?{urlencode(params)}", timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s3:Contents", namespace):
            objects.append({
                "key": item.findtext("s3:Key", namespaces=namespace),
                "bytes": int(item.findtext("s3:Size", namespaces=namespace)),
                "etag": item.findtext("s3:ETag", namespaces=namespace).strip('"'),
            })
        truncated = root.findtext("s3:IsTruncated", namespaces=namespace) == "true"
        if not truncated:
            break
        continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
    return objects


def download_s3_object(item: dict) -> Path:
    relative = item["key"].split("xmm/data/rev0/", 1)[1]
    target = RAW / relative
    if target.exists() and target.stat().st_size == item["bytes"]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    url = f"{S3_BUCKET_URL}/{quote(item['key'], safe='/')}"
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    if temporary.stat().st_size != item["bytes"]:
        raise RuntimeError(f"S3 size mismatch for {item['key']}")
    temporary.replace(target)
    return target


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate["authorization"]["download_exact_XMM_observation"]:
        raise RuntimeError("R1B3-P1 did not authorize XMM acquisition")
    expected = config["xmm_metadata_gate"]["required_obsid"]
    center = SkyCoord(config["center_ra_deg"] * u.deg, config["center_dec_deg"] * u.deg)
    heasarc = Heasarc()
    heasarc.timeout = 1800
    table = heasarc.query_region(center, catalog=config["xmm_metadata_gate"]["catalog"], radius=12 * u.arcmin)
    selected = table[[str(value) == expected for value in table["obsid"]]]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {expected} row, found {len(selected)}")
    links = heasarc.locate_data(selected)
    if len(links) != 1 or int(links[0]["content_length"]) > config["xmm_metadata_gate"]["maximum_located_data_bytes"]:
        raise RuntimeError("Located XMM data changed after the frozen P1 gate")
    RAW.mkdir(parents=True, exist_ok=True)
    objects = list_s3_objects(S3_PREFIX)
    manifest_bytes = sum(item["bytes"] for item in objects)
    located_bytes = int(links[0]["content_length"])
    if (
        not objects
        or manifest_bytes > config["xmm_metadata_gate"]["maximum_located_data_bytes"]
        or manifest_bytes < 0.8 * located_bytes
    ):
        raise RuntimeError("Public S3 object manifest fails the frozen size/completeness gate")
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download_s3_object, objects))
    files = sorted(path for path in RAW.rglob("*") if path.is_file() and path != PROVENANCE)
    if not files:
        raise RuntimeError("HEASARC download produced no local files")
    records = []
    remote = {item["key"].split("xmm/data/rev0/", 1)[1]: item for item in objects}
    for path in files:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        archive_relative = str(path.relative_to(RAW)).replace("\\", "/")
        item = remote[archive_relative]
        local_md5 = digest(path, "md5")
        etag_is_md5 = "-" not in item["etag"]
        if etag_is_md5 and local_md5.lower() != item["etag"].lower():
            raise RuntimeError(f"S3 ETag/MD5 mismatch for {archive_relative}")
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "md5": local_md5,
            "sha256": digest(path, "sha256"),
            "archive_s3_etag": item["etag"],
            "archive_s3_etag_verified_as_md5": etag_is_md5,
        })
    names = [record["path"].lower() for record in records]
    provenance = {
        "provenance_version": config["protocol_version"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_download": True,
        "obsid": expected,
        "archive_catalog": config["xmm_metadata_gate"]["catalog"],
        "archive_row": {key: scalar(selected[0][key]) for key in selected.colnames},
        "located_data": {key: scalar(links[0][key]) for key in links.colnames},
        "download_route": "public NASA HEASARC AWS S3 object tree after the HEASARC tar endpoint produced no bytes",
        "s3_prefix": S3_PREFIX,
        "s3_manifest_objects": len(objects),
        "s3_manifest_bytes": manifest_bytes,
        "heasarc_located_data_bytes": located_bytes,
        "s3_to_heasarc_located_size_ratio": manifest_bytes / located_bytes,
        "archive_packaging_size_note": "HEASARC locate_data reports an aggregate transfer estimate, while the public S3 tree is an enumerated object manifest. Both remain below the pre-frozen 1-GB ceiling; ODF and PPS presence is verified separately.",
        "XMM_pixels_inspected": False,
        "local_files": len(records),
        "local_bytes": sum(record["bytes"] for record in records),
        "contains_odf_or_raw_products": any("odf" in name or "raw" in name for name in names),
        "contains_pps_or_pipeline_products": any("pps" in name or "pipeline" in name for name in names),
        "records": records,
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: provenance[key] for key in (
        "obsid", "XMM_pixels_inspected", "local_files", "local_bytes",
        "contains_odf_or_raw_products", "contains_pps_or_pipeline_products"
    )}, indent=2))


if __name__ == "__main__":
    main()
