#!/usr/bin/env python3
"""Acquire the frozen DELVE DR3 coadd image, mask, and weight planes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19ax_delve_dr3_coadd_acquisition.json"
USER_AGENT = "SigmaGravity-V19AX-DELVE-DR3-coadd-acquisition/1.0"
PRODUCT_COLUMNS = [
    "band",
    "product",
    "source_reference",
    "source_extension",
    "access_url",
    "output_path",
    "bytes",
    "sha256",
    "height_pixels",
    "width_pixels",
    "pixel_scale_arcsec",
    "finite_fraction",
    "positive_fraction",
    "candidate_positions_inside",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with partial.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def text_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def sia_query_url(config: dict[str, Any]) -> str:
    source = config["source"]
    return source["endpoint"] + "?" + urllib.parse.urlencode(
        {
            "POS": f"{source['center_ra_deg']},{source['center_dec_deg']}",
            "SIZE": source["cutout_size_deg"],
        }
    )


def fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        return response.read()


def classify_product(
    access_url: str, band: str, product: str, config: dict[str, Any]
) -> tuple[str, int] | None:
    parsed = urllib.parse.urlparse(access_url)
    query = urllib.parse.parse_qs(parsed.query)
    references = query.get("siaRef", [])
    extensions = query.get("extn", [])
    if len(references) != 1 or len(extensions) != 1:
        return None
    reference = references[0]
    if config["selection"]["excluded_filename_token"] in reference:
        return None
    if not reference.endswith(f"_{band}.fits.fz"):
        return None
    expected_extension = int(config["selection"]["extension_by_product"][product])
    extension = int(extensions[0])
    if extension != expected_extension:
        return None
    return reference, extension


def select_products(payload: bytes, config: dict[str, Any]) -> list[dict[str, Any]]:
    table = parse_single_table(io.BytesIO(payload)).to_table()
    if len(table) != int(config["gates"]["exact_sia_rows"]):
        raise RuntimeError(f"SIA row count changed: {len(table)}")
    selected: list[dict[str, Any]] = []
    required_bands = set(config["selection"]["bands"])
    required_products = set(config["selection"]["products"])
    for row in table:
        band = text_value(row["obs_bandpass"]).strip()
        product = text_value(row["prodtype"]).strip()
        if band not in required_bands or product not in required_products:
            continue
        access_url = text_value(row["access_url"])
        identity = classify_product(access_url, band, product, config)
        if identity is None:
            continue
        reference, extension = identity
        selected.append(
            {
                "band": band,
                "product": product,
                "source_reference": reference,
                "source_extension": extension,
                "access_url": access_url,
            }
        )
    selected.sort(key=lambda row: (row["band"], row["product"]))
    keys = [(row["band"], row["product"]) for row in selected]
    expected = sorted(
        (band, product)
        for band in config["selection"]["bands"]
        for product in config["selection"]["products"]
    )
    if keys != expected:
        raise RuntimeError(f"DELVE product selection changed: {keys!r}")
    return selected


def candidate_positions(config: dict[str, Any]) -> SkyCoord:
    path = ROOT / config["inputs"]["candidate_hypotheses"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unique: dict[str, tuple[str, str]] = {}
    for row in rows:
        position = (row["candidate_ra_deg"], row["candidate_dec_deg"])
        previous = unique.setdefault(row["candidate_id"], position)
        if previous != position:
            raise RuntimeError(f"candidate coordinates disagree: {row['candidate_id']}")
    if len(unique) != int(config["gates"]["exact_candidates"]):
        raise RuntimeError("candidate count changed")
    return SkyCoord(
        [float(position[0]) for position in unique.values()],
        [float(position[1]) for position in unique.values()],
        unit="deg",
    )


def acquire_product(
    row: dict[str, Any], config: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    path = output_root / f"{row['band']}_{row['product']}.fits"
    if not path.exists():
        payload = fetch(row["access_url"], float(config["source"]["timeout_seconds"]))
        write_atomic(path, payload)
    return {**row, "path": path}


def inspect_product(
    row: dict[str, Any], config: dict[str, Any], candidates: SkyCoord
) -> tuple[dict[str, Any], set[str]]:
    path = row["path"]
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) != 1 or hdul[0].data is None:
            raise RuntimeError(f"unexpected FITS structure: {path}")
        data = np.asarray(hdul[0].data)
        header = hdul[0].header
        if data.ndim != 2:
            raise RuntimeError(f"non-2D DELVE product: {path}")
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            raise RuntimeError(f"missing celestial WCS: {path}")
        scales = np.abs(proj_plane_pixel_scales(wcs) * 3600.0)
        scale = float(np.mean(scales))
        if not np.allclose(
            scales,
            float(config["gates"]["expected_pixel_scale_arcsec"]),
            atol=float(config["gates"]["pixel_scale_tolerance_arcsec"]),
            rtol=0,
        ):
            raise RuntimeError(f"pixel scale changed: {path} {scales}")
        height, width = data.shape
        minimum = int(config["gates"]["minimum_image_dimension_pixels"])
        maximum = int(config["gates"]["maximum_image_dimension_pixels"])
        if not (minimum <= height <= maximum and minimum <= width <= maximum):
            raise RuntimeError(f"unexpected image dimensions: {path} {data.shape}")
        x, y = wcs.world_to_pixel(candidates)
        margin = float(config["gates"]["minimum_candidate_edge_margin_pixels"])
        inside = (x >= margin) & (x <= width - 1 - margin)
        inside &= (y >= margin) & (y <= height - 1 - margin)
        finite = np.isfinite(data)
        finite_fraction = float(np.mean(finite))
        positive_fraction = float(np.mean((data > 0) & finite))
        if finite_fraction < float(config["gates"]["minimum_finite_pixel_fraction"]):
            raise RuntimeError(f"too many non-finite pixels: {path}")
        if row["product"] == "image":
            if not math.isclose(
                float(header.get("MAGZERO", math.nan)),
                float(config["gates"]["expected_image_magzero"]),
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"image MAGZERO changed: {path}")
        elif row["product"] == "weight":
            if positive_fraction < float(
                config["gates"]["minimum_positive_weight_fraction"]
            ):
                raise RuntimeError(f"insufficient positive weights: {path}")
        elif not np.issubdtype(data.dtype, np.integer):
            raise RuntimeError(f"mask plane is not integral: {path}")
        flags = {
            "celestial_wcs",
            "pixel_scale",
            "dimensions",
            "finite_pixels",
        }
        if np.all(inside):
            flags.add("all_candidates_inside")
        return (
            {
                **{key: value for key, value in row.items() if key != "path"},
                "output_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "height_pixels": height,
                "width_pixels": width,
                "pixel_scale_arcsec": scale,
                "finite_fraction": finite_fraction,
                "positive_fraction": positive_fraction,
                "candidate_positions_inside": int(np.sum(inside)),
            },
            flags,
        )


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_full_coadd_pixel_acquisition":
        raise RuntimeError("V19AX acquisition is not frozen")
    prohibited = (
        "measure_anchor_or_candidate_flux",
        "choose_photometry_or_deblend_model",
        "select_or_rank_counterparts",
        "infer_mass_or_current",
        "read_lensing_or_halo_payload",
        "change_gravity_physics_or_parameters",
        "open_holdout",
    )
    if any(config["authorization"][name] for name in prohibited):
        raise RuntimeError("V19AX authorizes a prohibited action")
    for artifact in config["parent_artifacts"]:
        path = ROOT / artifact["path"]
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"parent artifact hash changed: {artifact['path']}")
    runner = ROOT / config["implementation"]["runner"]
    if sha256(runner) != config["implementation"]["runner_sha256"]:
        raise RuntimeError("V19AX runner hash changed")

    outputs = config["outputs"]
    query_url = sia_query_url(config)
    metadata_path = ROOT / outputs["sia_response"]
    query_path = ROOT / outputs["sia_query_url"]
    expected_query = (query_url + "\n").encode()
    if metadata_path.exists() or query_path.exists():
        if not metadata_path.exists() or not query_path.exists():
            raise RuntimeError("partial SIA metadata acquisition exists")
        if query_path.read_bytes() != expected_query:
            raise RuntimeError("frozen SIA query changed")
        payload = metadata_path.read_bytes()
        reused_metadata = True
    else:
        payload = fetch(query_url, float(config["source"]["timeout_seconds"]))
        write_atomic(metadata_path, payload)
        write_atomic(query_path, expected_query)
        reused_metadata = False

    selected = select_products(payload, config)
    candidates = candidate_positions(config)
    output_root = ROOT / outputs["coadd_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    acquired: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config["acquisition"]["workers"])) as pool:
        jobs = {
            pool.submit(acquire_product, row, config, output_root): row for row in selected
        }
        for future in as_completed(jobs):
            acquired.append(future.result())
    acquired.sort(key=lambda row: (row["band"], row["product"]))

    inspected: list[dict[str, Any]] = []
    flags_by_key: dict[str, list[str]] = {}
    for row in acquired:
        product, flags = inspect_product(row, config, candidates)
        inspected.append(product)
        flags_by_key[f"{row['band']}_{row['product']}"] = sorted(flags)
    shapes = {(row["height_pixels"], row["width_pixels"]) for row in inspected}
    gate_results = {
        "exact_sia_rows": len(parse_single_table(io.BytesIO(payload)).to_table())
        == int(config["gates"]["exact_sia_rows"]),
        "exact_products": len(inspected) == int(config["gates"]["exact_products"]),
        "common_shape": len(shapes) == 1,
        "all_candidates_inside_every_product": all(
            row["candidate_positions_inside"] == int(config["gates"]["exact_candidates"])
            for row in inspected
        ),
        "all_integrity_flags": all(
            set(flags)
            >= {"celestial_wcs", "pixel_scale", "dimensions", "finite_pixels"}
            for flags in flags_by_key.values()
        ),
        "no_source_photometry_or_association": True,
    }
    passed = all(gate_results.values())
    manifest_path = ROOT / outputs["product_manifest"]
    write_csv(manifest_path, inspected, PRODUCT_COLUMNS)
    report = {
        "protocol_version": config["protocol_version"],
        "decision": "passed" if passed else "failed_closed",
        "sia": {
            "rows": int(config["gates"]["exact_sia_rows"]),
            "response_sha256": sha256(metadata_path),
            "query_url_sha256": sha256(query_path),
            "reused_verified_metadata": reused_metadata,
        },
        "products": {
            "count": len(inspected),
            "bands": config["selection"]["bands"],
            "planes_per_band": config["selection"]["products"],
            "shapes": [list(shape) for shape in sorted(shapes)],
            "bytes": sum(row["bytes"] for row in inspected),
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(manifest_path),
        },
        "integrity_flags_by_product": flags_by_key,
        "gate_results": gate_results,
        "source_photometry_or_candidate_association_computed": False,
        "claim_boundary": config["claim_boundary"],
    }
    report_path = ROOT / outputs["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
