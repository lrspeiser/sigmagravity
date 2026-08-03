"""Acquire frozen AllWISE W1 cutouts without opening any image arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/p0740_allwise_w1_coverage_supplement.json"
DEFAULT_RAW = ROOT / "data/raw/p0740_allwise_w1_supplement"
DEFAULT_OUTPUT = ROOT / "results/p0740_allwise_w1_coverage_supplement"
USER_AGENT = "sigma-gravity-research/1.0 (scientific reproducibility)"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_url(config: dict[str, Any], system: dict[str, Any], coadd: str, product: str) -> str:
    suffix = "int-3.fits" if product == "intensity" else "unc-3.fits.gz"
    parent = (
        f"{config['source']['dataBaseUrl']}/{coadd[:2]}/{coadd[:4]}/{coadd}/"
        f"{coadd}-w1-{suffix}"
    )
    query = urlencode(
        {
            "center": f"{system['raDeg']:.8f},{system['decDeg']:.8f}deg",
            "size": f"{system['cutoutDiameterDeg']:.8f}deg",
            "gzip": "false",
        }
    )
    return f"{parent}?{query}"


def download(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if partial.exists() and partial.stat().st_size > 0:
        headers["Range"] = f"bytes={partial.stat().st_size}-"
        mode = "ab"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=180) as response:  # noqa: S310 - frozen HTTPS archive URL
        status = int(getattr(response, "status", 200))
        if mode == "ab" and status != 206:
            mode = "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)
    return status


def header_record(path: Path, center: SkyCoord) -> dict[str, Any]:
    # Deliberately request only the primary header. Validation and holdout image
    # arrays remain unopened at this acquisition stage.
    header = fits.getheader(path, 0)
    wcs = WCS(header).celestial
    if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
        raise ValueError(f"{path} does not have a two-dimensional celestial WCS")
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    x_pixel, y_pixel = (float(value) for value in wcs.world_to_pixel(center))
    return {
        "shape": list(shape),
        "naxis1": shape[1],
        "naxis2": shape[0],
        "wcs": wcs,
        "centerPixelX": x_pixel,
        "centerPixelY": y_pixel,
        "centerInsideWcsBounds": bool(
            -0.5 <= x_pixel < shape[1] - 0.5 and -0.5 <= y_pixel < shape[0] - 0.5
        ),
        "ctype1": str(header.get("CTYPE1", "")),
        "ctype2": str(header.get("CTYPE2", "")),
        "bunit": str(header.get("BUNIT", "")),
    }


def requested_square_coordinates(center: SkyCoord, diameter_deg: float, cells: int = 129) -> SkyCoord:
    # Sample equal-area cell centers, not the zero-area mathematical boundary.
    # IBE rounds angular cutouts to whole detector pixels, so evaluating the
    # exact requested edge can label a complete cutout as one pixel short.
    step = diameter_deg / cells
    offsets = np.linspace(
        -diameter_deg / 2.0 + step / 2.0,
        diameter_deg / 2.0 - step / 2.0,
        cells,
    )
    east, north = np.meshgrid(offsets, offsets)
    separation = np.hypot(east, north) * u.deg
    position_angle = np.arctan2(east, north) * u.rad
    return center.directional_offset_by(position_angle.ravel(), separation.ravel())


def union_wcs_coverage(headers: list[dict[str, Any]], coordinates: SkyCoord) -> float:
    covered = np.zeros(len(coordinates), dtype=bool)
    for item in headers:
        x_pixel, y_pixel = item["wcs"].world_to_pixel(coordinates)
        covered |= (
            np.isfinite(x_pixel)
            & np.isfinite(y_pixel)
            & (x_pixel >= -0.5)
            & (x_pixel < item["naxis1"] - 0.5)
            & (y_pixel >= -0.5)
            & (y_pixel < item["naxis2"] - 0.5)
        )
    return float(np.mean(covered))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    system_audit: list[dict[str, Any]] = []
    product_headers: dict[tuple[str, str, str], dict[str, Any]] = {}
    opened_arrays = {"development": 0, "validation": 0, "holdout": 0}

    for system in config["systems"]:
        galaxy = system["id"]
        center = SkyCoord(system["raDeg"], system["decDeg"], unit="deg", frame="icrs")
        intensity_headers: list[dict[str, Any]] = []
        for coadd in system["coaddIds"]:
            for product in config["source"]["products"]:
                url = product_url(config, system, coadd, product)
                filename = f"{coadd}-w1-{product}-cutout.fits"
                destination = args.raw / galaxy / filename
                if destination.exists() and destination.stat().st_size > 0:
                    status = 200
                    reused = True
                else:
                    status = download(url, destination)
                    reused = False
                item = header_record(destination, center)
                product_headers[(galaxy, coadd, product)] = item
                if product == "intensity":
                    intensity_headers.append(item)
                records.append(
                    {
                        "galaxy": galaxy,
                        "split": system["split"],
                        "coadd_id": coadd,
                        "product": product,
                        "relative_path": destination.relative_to(ROOT).as_posix(),
                        "url": url,
                        "http_status": status,
                        "reused_existing_file": reused,
                        "bytes": destination.stat().st_size,
                        "sha256": file_sha256(destination),
                        "naxis1": item["naxis1"],
                        "naxis2": item["naxis2"],
                        "ctype1": item["ctype1"],
                        "ctype2": item["ctype2"],
                        "bunit": item["bunit"],
                        "center_pixel_x": item["centerPixelX"],
                        "center_pixel_y": item["centerPixelY"],
                        "center_inside_wcs_bounds": item["centerInsideWcsBounds"],
                        "array_opened": 0,
                    }
                )
                print(f"{galaxy} {coadd} {product}: {destination.stat().st_size:,} bytes")

        requested_coordinates = requested_square_coordinates(center, system["cutoutDiameterDeg"])
        coverage = union_wcs_coverage(intensity_headers, requested_coordinates)
        center_inside = any(item["centerInsideWcsBounds"] for item in intensity_headers)
        shapes_match = all(
            product_headers[(galaxy, coadd, "intensity")]["shape"]
            == product_headers[(galaxy, coadd, "uncertainty")]["shape"]
            for coadd in system["coaddIds"]
        )
        system_audit.append(
            {
                "galaxy": galaxy,
                "split": system["split"],
                "coadds": len(system["coaddIds"]),
                "cutout_diameter_deg": system["cutoutDiameterDeg"],
                "center_inside_at_least_one_wcs": center_inside,
                "union_wcs_footprint_fraction": coverage,
                "intensity_uncertainty_shapes_match": shapes_match,
                "arrays_opened": 0,
            }
        )

    files = pd.DataFrame(records)
    systems = pd.DataFrame(system_audit)
    files.to_csv(args.output / "file_manifest.csv", index=False)
    systems.to_csv(args.output / "system_audit.csv", index=False)

    rules = config["acquisitionRules"]
    gates = config["acquisitionGates"]
    split_counts = systems.groupby("split").size().to_dict()
    checks = {
        "requiredSystems": len(systems) == int(rules["requiredSystems"]),
        "requiredDevelopmentSystems": split_counts.get("development", 0)
        == int(rules["requiredDevelopmentSystems"]),
        "requiredValidationSystems": split_counts.get("validation", 0)
        == int(rules["requiredValidationSystems"]),
        "requiredHoldoutSystems": split_counts.get("holdout", 0)
        == int(rules["requiredHoldoutSystems"]),
        "requiredCoadds": int(systems.coadds.sum()) == int(rules["requiredCoadds"]),
        "requiredFiles": len(files) == int(rules["requiredFiles"]),
        "allHttpStatus200": bool(files.http_status.eq(200).all()),
        "allSha256Recorded": bool(files.sha256.str.len().eq(64).all()),
        "allFitsPrimaryHdusReadable": bool((files.naxis1 > 0).all() and (files.naxis2 > 0).all()),
        "allFitsHaveTwoCelestialAxes": bool(
            files.ctype1.str.contains("RA").all() and files.ctype2.str.contains("DEC").all()
        ),
        "everySystemCenterInsideAtLeastOneImageWcsBounds": bool(
            systems.center_inside_at_least_one_wcs.all()
        ),
        "minimumUnionWcsFootprintFractionOfRequestedSquare": float(
            systems.union_wcs_footprint_fraction.min()
        )
        >= float(gates["minimumUnionWcsFootprintFractionOfRequestedSquare"]),
        "allIntensityAndUncertaintyShapesMatch": bool(
            systems.intensity_uncertainty_shapes_match.all()
        ),
        "requiredValidationArraysOpened": opened_arrays["validation"]
        == int(gates["requiredValidationArraysOpened"]),
        "requiredHoldoutArraysOpened": opened_arrays["holdout"]
        == int(gates["requiredHoldoutArraysOpened"]),
        "requiredVelocityOrDispersionArraysOpened": 0
        == int(gates["requiredVelocityOrDispersionArraysOpened"]),
        "maximumGravityParameters": 0 <= int(gates["maximumGravityParameters"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    report = {
        "schemaVersion": "sigma-p0740-allwise-w1-coverage-supplement-result/1",
        "stage": "P0740",
        "status": status,
        "configSha256": file_sha256(args.config),
        "systems": len(systems),
        "coadds": int(systems.coadds.sum()),
        "files": len(files),
        "bytes": int(files.bytes.sum()),
        "arraysOpened": opened_arrays,
        "velocityOrDispersionArraysOpened": 0,
        "gravityParameters": 0,
        "checks": checks,
        "aggregate": {
            "minimumUnionWcsFootprintFraction": float(
                systems.union_wcs_footprint_fraction.min()
            ),
            "maximumUnionWcsFootprintFraction": float(
                systems.union_wcs_footprint_fraction.max()
            ),
        },
        "claimBoundary": config["claimBoundary"],
    }
    report["reportSha256"] = canonical_sha256(report)
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0740 AllWISE W1 coverage supplement

Status: **{status.upper()}**

- Systems: {len(systems)}
- Frozen coadds: {int(systems.coadds.sum())}
- Downloaded FITS cutouts: {len(files)}
- Total bytes: {int(files.bytes.sum()):,}
- Minimum header-only WCS union coverage: {100.0 * systems.union_wcs_footprint_fraction.min():.2f}%
- Development image arrays opened: 0
- Validation image arrays opened: 0
- Holdout image arrays opened: 0
- Velocity or dispersion arrays opened: 0
- Gravity parameters: 0
- Report SHA-256: `{report['reportSha256']}`

This acquisition repairs stellar-light footprint coverage. It does not score a gravity formula.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
