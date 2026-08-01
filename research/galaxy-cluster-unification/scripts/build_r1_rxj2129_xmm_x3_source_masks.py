#!/usr/bin/env python3
"""Derive X3 masks that retain the predeclared RX J2129 target center."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_x2/point_source_catalog.csv"
MANIFEST_PATH = PROJECT / "data/derived/r1_rxj2129_xmm_x3_source_mask_manifest.json"
PROTOCOL_PATH = PROJECT / "configs/r1_rxj2129_xmm_x3_annular_protocol.json"
LINUX_BACKGROUND_ROOT = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/"
    "x2b/background"
)
WINDOWS_BACKGROUND_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x2b/background"
)
LINUX_OUTPUT_ROOT = Path(
    "/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/masks"
)
WINDOWS_OUTPUT_ROOT = Path(
    "//wsl.localhost/Ubuntu-24.04/home/henry/.local/share/sigmagravity-xmm/"
    "work/rxj2129/0093030201/x3/masks"
)
INSTRUMENTS = ("MOS2", "pn")
CENTER_RA_DEG = 322.41651
CENTER_DEC_DEG = 0.08923


def parse_args() -> argparse.Namespace:
    background = WINDOWS_BACKGROUND_ROOT if os.name == "nt" else LINUX_BACKGROUND_ROOT
    output = WINDOWS_OUTPUT_ROOT if os.name == "nt" else LINUX_OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-root", type=Path, default=background)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def separation_arcsec(ra_deg: float, dec_deg: float) -> float:
    dra = (ra_deg - CENTER_RA_DEG) * math.cos(math.radians(CENTER_DEC_DEG))
    return 3600.0 * math.hypot(dra, dec_deg - CENTER_DEC_DEG)


def read_ascii_circles(path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[0] != "!CIRCLE":
            raise ValueError(f"invalid source-mask row in {path}: {line}")
        rows.append(tuple(float(item) for item in fields[1:]))
    values = np.asarray(rows, dtype=float)
    if values.shape != (87, 3) or not np.isfinite(values).all():
        raise ValueError(f"expected finite 87x3 source-mask table in {path}")
    return values


def filter_fits_region(
    source: Path,
    destination: Path,
    selected: np.ndarray,
    ascii_values: np.ndarray,
    first_axis: str,
    second_axis: str,
    created: str,
) -> dict[str, Any]:
    with fits.open(source, memmap=False) as hdul:
        if hdul[1].name != "REGION" or len(hdul[1].data) != 87:
            raise ValueError(f"unexpected original FITS region table {source}")
        table = hdul[1].data
        if not np.allclose(table[first_axis][:, 0], ascii_values[:, 0], atol=1e-3):
            raise ValueError(f"{source} first axis does not match authoritative ASCII")
        if not np.allclose(table[second_axis][:, 0], ascii_values[:, 1], atol=1e-3):
            raise ValueError(f"{source} second axis does not match authoritative ASCII")
        if not np.allclose(table["R"][:, 0], ascii_values[:, 2], atol=1e-3):
            raise ValueError(f"{source} radii do not match authoritative ASCII")
        primary_header = hdul[0].header.copy()
        region_header = hdul[1].header.copy()
        filtered_data = table[selected].copy()

    creator = "build_r1_rxj2129_xmm_x3_source_masks.py"
    primary_header["CREATOR"] = creator
    primary_header["DATE"] = created
    primary_header["HISTORY"] = "Removed only source_id 50 because its circle contains the frozen target center."
    region_header["CREATOR"] = creator
    region_header["DATE"] = created
    region_header["HISTORY"] = (
        "X3 target-aware mask: frozen geometric subset of 86 noncentral exclusions."
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(header=primary_header),
            fits.BinTableHDU(data=filtered_data, header=region_header, name="REGION"),
        ]
    ).writeto(destination, overwrite=True, checksum=True)
    with fits.open(destination, memmap=False) as hdul:
        assert len(hdul["REGION"].data) == int(np.count_nonzero(selected))
        if len(hdul["REGION"].data):
            assert str(hdul["REGION"].data[0]["SHAPE"]).strip() in (
                "!CIRCLE",
                "b'!CIRCLE'",
            )
    return {
        "original": str(source),
        "original_sha256": sha256(source),
        "derived": str(destination),
        "derived_sha256": sha256(destination),
        "original_rows": 87,
        "derived_rows": int(np.count_nonzero(selected)),
    }


def main() -> None:
    args = parse_args()
    with CATALOG_PATH.open() as handle:
        catalog = list(csv.DictReader(handle))
    if len(catalog) != 87:
        raise ValueError(f"expected 87 catalog detections, found {len(catalog)}")
    overlaps = []
    for index, row in enumerate(catalog):
        separation = separation_arcsec(float(row["ra_deg"]), float(row["dec_deg"]))
        radius = float(row["mask_radius_arcsec"])
        if separation <= radius:
            overlaps.append((index, row, separation, radius))
    if len(overlaps) != 1:
        raise ValueError(f"expected exactly one center-overlap detection, found {len(overlaps)}")
    central_index, central, separation, radius = overlaps[0]
    if central["source_id"] != "50":
        raise ValueError(f"expected central source_id 50, found {central['source_id']}")
    central_separation = separation
    central_radius = radius
    selected = np.ones(87, dtype=bool)
    selected[central_index] = False
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    products: dict[str, Any] = {}
    for instrument in INSTRUMENTS:
        instrument_products = {}
        for coordinate, axes, ascii_suffix in (
            ("detector", ("DETX", "DETY"), "detector.txt"),
            ("sky", ("X", "Y"), "sky_xy.txt"),
        ):
            ascii_path = args.background_root / f"{instrument}_point_source_mask_{ascii_suffix}"
            fits_path = args.background_root / f"{instrument}_point_source_mask_{coordinate}.fits"
            destination = args.output_root / f"{instrument}_point_source_mask_{coordinate}.fits"
            instrument_products[coordinate] = filter_fits_region(
                fits_path,
                destination,
                selected,
                read_ascii_circles(ascii_path),
                axes[0],
                axes[1],
                created,
            )
        products[instrument] = instrument_products

    protocol = json.loads(PROTOCOL_PATH.read_text())
    geometry = protocol["fixed_geometry"]
    edges_arcsec = geometry["radial_edges_arcsec"]
    annular_products: dict[str, Any] = {}
    for annulus_index, annulus_id in enumerate(geometry["annulus_ids"]):
        inner_arcsec = float(edges_arcsec[annulus_index])
        outer_arcsec = float(edges_arcsec[annulus_index + 1])
        intersects = np.zeros(87, dtype=bool)
        intersecting_ids = []
        for index, row in enumerate(catalog):
            if not selected[index]:
                continue
            separation = separation_arcsec(float(row["ra_deg"]), float(row["dec_deg"]))
            radius = float(row["mask_radius_arcsec"])
            if separation - radius < outer_arcsec and separation + radius > inner_arcsec:
                intersects[index] = True
                intersecting_ids.append(int(row["source_id"]))
        annulus_entry: dict[str, Any] = {
            "radial_range_arcsec": [inner_arcsec, outer_arcsec],
            "intersecting_source_ids": intersecting_ids,
            "exclusion_count": len(intersecting_ids),
            "geometric_equivalence": (
                "Every omitted catalog circle is disjoint from the closed annular support; "
                "therefore the compact table and the full 86-circle X3 table select identical "
                "events inside this annulus."
            ),
            "products": {},
        }
        for instrument in INSTRUMENTS:
            instrument_products = {}
            for coordinate, axes, ascii_suffix in (
                ("detector", ("DETX", "DETY"), "detector.txt"),
                ("sky", ("X", "Y"), "sky_xy.txt"),
            ):
                ascii_path = args.background_root / f"{instrument}_point_source_mask_{ascii_suffix}"
                fits_path = args.background_root / f"{instrument}_point_source_mask_{coordinate}.fits"
                destination = (
                    args.output_root
                    / annulus_id
                    / f"{instrument}_point_source_mask_{coordinate}.fits"
                )
                instrument_products[coordinate] = filter_fits_region(
                    fits_path,
                    destination,
                    intersects,
                    read_ascii_circles(ascii_path),
                    axes[0],
                    axes[1],
                    created,
                )
                instrument_products[coordinate]["derived_rows"] = len(intersecting_ids)
            annulus_entry["products"][instrument] = instrument_products
        annular_products[annulus_id] = annulus_entry

    manifest = {
        "manifest_version": "R1B3-RXJ2129-XMM-X3-source-mask-0.3",
        "generated_utc": created,
        "catalog": CATALOG_PATH.relative_to(PROJECT).as_posix(),
        "catalog_sha256": sha256(CATALOG_PATH),
        "catalog_detection_count_unchanged": 87,
        "X3_exclusion_count": 86,
        "central_target_component": {
            "source_id": int(central["source_id"]),
            "ra_deg": float(central["ra_deg"]),
            "dec_deg": float(central["dec_deg"]),
            "separation_from_frozen_center_arcsec": central_separation,
            "original_mask_radius_arcsec": central_radius,
            "detection_likelihood": float(central["maximum_detection_likelihood"]),
            "action": "retain in X3 source spectra and require an unresolved central-emission nuisance in the later spectral likelihood",
        },
        "selection_rule": "Remove from the exclusion mask every catalog circle containing the predeclared target center; exactly source_id 50 meets this rule. No radial count, temperature, density, or residual chooses the source.",
        "products": products,
        "annular_compact_masks": annular_products,
        "gates": {
            "exactly_one_center_overlap_detection": True,
            "central_source_is_source_id_50": True,
            "all_original_tables_have_87_rows": True,
            "all_X3_tables_have_86_rows": True,
            "all_compact_masks_are_valid_subsets_of_86": all(
                0 <= item["exclusion_count"] <= 86 for item in annular_products.values()
            ),
            "X3_target_aware_source_mask_gate_passed": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
