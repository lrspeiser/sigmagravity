#!/usr/bin/env python3
"""Build SAS-standard FITS region tables for the immutable 87-source mask."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


INSTRUMENTS = ("MOS1", "MOS2", "pn")


def read_circles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 4 or fields[0] != "!CIRCLE":
            raise ValueError(f"invalid circle at {path}:{line_number}")
        rows.append(tuple(float(value) for value in fields[1:]))
    if len(rows) != 87:
        raise ValueError(f"expected 87 circles in {path}, found {len(rows)}")
    values = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(values).all() or not np.all(values[:, 2] > 0):
        raise ValueError(f"non-finite or non-positive circle in {path}")
    return values[:, 0], values[:, 1], values[:, 2]


def write_region(path: Path, axis_names: tuple[str, str], source: Path) -> None:
    first, second, radius = read_circles(source)
    count = len(first)
    shape = np.full(count, "!CIRCLE", dtype="S16")
    first4 = np.zeros((count, 4), dtype=np.float32)
    second4 = np.zeros((count, 4), dtype=np.float32)
    radius4 = np.zeros((count, 4), dtype=np.float32)
    rotation4 = np.zeros((count, 4), dtype=np.float32)
    first4[:, 0] = first
    second4[:, 0] = second
    radius4[:, 0] = radius
    columns = fits.ColDefs(
        [
            fits.Column(name="SHAPE", format="16A", array=shape),
            fits.Column(name=axis_names[0], format="4E", array=first4),
            fits.Column(name=axis_names[1], format="4E", array=second4),
            fits.Column(name="R", format="4E", array=radius4),
            fits.Column(name="ROTANG", format="4E", array=rotation4),
            fits.Column(
                name="COMPONENT", format="J", array=np.ones(count, dtype=np.int32)
            ),
        ]
    )
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    table = fits.BinTableHDU.from_columns(columns, name="REGION")
    table.header["HDUVERS"] = "1.0.0"
    table.header["HDUCLASS"] = "ASC"
    table.header["HDUCLAS1"] = "REGION"
    table.header["HDUCLAS2"] = "STANDARD"
    table.header["MTYPE1"] = "pos"
    table.header["MFORM1"] = f"{axis_names[0]},{axis_names[1]}"
    table.header["CREATOR"] = "build_r1_rxj2129_xmm_fits_source_masks.py"
    table.header["DATE"] = created
    primary = fits.PrimaryHDU()
    primary.header["CREATOR"] = "build_r1_rxj2129_xmm_fits_source_masks.py"
    primary.header["DATE"] = created
    fits.HDUList([primary, table]).writeto(path, overwrite=True, checksum=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.background_root
    for instrument in INSTRUMENTS:
        write_region(
            root / f"{instrument}_point_source_mask_detector.fits",
            ("DETX", "DETY"),
            root / f"{instrument}_point_source_mask_detector.txt",
        )
        write_region(
            root / f"{instrument}_point_source_mask_sky.fits",
            ("X", "Y"),
            root / f"{instrument}_point_source_mask_sky_xy.txt",
        )
    print(root)


if __name__ == "__main__":
    main()
