#!/usr/bin/env python3
"""Write the immutable source catalog in convregion mode-2 ASCII syntax."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT / "data/derived/r1_rxj2129_xmm_x2/point_source_catalog.csv"
OUTPUT = PROJECT / "data/derived/r1_rxj2129_xmm_x2/point_source_mask_convregion_sky.txt"


def main() -> None:
    with CATALOG.open() as handle:
        rows = list(csv.DictReader(handle))
    with OUTPUT.open("w") as handle:
        for row in rows:
            radius_arcmin = float(row["mask_radius_arcsec"]) / 60.0
            handle.write(
                f'!CIRCLE {float(row["ra_deg"]):.10f} {float(row["dec_deg"]):.10f} {radius_arcmin:.10f}\n'
            )
    if len(rows) != 87:
        raise RuntimeError(f"expected 87 immutable sources, found {len(rows)}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
