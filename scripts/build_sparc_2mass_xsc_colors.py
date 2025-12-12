#!/usr/bin/env python3
"""Build SPARC galaxy near-IR colors from the 2MASS XSC (extended source catalog).

Goal: get a *stellar-population* color that more directly traces stellar M/L than W1–W2.
We use 2MASS XSC integrated magnitudes (J.ext, H.ext, K.ext) from VizieR (catalog VII/233/xsc).

Outputs:
- resources/photometry/sparc_2mass_xsc_matches.csv

Notes:
- Many low-luminosity dwarfs will not appear in the XSC; those will have match_found=False.
- This script uses coordinates from resources/photometry/sparc_coordinates.csv (built earlier).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Avoid writing tracked __pycache__ artifacts in this repo.
sys.dont_write_bytecode = True


def angular_separation_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: np.ndarray, dec2_deg: np.ndarray) -> np.ndarray:
    ra1 = math.radians(float(ra1_deg))
    dec1 = math.radians(float(dec1_deg))
    ra2r = np.radians(np.asarray(ra2_deg, dtype=float))
    dec2r = np.radians(np.asarray(dec2_deg, dtype=float))

    dra = ra2r - ra1
    ddec = dec2r - dec1
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2r) * np.sin(dra / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.degrees(c) * 3600.0


def query_2mass_xsc_vizier(ra_deg: float, dec_deg: float, radius_arcsec: float, row_limit: int = 50):
    from astroquery.vizier import Vizier
    import astropy.coordinates as coord
    import astropy.units as u

    Vizier.ROW_LIMIT = int(row_limit)
    v = Vizier(columns=["*"])
    catalog = "VII/233/xsc"
    c = coord.SkyCoord(float(ra_deg), float(dec_deg), unit="deg")
    res = v.query_region(c, radius=float(radius_arcsec) * u.arcsec, catalog=catalog)
    if not res:
        return None
    # VizieR returns a dict-like of tables; the first table is the catalog table.
    return res[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="resources/photometry/sparc_coordinates.csv", help="SPARC coordinate CSV")
    ap.add_argument("--out", default="resources/photometry/sparc_2mass_xsc_matches.csv", help="Output CSV")
    ap.add_argument("--radius-arcsec", type=float, default=60.0, help="Search radius around galaxy center")
    ap.add_argument("--row-limit", type=int, default=50, help="Max rows returned by VizieR per query")
    ap.add_argument("--sleep-s", type=float, default=0.2, help="Sleep between queries")
    ap.add_argument("--max-galaxies", type=int, default=0, help="Limit number of galaxies (0 = all)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    coords_path = (repo_root / args.coords).resolve()
    out_path = (repo_root / args.out).resolve()

    coords = pd.read_csv(coords_path)
    if args.max_galaxies and args.max_galaxies > 0:
        coords = coords.head(int(args.max_galaxies)).copy()

    rows: List[Dict[str, Any]] = []

    for _, row in coords.iterrows():
        name = str(row["name"])
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])

        try:
            t = query_2mass_xsc_vizier(ra, dec, radius_arcsec=args.radius_arcsec, row_limit=args.row_limit)
        except Exception as e:
            rows.append(
                {
                    "name": name,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "match_found": False,
                    "n_candidates": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            if args.sleep_s > 0:
                time.sleep(float(args.sleep_s))
            continue

        if t is None or len(t) == 0:
            rows.append(
                {
                    "name": name,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "match_found": False,
                    "n_candidates": 0,
                    "error": "",
                }
            )
            if args.sleep_s > 0:
                time.sleep(float(args.sleep_s))
            continue

        df = t.to_pandas()
        n_candidates = int(len(df))

        # Standardize RA/Dec column names.
        if "RAJ2000" not in df.columns or "DEJ2000" not in df.columns:
            rows.append(
                {
                    "name": name,
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "match_found": False,
                    "n_candidates": n_candidates,
                    "error": "Missing RAJ2000/DEJ2000 columns",
                }
            )
            if args.sleep_s > 0:
                time.sleep(float(args.sleep_s))
            continue

        df["sep_arcsec"] = angular_separation_arcsec(ra, dec, df["RAJ2000"].values, df["DEJ2000"].values)
        best = df.sort_values("sep_arcsec", ascending=True).iloc[0]

        j = best.get("J.ext", np.nan)
        k = best.get("K.ext", np.nan)
        jmk = (float(j) - float(k)) if (np.isfinite(j) and np.isfinite(k)) else np.nan

        rows.append(
            {
                "name": name,
                "ra_deg": ra,
                "dec_deg": dec,
                "match_found": True,
                "n_candidates": n_candidates,
                "2masx": best.get("2MASX"),
                "xsc_ra_deg": float(best.get("RAJ2000")),
                "xsc_dec_deg": float(best.get("DEJ2000")),
                "sep_arcsec": float(best.get("sep_arcsec")),
                "J_ext_mag": float(best.get("J.ext")) if pd.notna(best.get("J.ext")) else np.nan,
                "e_J_ext_mag": float(best.get("e_J.ext")) if pd.notna(best.get("e_J.ext")) else np.nan,
                "H_ext_mag": float(best.get("H.ext")) if pd.notna(best.get("H.ext")) else np.nan,
                "e_H_ext_mag": float(best.get("e_H.ext")) if pd.notna(best.get("e_H.ext")) else np.nan,
                "K_ext_mag": float(best.get("K.ext")) if pd.notna(best.get("K.ext")) else np.nan,
                "e_K_ext_mag": float(best.get("e_K.ext")) if pd.notna(best.get("e_K.ext")) else np.nan,
                "J_minus_K_ext": float(jmk) if np.isfinite(jmk) else np.nan,
                "Sb_over_a": best.get("Sb/a"),
                "Spa_deg": best.get("Spa"),
                "error": "",
            }
        )

        if args.sleep_s > 0:
            time.sleep(float(args.sleep_s))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
