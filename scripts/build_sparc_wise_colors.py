#!/usr/bin/env python3
"""Build SPARC galaxy coordinates + AllWISE W1/W2 colors.

Outputs (in --out-dir):
- sparc_coordinates.csv
- sparc_allwise_matches.csv

Design goals:
- Works with either the IRSA AllWISE parquet dataset on S3 (default) or a fully-downloaded local copy.
- Uses the SPARC file list as the canonical galaxy list (no hand-maintained names).
- Produces a small, analysis-ready table (no giant catalogs committed to git).

Notes:
- SPARC rotmod files do not include RA/Dec.
- We resolve galaxy coordinates using NED first, then SIMBAD.
- For photometry we prefer extended-source measurements (ext_flg>0) and use W1/W2 elliptical aperture mags (w1gmag/w2gmag).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GalaxyCoord:
    name: str
    ra_deg: float
    dec_deg: float
    resolver: str
    resolved_name: str


def normalize_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "")


def list_sparc_galaxy_names(sparc_dir: Path) -> List[str]:
    files = sorted(sparc_dir.glob("*_rotmod.dat"))
    names = [f.stem.replace("_rotmod", "") for f in files]
    # Keep deterministic order.
    return names


def resolve_coords_ned(name: str) -> Optional[GalaxyCoord]:
    from astroquery.ipac.ned import Ned

    try:
        t = Ned.query_object(name)
        if t is None or len(t) == 0:
            return None
        ra = float(t["RA"][0])
        dec = float(t["DEC"][0])
        resolved = str(t["Object Name"][0])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return None
        return GalaxyCoord(name=name, ra_deg=ra, dec_deg=dec, resolver="NED", resolved_name=resolved)
    except Exception:
        # Common SPARC alias: many F* galaxies are LSBC entries in NED.
        if str(name).upper().startswith("F"):
            try:
                alt = f"LSBC {name}"
                t = Ned.query_object(alt)
                if t is None or len(t) == 0:
                    return None
                ra = float(t["RA"][0])
                dec = float(t["DEC"][0])
                resolved = str(t["Object Name"][0])
                if not (np.isfinite(ra) and np.isfinite(dec)):
                    return None
                return GalaxyCoord(name=name, ra_deg=ra, dec_deg=dec, resolver="NED", resolved_name=resolved)
            except Exception:
                return None
        return None


def resolve_coords_simbad(name: str) -> Optional[GalaxyCoord]:
    from astroquery.simbad import Simbad

    try:
        sim = Simbad()
        sim.add_votable_fields("ra(d)", "dec(d)")
        t = sim.query_object(name)
        if t is None or len(t) == 0:
            return None
        ra = float(t["ra"][0])
        dec = float(t["dec"][0])
        resolved = str(t["MAIN_ID"][0])
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return None
        return GalaxyCoord(name=name, ra_deg=ra, dec_deg=dec, resolver="SIMBAD", resolved_name=resolved)
    except Exception:
        return None


def resolve_coords(name: str, sleep_s: float = 0.1) -> Optional[GalaxyCoord]:
    # NED handles many SPARC-specific aliases; SIMBAD is a good backup.
    coord = resolve_coords_ned(name)
    if coord is None:
        coord = resolve_coords_simbad(name)

    # Be gentle with remote services.
    if sleep_s > 0:
        time.sleep(sleep_s)

    return coord


def write_coords_csv(coords: Sequence[GalaxyCoord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "ra_deg", "dec_deg", "resolver", "resolved_name"],
        )
        w.writeheader()
        for c in coords:
            w.writerow(
                {
                    "name": c.name,
                    "ra_deg": f"{c.ra_deg:.8f}",
                    "dec_deg": f"{c.dec_deg:.8f}",
                    "resolver": c.resolver,
                    "resolved_name": c.resolved_name,
                }
            )


def read_coords_csv(path: Path) -> List[GalaxyCoord]:
    out: List[GalaxyCoord] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(
                GalaxyCoord(
                    name=row["name"],
                    ra_deg=float(row["ra_deg"]),
                    dec_deg=float(row["dec_deg"]),
                    resolver=row.get("resolver", ""),
                    resolved_name=row.get("resolved_name", ""),
                )
            )
    return out


def angular_separation_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: np.ndarray, dec2_deg: np.ndarray) -> np.ndarray:
    """Vectorized great-circle separation."""
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2r = np.radians(np.asarray(ra2_deg, dtype=float))
    dec2r = np.radians(np.asarray(dec2_deg, dtype=float))

    # Haversine
    dra = ra2r - ra1
    ddec = dec2r - dec1
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2r) * np.sin(dra / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.degrees(c) * 3600.0


def build_allwise_dataset(wise_mode: str, local_root: Optional[str]) -> Tuple[Any, str]:
    import pyarrow.dataset as ds

    mode = str(wise_mode).lower().strip()
    if mode == "s3":
        from pyarrow.fs import S3FileSystem

        bucket = "nasa-irsa-wise"
        folder = "wise/allwise/catalogs/p3as_psd/healpix_k5"
        parquet_root = f"{bucket}/{folder}/wise-allwise.parquet"
        fs = S3FileSystem(region="us-west-2")
        dataset = ds.dataset(parquet_root, filesystem=fs, format="parquet", partitioning="hive")
        return dataset, parquet_root

    if mode == "local":
        if not local_root:
            raise ValueError("--wise-local-root is required for --wise-mode=local")
        root = str(local_root)
        dataset = ds.dataset(root, format="parquet", partitioning="hive")
        return dataset, root

    raise ValueError("--wise-mode must be 's3' or 'local'")


def query_allwise_candidates(
    dataset: Any,
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    columns: Sequence[str],
) -> "np.ndarray":
    import hpgeom as hp
    import pyarrow.dataset as ds

    radius_deg = float(radius_arcsec) / 3600.0
    nside = hp.order_to_nside(5)

    pixels = hp.query_circle(
        nside,
        ra_deg,
        dec_deg,
        radius_deg,
        nest=True,
        inclusive=True,
        lonlat=True,
    )
    pixels = [int(p) for p in np.asarray(pixels).tolist()]

    # Conservative bounding box for fast pre-filter.
    dec_min = dec_deg - radius_deg
    dec_max = dec_deg + radius_deg
    cos_dec = max(math.cos(math.radians(dec_deg)), 1e-3)
    dra = radius_deg / cos_dec
    ra_min = ra_deg - dra
    ra_max = ra_deg + dra

    # Handle RA wrap-around.
    ra_field = ds.field("ra")
    dec_field = ds.field("dec")
    hp_field = ds.field("healpix_k5")

    filt = (hp_field.isin(pixels)) & (dec_field >= dec_min) & (dec_field <= dec_max)
    if ra_min < 0:
        filt = filt & ((ra_field >= (ra_min + 360.0)) | (ra_field <= ra_max))
    elif ra_max > 360:
        filt = filt & ((ra_field >= ra_min) | (ra_field <= (ra_max - 360.0)))
    else:
        filt = filt & (ra_field >= ra_min) & (ra_field <= ra_max)

    table = dataset.to_table(columns=list(columns), filter=filt)
    return table


def pick_best_allwise_match(df: "Any") -> Optional[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return None

    # Ensure numeric.
    df = df.copy()
    if "ext_flg" in df:
        df["ext_flg"] = df["ext_flg"].fillna(0).astype(int)
    else:
        df["ext_flg"] = 0

    # Prefer extended sources.
    df["is_extended"] = df["ext_flg"] > 0

    # Brightness proxy for tie-breaking.
    if "w1gmag" in df:
        bright = df["w1gmag"].copy()
    else:
        bright = np.nan
    if np.all(~np.isfinite(bright)) and "w1mpro" in df:
        bright = df["w1mpro"].copy()
    df["bright_proxy"] = bright

    # Sort: extended first, then separation, then brightest.
    df = df.sort_values(by=["is_extended", "sep_arcsec", "bright_proxy"], ascending=[False, True, True])
    return df.iloc[0].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparc-dir", default="data/Rotmod_LTG", help="Path to SPARC Rotmod_LTG directory")
    ap.add_argument("--out-dir", default="resources/photometry", help="Output directory for derived tables")
    ap.add_argument("--wise-mode", default="s3", choices=["s3", "local"], help="Query AllWISE parquet via S3 or local")
    ap.add_argument(
        "--wise-local-root",
        default="/Users/leonardspeiser/Projects/sigmagravity_external_data/wise/allwise/healpix_k5/wise-allwise-parquet/wise-allwise.parquet",
        help="Local path to wise-allwise.parquet root (only for --wise-mode=local)",
    )
    ap.add_argument("--radius-arcsec", type=float, default=60.0, help="Search radius around each galaxy center")
    ap.add_argument("--sleep-s", type=float, default=0.1, help="Sleep between resolver calls")
    ap.add_argument("--max-galaxies", type=int, default=0, help="Limit number of galaxies (0 = all)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sparc_dir = (repo_root / args.sparc_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_path = out_dir / "sparc_coordinates.csv"
    matches_path = out_dir / "sparc_allwise_matches.csv"

    names = list_sparc_galaxy_names(sparc_dir)
    if args.max_galaxies and args.max_galaxies > 0:
        names = names[: args.max_galaxies]

    # Resolve coords
    coords: List[GalaxyCoord] = []
    unresolved: List[str] = []
    for name in names:
        c = resolve_coords(name, sleep_s=args.sleep_s)
        if c is None:
            unresolved.append(name)
            continue
        coords.append(c)

    write_coords_csv(coords, coords_path)

    # Build WISE matches
    dataset, root = build_allwise_dataset(args.wise_mode, args.wise_local_root)

    columns = [
        "designation",
        "ra",
        "dec",
        "ext_flg",
        "xscprox",
        "w1gmag",
        "w2gmag",
        "w1mpro",
        "w2mpro",
        "w1sigmpro",
        "w2sigmpro",
        "w1snr",
        "w2snr",
        "cc_flags",
        "ph_qual",
        "healpix_k5",
    ]

    rows: List[Dict[str, Any]] = []
    import pandas as pd

    for c in coords:
        table = query_allwise_candidates(dataset, c.ra_deg, c.dec_deg, args.radius_arcsec, columns)
        df = table.to_pandas()
        n_candidates_bbox = int(len(df))
        if len(df) == 0:
            rows.append(
                {
                    "name": c.name,
                    "ra_deg": c.ra_deg,
                    "dec_deg": c.dec_deg,
                    "resolver": c.resolver,
                    "resolved_name": c.resolved_name,
                    "match_found": False,
                    "n_candidates_bbox": 0,
                    "n_candidates_within_radius": 0,
                    "n_candidates": 0,
                }
            )
            continue

        df["sep_arcsec"] = angular_separation_arcsec(c.ra_deg, c.dec_deg, df["ra"].values, df["dec"].values)
        df = df[df["sep_arcsec"] <= float(args.radius_arcsec)].copy()
        n_candidates_within = int(len(df))
        if n_candidates_within == 0:
            rows.append(
                {
                    "name": c.name,
                    "ra_deg": c.ra_deg,
                    "dec_deg": c.dec_deg,
                    "resolver": c.resolver,
                    "resolved_name": c.resolved_name,
                    "match_found": False,
                    "n_candidates_bbox": n_candidates_bbox,
                    "n_candidates_within_radius": 0,
                    "n_candidates": 0,
                }
            )
            continue

        best = pick_best_allwise_match(df)
        if best is None:
            rows.append(
                {
                    "name": c.name,
                    "ra_deg": c.ra_deg,
                    "dec_deg": c.dec_deg,
                    "resolver": c.resolver,
                    "resolved_name": c.resolved_name,
                    "match_found": False,
                    "n_candidates_bbox": n_candidates_bbox,
                    "n_candidates_within_radius": n_candidates_within,
                    "n_candidates": n_candidates_within,
                }
            )
            continue

        # Compute colors
        w1g = best.get("w1gmag")
        w2g = best.get("w2gmag")
        w1m = best.get("w1mpro")
        w2m = best.get("w2mpro")

        color_g = (w1g - w2g) if (w1g is not None and w2g is not None and np.isfinite(w1g) and np.isfinite(w2g)) else np.nan
        color_m = (w1m - w2m) if (w1m is not None and w2m is not None and np.isfinite(w1m) and np.isfinite(w2m)) else np.nan

        row = {
            "name": c.name,
            "ra_deg": c.ra_deg,
            "dec_deg": c.dec_deg,
            "resolver": c.resolver,
            "resolved_name": c.resolved_name,
            "match_found": True,
            "n_candidates_bbox": n_candidates_bbox,
            "n_candidates_within_radius": n_candidates_within,
            "n_candidates": n_candidates_within,
            "designation": best.get("designation"),
            "wise_ra": best.get("ra"),
            "wise_dec": best.get("dec"),
            "sep_arcsec": float(best.get("sep_arcsec", np.nan)),
            "healpix_k5": best.get("healpix_k5"),
            "ext_flg": best.get("ext_flg"),
            "xscprox_arcsec": best.get("xscprox"),
            "cc_flags": best.get("cc_flags"),
            "ph_qual": best.get("ph_qual"),
            "w1gmag": w1g,
            "w2gmag": w2g,
            "w1mpro": w1m,
            "w2mpro": w2m,
            "w1sigmpro": best.get("w1sigmpro"),
            "w2sigmpro": best.get("w2sigmpro"),
            "w1snr": best.get("w1snr"),
            "w2snr": best.get("w2snr"),
            "w1_w2_gmag": color_g,
            "w1_w2_mpro": color_m,
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(matches_path, index=False)

    unresolved_path = out_dir / "sparc_unresolved_names.txt"
    if unresolved:
        unresolved_path.write_text("\n".join(unresolved) + "\n")
    else:
        if unresolved_path.exists():
            unresolved_path.unlink()

    print(f"Wrote: {coords_path}")
    print(f"Wrote: {matches_path}")
    print(f"AllWISE source: {args.wise_mode} ({root})")
    if unresolved:
        print(f"Unresolved names: {len(unresolved)} (see sparc_unresolved_names.txt)")


if __name__ == "__main__":
    main()
