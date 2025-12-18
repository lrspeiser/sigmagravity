#!/usr/bin/env python3
"""add_gaia_proper_motions.py

Add Gaia DR3 proper motions to bulge kinematics catalogs.

This script cross-matches existing bulge/inner-Galaxy catalogs (BRAVA, APOGEE,
GIBS) against Gaia DR3 in order to attach:

  - pmra/pmdec (and errors)
  - parallax (and error)
  - Gaia radial velocity when available
  - quality flags (RUWE, visibility_periods_used)

Why this file exists
--------------------
For bulge datasets (8k–50k stars), querying Gaia *one star at a time* becomes a
multi-day operation.

The default mode here uses a **bulk** ADQL join against an uploaded
`TAP_UPLOAD` table of query coordinates. In practice, this reduces runtime from
minutes/star → minutes per 10–20k stars (depending on TAP service load).
Default chunk size is 10000 stars per query to minimize the number of TAP queries.
Each TAP query takes ~2-3 minutes regardless of chunk size, so fewer chunks = much faster.

Usage
-----
    # Fast path (recommended, default chunk-size=10000)
    python scripts/add_gaia_proper_motions.py --catalog BRAVA --method bulk

    # Fast path with GPU reduction (needs CuPy + NVIDIA card)
    python scripts/add_gaia_proper_motions.py --catalog BRAVA --method bulk --gpu

    # Maximum chunk size for very large catalogs (fewest queries = fastest)
    python scripts/add_gaia_proper_motions.py --catalog BRAVA --method bulk --chunk-size 20000 --gpu

    # Debug / fallback
    python scripts/add_gaia_proper_motions.py --catalog BRAVA --method per_star --max-stars 10

    # Process all catalogs that exist locally
    python scripts/add_gaia_proper_motions.py --all --method bulk
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional, Literal
from functools import partial
from multiprocessing import Pool, cpu_count
import time
import random

import numpy as np
from astropy.table import Table, vstack, join
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parent.parent
BULGE_DIR = PROJECT_ROOT / "data" / "bulge_kinematics"
OUTPUT_DIR = BULGE_DIR / "crossmatched"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Gaia TAP table (DR3)
GAIA_TABLE = "gaiadr3.gaia_source"

# Columns we want from Gaia
GAIA_SELECT_COLS = (
    "g.source_id, g.ra, g.dec, "
    "g.parallax, g.parallax_error, "
    "g.pmra, g.pmra_error, "
    "g.pmdec, g.pmdec_error, "
    "g.radial_velocity, g.radial_velocity_error, "
    "g.phot_g_mean_mag, "
    "g.ruwe, "
    "g.visibility_periods_used"
)


def _get_gaia():
    """Import astroquery.gaia.Gaia (installing astroquery if missing)."""
    try:
        from astroquery.gaia import Gaia  # type: ignore
        return Gaia
    except ImportError:
        print("Installing astroquery...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "astroquery"])
        from astroquery.gaia import Gaia  # type: ignore
        return Gaia


def _get_cupy():
    """Return CuPy module if available, else None."""
    try:
        import cupy as cp  # type: ignore
        return cp
    except Exception:
        return None


_CUPY_UNAVAILABLE_WARNED = False


def ensure_radec_deg(table: Table) -> Table:
    """Ensure float 'ra'/'dec' columns exist (degrees).

    Accepts common naming variants and can convert Galactic (l,b) to ICRS.
    Modifies the table in-place and returns it.
    """
    if table is None:
        return table

    cols = set(table.colnames)

    # 1) Direct RA/Dec variants
    radec_candidates = [
        ("ra", "dec"),
        ("RA", "DEC"),
        ("RAJ2000", "DEJ2000"),
        ("raj2000", "dej2000"),
        ("RA_ICRS", "DE_ICRS"),
        ("ra_icrs", "dec_icrs"),
        ("_RAJ2000", "_DEJ2000"),
    ]
    for ra_col, dec_col in radec_candidates:
        if ra_col in cols and dec_col in cols:
            table["ra"] = np.array(table[ra_col], dtype=float)
            table["dec"] = np.array(table[dec_col], dtype=float)
            return table

    # 2) Galactic lon/lat variants
    lb_candidates = [
        ("l", "b"),
        ("L", "B"),
        ("glon", "glat"),
        ("GLON", "GLAT"),
        ("l_deg", "b_deg"),
        ("GLON_DEG", "GLAT_DEG"),
    ]
    for l_col, b_col in lb_candidates:
        if l_col in cols and b_col in cols:
            l = np.array(table[l_col], dtype=float)
            b = np.array(table[b_col], dtype=float)
            c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
            table["ra"] = c.ra.deg
            table["dec"] = c.dec.deg
            return table

    raise ValueError(
        "Catalog is missing coordinates. Need RA/Dec columns (ra/dec, RA/DEC, RAJ2000/DEJ2000, etc) "
        "or Galactic lon/lat (l/b, glon/glat)."
    )


def load_catalog(catalog_name: str) -> tuple[Table, str]:
    """Load one bulge kinematics catalog from the repo data directory."""
    name = catalog_name.upper()

    if name == "BRAVA":
        catalog_path = BULGE_DIR / "BRAVA" / "brava_catalog.tbl"
        if not catalog_path.exists():
            raise FileNotFoundError(f"BRAVA catalog not found: {catalog_path}")

        from astropy.io import ascii

        table = ascii.read(catalog_path, format="ipac")
        table = ensure_radec_deg(table)
        return table, "BRAVA"

    if name == "APOGEE":
        for dr in (18, 17):
            catalog_path = BULGE_DIR / "APOGEE" / f"apogee_bulge_dr{dr}.fits"
            if catalog_path.exists():
                table = Table.read(catalog_path)
                table = ensure_radec_deg(table)
                return table, f"APOGEE_DR{dr}"
        raise FileNotFoundError("APOGEE catalog not found. Run download_apogee_bulge.py (or CasJobs manual download) first.")

    if name == "GIBS":
        catalog_path = BULGE_DIR / "GIBS" / "gibs_catalog_1.fits"
        if not catalog_path.exists():
            raise FileNotFoundError(f"GIBS catalog not found: {catalog_path}")
        table = Table.read(catalog_path)
        table = ensure_radec_deg(table)
        return table, "GIBS"

    raise ValueError(f"Unknown catalog: {catalog_name}")


def load_table_any(path: Path) -> Table:
    """Load a table from a user-provided path (FITS/ECSV/CSV/VOTable/IPAC).

    Astropy is usually able to infer the format from the extension. For CSV,
    we fall back to explicit format='csv' if needed.
    """
    try:
        return Table.read(str(path))
    except Exception:
        # Common fallback for CSV
        return Table.read(str(path), format="csv")


def _gaia_join_query(radius_deg: float) -> str:
    """ADQL to join uploaded coords (TAP_UPLOAD.t) to Gaia DR3."""
    # We intentionally return *all* matches within radius.
    # We de-duplicate per input row_id in Python by taking the minimum 'sep'.
    return f"""
    SELECT
        t.row_id,
        {GAIA_SELECT_COLS},
        DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', t.ra, t.dec)) AS sep
    FROM {GAIA_TABLE} AS g
    JOIN TAP_UPLOAD.t AS t
      ON 1 = CONTAINS(
            POINT('ICRS', g.ra, g.dec),
            CIRCLE('ICRS', t.ra, t.dec, {radius_deg})
      )
    """


def _bulk_query_gaia(
    Gaia,
    upload: Table,
    radius_deg: float,
) -> Optional[Table]:
    """Run one bulk Gaia query for an uploaded chunk."""
    query = _gaia_join_query(radius_deg)
    job = Gaia.launch_job_async(
        query,
        upload_resource=upload,
        upload_table_name="t",
        dump_to_file=False,
    )
    res = job.get_results()
    if res is None or len(res) == 0:
        return None
    return res


def _select_best_matches(gaia_all: Table, use_gpu: bool = False) -> Table:
    """Select nearest Gaia match per row_id, optionally using CuPy."""
    row_id_np = np.asarray(gaia_all["row_id"], dtype=np.int64)
    sep_np = np.asarray(gaia_all["sep"], dtype=float)

    global _CUPY_UNAVAILABLE_WARNED
    cp = _get_cupy() if use_gpu else None

    if cp is not None:
        # GPU-accelerated lexsort+unique
        sep_cp = cp.asarray(sep_np)
        row_cp = cp.asarray(row_id_np)
        # cupy 13.x expects a stacked array, not a tuple, for lexsort
        order = cp.lexsort(cp.stack((sep_cp, row_cp)))
        row_sorted = cp.take(row_cp, order)
        _, first_idx = cp.unique(row_sorted, return_index=True)
        best_idx = cp.take(order, first_idx).get()
    else:
        if use_gpu and cp is None and not _CUPY_UNAVAILABLE_WARNED:
            print("[warn] CuPy requested but not available; falling back to NumPy.")
            _CUPY_UNAVAILABLE_WARNED = True
        order = np.lexsort((sep_np, row_id_np))
        row_sorted = row_id_np[order]
        _, first_idx = np.unique(row_sorted, return_index=True)
        best_idx = order[first_idx]

    return gaia_all[best_idx]


def _process_chunk_worker(args_tuple):
    """Worker function for parallel chunk processing."""
    start, end, table_data, radius_deg, chunk_idx, use_gpu = args_tuple
    
    # Re-import Gaia in worker process (required for multiprocessing)
    Gaia = _get_gaia()
    
    # Reconstruct table slice
    row_ids = table_data['row_id'][start:end]
    ra_vals = table_data['ra'][start:end]
    dec_vals = table_data['dec'][start:end]
    
    upload = Table({
        "row_id": np.array(row_ids, dtype=np.int64),
        "ra": np.array(ra_vals, dtype=float),
        "dec": np.array(dec_vals, dtype=float),
    })
    
    # Retry with backoff to handle TAP timeouts
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            res = _bulk_query_gaia(Gaia, upload, radius_deg)
            if res is None or len(res) == 0:
                return (chunk_idx, None, None)
            best = _select_best_matches(res, use_gpu=use_gpu)
            return (chunk_idx, best, None)
        except Exception as e:
            if attempt == max_attempts:
                return (chunk_idx, None, f"{e}")
            sleep_s = 3 * attempt + random.random()
            time.sleep(sleep_s)


def crossmatch_gaia_bulk(
    catalog_table: Table,
    max_sep_arcsec: float = 2.0,
    chunk_size: int = 10000,
    max_stars: Optional[int] = None,
    keep_unmatched: bool = True,
    n_workers: Optional[int] = None,
    use_gpu: bool = False,
) -> Optional[Table]:
    """Bulk Gaia crossmatch using TAP_UPLOAD join.

    Parameters
    ----------
    catalog_table
        Input table that must contain ra/dec (or l/b) columns.
    max_sep_arcsec
        Match radius.
    chunk_size
        Number of targets per TAP upload/query. Larger chunks = fewer queries = faster.
        Default 10000 minimizes queries. Can go up to 20000 if TAP is stable.
        TAP queries are the bottleneck (each takes ~2-3 min), so fewer chunks = much faster.
    max_stars
        Optional limit for testing.
    keep_unmatched
        If True, return the full input table with Gaia columns masked for
        unmatched rows. If False, return only matched rows.
    use_gpu
        If True and CuPy is installed, use GPU acceleration for best-match
        selection (row_id/sep reduction). Note: GPU utilization may be low
        because TAP query latency is the main bottleneck, not GPU reduction.
    """
    Gaia = _get_gaia()

    table = catalog_table.copy()
    table = ensure_radec_deg(table)

    if max_stars is not None and len(table) > max_stars:
        table = table[: max_stars]

    n = len(table)
    if n == 0:
        return None

    # Add row_id for stable round-trip mapping
    table["row_id"] = np.arange(n, dtype=np.int64)

    radius_deg = float(max_sep_arcsec) / 3600.0
    best_hits: list[Table] = []

    ranges = list(range(0, n, int(chunk_size)))
    
    # Determine number of workers - maximize parallelism to saturate TAP
    if n_workers is None:
        # TAP queries are I/O bound, so we can use more workers than CPU cores
        # Use all CPU cores + some extra for I/O wait time
        # TAP can typically handle 16-20 concurrent queries
        cpu_cores = cpu_count()
        n_workers = min(cpu_cores * 2, len(ranges), 20)  # 2x CPU cores, up to 20 workers
    
    print(f"\nBulk Gaia DR3 cross-match: {n:,} targets in {len(ranges)} chunk(s) (chunk_size={chunk_size})")
    if n_workers > 1:
        print(f"Using {n_workers} parallel workers")
    if use_gpu:
        print("CuPy mode enabled: best-match reduction will run on GPU")
    
    # Prepare data for workers (convert table to dict for pickling)
    table_data = {
        'row_id': np.array(table["row_id"], dtype=np.int64),
        'ra': np.array(table["ra"], dtype=float),
        'dec': np.array(table["dec"], dtype=float),
    }
    
    # Prepare arguments for workers
    worker_args = [
        (start, min(start + int(chunk_size), n), table_data, radius_deg, i, use_gpu)
        for i, start in enumerate(ranges)
    ]
    
    # Process chunks (parallel or sequential)
    if n_workers > 1 and len(ranges) > 1:
        # Parallel processing
        with Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_process_chunk_worker, worker_args),
                    total=len(worker_args),
                    desc="Gaia bulk chunks (parallel)"
                )
            )
        
        # Collect results
        for chunk_idx, res, error in results:
            if error:
                print(f"\n[warn] Gaia bulk query failed for chunk {chunk_idx}: {error}")
            if res is not None and len(res) > 0:
                best_hits.append(res)
    else:
        # Sequential processing (fallback or single chunk)
        for start in tqdm(ranges, desc="Gaia bulk chunks"):
            end = min(start + int(chunk_size), n)
            upload = Table(
                {
                    "row_id": np.array(table["row_id"][start:end], dtype=np.int64),
                    "ra": np.array(table["ra"][start:end], dtype=float),
                    "dec": np.array(table["dec"][start:end], dtype=float),
                }
            )

            # Retry with backoff
            max_attempts = 3
            res = None
            for attempt in range(1, max_attempts + 1):
                try:
                    res = _bulk_query_gaia(Gaia, upload, radius_deg)
                    break
                except Exception as e:
                    if attempt == max_attempts:
                        print(f"\n[warn] Gaia bulk query failed for chunk {start}:{end}: {e}")
                    else:
                        sleep_s = 3 * attempt + random.random()
                        time.sleep(sleep_s)

            if res is not None and len(res) > 0:
                best = _select_best_matches(res, use_gpu=use_gpu)
                best_hits.append(best)

    if not best_hits:
        print("[warn] No Gaia matches found")
        return None

    # Each chunk has disjoint row_id ranges, so per-chunk reduction is sufficient.
    gaia_best = vstack(best_hits, metadata_conflicts="silent")

    # Prefix Gaia columns to avoid collisions
    rename_map = {}
    for col in gaia_best.colnames:
        if col == "row_id":
            continue
        rename_map[col] = f"gaia_{col}"
    gaia_best.rename_columns(list(rename_map.keys()), list(rename_map.values()))

    # Join back to the catalog
    out = join(table, gaia_best, keys="row_id", join_type="left")
    # Convert Gaia sep (deg) → arcsec if present
    if "gaia_sep" in out.colnames:
        out["gaia_sep_arcsec"] = np.array(out["gaia_sep"], dtype=float) * 3600.0

    # Matched flag
    out["gaia_matched"] = ~out["gaia_source_id"].mask if hasattr(out["gaia_source_id"], "mask") else np.isfinite(out["gaia_source_id"])

    if not keep_unmatched:
        out = out[out["gaia_matched"]]

    # Clean up
    if "row_id" in out.colnames:
        out.remove_column("row_id")

    n_match = int(np.sum(out["gaia_matched"])) if "gaia_matched" in out.colnames else 0
    print(f"  Gaia matches: {n_match:,}/{len(out):,} ({(n_match/max(len(out),1))*100:.1f}%)")
    return out


def crossmatch_gaia_per_star(
    catalog_table: Table,
    max_sep_arcsec: float = 2.0,
    max_stars: Optional[int] = None,
    keep_unmatched: bool = True,
) -> Optional[Table]:
    """Slow fallback: one cone-search per star.

    This is mainly kept for debugging or when TAP_UPLOAD is unavailable.
    """
    Gaia = _get_gaia()

    table = catalog_table.copy()
    table = ensure_radec_deg(table)

    if max_stars is not None and len(table) > max_stars:
        table = table[: max_stars]

    n = len(table)
    if n == 0:
        return None

    # Prepare output columns
    out = table.copy()
    out["gaia_matched"] = np.zeros(n, dtype=bool)
    out["gaia_sep_arcsec"] = np.full(n, np.nan)

    # Precreate Gaia columns (masked)
    gaia_cols = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "radial_velocity",
        "radial_velocity_error",
        "phot_g_mean_mag",
        "ruwe",
        "visibility_periods_used",
    ]
    for c in gaia_cols:
        out[f"gaia_{c}"] = np.full(n, np.nan)

    coords = SkyCoord(
        ra=np.array(out["ra"], dtype=float) * u.deg,
        dec=np.array(out["dec"], dtype=float) * u.deg,
        frame="icrs",
    )

    radius = u.Quantity(max_sep_arcsec, u.arcsec)
    print(f"\nPer-star Gaia DR3 cross-match: {n:,} target(s) (this is slow)")
    for i, c in enumerate(tqdm(coords, desc="Gaia per-star")):
        try:
            job = Gaia.cone_search_async(c, radius=radius)
            res = job.get_results()
        except Exception:
            continue
        if res is None or len(res) == 0:
            continue

        # Choose closest
        # cone_search_async already returns within radius; use DISTANCE in python
        sky = SkyCoord(ra=res["ra"] * u.deg, dec=res["dec"] * u.deg, frame="icrs")
        sep = c.separation(sky).arcsec
        j = int(np.argmin(sep))
        if sep[j] > max_sep_arcsec:
            continue

        out["gaia_matched"][i] = True
        out["gaia_sep_arcsec"][i] = float(sep[j])
        for col in gaia_cols:
            if col in res.colnames:
                out[f"gaia_{col}"][i] = res[col][j]

    if not keep_unmatched:
        out = out[out["gaia_matched"]]

    n_match = int(np.sum(out["gaia_matched"])) if "gaia_matched" in out.colnames else 0
    print(f"  Gaia matches: {n_match:,}/{len(out):,} ({(n_match/max(len(out),1))*100:.1f}%)")
    return out


def crossmatch_gaia(
    catalog_table: Table,
    method: Literal["bulk", "per_star"] = "bulk",
    max_sep_arcsec: float = 2.0,
    chunk_size: int = 2000,
    max_stars: Optional[int] = None,
    keep_unmatched: bool = True,
    n_workers: Optional[int] = None,
    use_gpu: bool = False,
) -> Optional[Table]:
    """Dispatch to bulk/per-star crossmatch."""
    if method == "per_star":
        return crossmatch_gaia_per_star(
            catalog_table,
            max_sep_arcsec=max_sep_arcsec,
            max_stars=max_stars,
            keep_unmatched=keep_unmatched,
        )
    return crossmatch_gaia_bulk(
        catalog_table,
        max_sep_arcsec=max_sep_arcsec,
        chunk_size=chunk_size,
        max_stars=max_stars,
        keep_unmatched=keep_unmatched,
        n_workers=n_workers,
        use_gpu=use_gpu,
    )


def process_catalog(
    catalog_name: str,
    method: Literal["bulk", "per_star"],
    max_sep_arcsec: float,
    chunk_size: int,
    max_stars: Optional[int],
    keep_unmatched: bool,
    n_workers: Optional[int] = None,
    use_gpu: bool = False,
) -> bool:
    """Process one catalog end-to-end."""
    print("=" * 70)
    print(f"PROCESSING {catalog_name.upper()}")
    print("=" * 70)

    try:
        catalog_table, catalog_label = load_catalog(catalog_name)
        print(f"Loaded {len(catalog_table):,} rows from {catalog_label}")
    except Exception as e:
        print(f"[error] Error loading catalog: {e}")
        return False

    matched = crossmatch_gaia(
        catalog_table,
        method=method,
        max_sep_arcsec=max_sep_arcsec,
        chunk_size=chunk_size,
        max_stars=max_stars,
        keep_unmatched=keep_unmatched,
        n_workers=n_workers,
        use_gpu=use_gpu,
    )

    if matched is None:
        return False

    output_path = OUTPUT_DIR / f"{catalog_label.lower()}_with_gaia.fits"
    matched.write(str(output_path), overwrite=True)

    n_match = int(np.sum(matched["gaia_matched"])) if "gaia_matched" in matched.colnames else 0
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(matched):,} (matched: {n_match:,})")
    return True


def process_input_file(
    input_path: Path,
    output_path: Optional[Path],
    method: Literal["bulk", "per_star"],
    max_sep_arcsec: float,
    chunk_size: int,
    max_stars: Optional[int],
    keep_unmatched: bool,
    use_gpu: bool = False,
) -> bool:
    """Process an arbitrary input catalog file (not just BRAVA/APOGEE/GIBS)."""
    print("=" * 70)
    print(f"PROCESSING FILE: {input_path}")
    print("=" * 70)

    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        return False

    try:
        table = load_table_any(input_path)
        table = ensure_radec_deg(table)
    except Exception as e:
        print(f"[error] Failed to read/standardize input table: {e}")
        return False

    print(f"Loaded {len(table):,} rows")

    matched = crossmatch_gaia(
        table,
        method=method,
        max_sep_arcsec=max_sep_arcsec,
        chunk_size=chunk_size,
        max_stars=max_stars,
        keep_unmatched=keep_unmatched,
        use_gpu=use_gpu,
    )
    if matched is None:
        return False

    if output_path is None:
        output_path = OUTPUT_DIR / f"{input_path.stem}_with_gaia.fits"
    else:
        output_path = output_path

    matched.write(str(output_path), overwrite=True)
    n_match = int(np.sum(matched["gaia_matched"])) if "gaia_matched" in matched.colnames else 0
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(matched):,} (matched: {n_match:,})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Gaia DR3 proper motions to bulge kinematics catalogs",
    )
    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Custom input table (FITS/ECSV/CSV/VOTable/IPAC) to crossmatch. "
            "If provided, this overrides --catalog/--all."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        help=(
            "Output FITS path for --input. Default: data/bulge_kinematics/crossmatched/<stem>_with_gaia.fits"
        ),
    )
    parser.add_argument(
        "--catalog",
        choices=["BRAVA", "APOGEE", "GIBS"],
        help="Catalog to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available catalogs",
    )
    parser.add_argument(
        "--method",
        choices=["bulk", "per_star"],
        default="bulk",
        help="Gaia crossmatch method (default: bulk)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Targets per TAP upload/query when --method=bulk (default: 10000, larger = fewer queries = faster. Try 20000 for very large catalogs)",
    )
    parser.add_argument(
        "--max-sep",
        type=float,
        default=2.0,
        help="Maximum separation for cross-match (arcsec, default: 2.0)",
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        help="Maximum number of stars to process (for testing)",
    )
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help="Only keep matched rows (default keeps all rows with masked Gaia cols)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for bulk mode (default: 2x CPU cores, up to 20. TAP queries are I/O bound so more workers = faster)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CuPy on GPU (e.g., RTX 5090) to speed up post-query reductions",
    )

    args = parser.parse_args()

    keep_unmatched = not bool(args.matched_only)

    # Custom input mode
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")

        table = load_table_any(input_path)
        print(f"Loaded {len(table):,} rows from {input_path}")

        matched = crossmatch_gaia(
            table,
            method=args.method,
            max_sep_arcsec=float(args.max_sep),
            chunk_size=int(args.chunk_size),
            max_stars=args.max_stars,
            keep_unmatched=keep_unmatched,
            n_workers=args.n_workers,
            use_gpu=args.gpu,
        )
        if matched is None:
            raise SystemExit("No matches (or Gaia query failed)")

        if args.output:
            out_path = Path(args.output)
        else:
            out_path = OUTPUT_DIR / f"{input_path.stem}_with_gaia.fits"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        matched.write(str(out_path), overwrite=True)

        n_match = int(np.sum(matched["gaia_matched"])) if "gaia_matched" in matched.colnames else 0
        print(f"\nSaved: {out_path}")
        print(f"  Rows: {len(matched):,} (matched: {n_match:,})")
        return

    # Named catalog mode
    if args.all:
        catalogs = ["BRAVA", "APOGEE", "GIBS"]
    elif args.catalog:
        catalogs = [args.catalog]
    else:
        parser.print_help()
        sys.exit(1)

    results: dict[str, bool] = {}
    for cat in catalogs:
        results[cat] = process_catalog(
            cat,
            method=args.method,
            max_sep_arcsec=float(args.max_sep),
            chunk_size=int(args.chunk_size),
            max_stars=args.max_stars,
            keep_unmatched=keep_unmatched,
            n_workers=args.n_workers,
            use_gpu=args.gpu,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cat, ok in results.items():
        print(f"{'OK' if ok else 'FAIL'} {cat}")
    print(f"\nCross-matched catalogs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
