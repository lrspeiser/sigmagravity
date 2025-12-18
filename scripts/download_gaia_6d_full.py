#!/usr/bin/env python3
"""
Download full Gaia DR3 6D catalog (positions + proper motions + radial velocities).

This script downloads ~470 million stars with complete 6D phase space information
from Gaia DR3. Due to the massive size, it downloads in chunks by sky region.

Strategy:
- Split sky into HEALPix pixels (or l/b bins) for manageable chunks
- Query each region separately
- Save as FITS files (more efficient than CSV for large datasets)
- Can combine files later if needed

Usage:
    # Download all 6D stars (will take many hours/days)
    python scripts/download_gaia_6d_full.py --output-dir data/gaia/6d_full

    # Download specific region only
    python scripts/download_gaia_6d_full.py --l-min 0 --l-max 360 --b-min -10 --b-max 10 --output-dir data/gaia/6d_bulge

    # Test with small sample first
    python scripts/download_gaia_6d_full.py --max-stars 10000 --output-dir data/gaia/6d_test
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import time
from typing import Optional
import numpy as np
from astropy.table import Table, vstack
from astroquery.gaia import Gaia
from tqdm import tqdm


# Base ADQL query for 6D stars (with radial velocity)
BASE_QUERY_6D = """
SELECT {limit_clause}
    g.source_id,
    g.ra, g.dec,
    g.l, g.b,
    g.parallax, g.parallax_error,
    g.pmra, g.pmra_error,
    g.pmdec, g.pmdec_error,
    g.radial_velocity, g.radial_velocity_error,
    g.phot_g_mean_mag,
    g.bp_rp,
    g.ruwe,
    g.visibility_periods_used
FROM gaiadr3.gaia_source AS g
WHERE
    g.radial_velocity IS NOT NULL
    {sky_cuts}
    {quality_cuts}
    {order_by}
"""

# Query for 6D stars matching 2MASS IDs (via cross-match table)
BASE_QUERY_6D_2MASS = """
SELECT
    g.source_id,
    g.ra, g.dec,
    g.l, g.b,
    g.parallax, g.parallax_error,
    g.pmra, g.pmra_error,
    g.pmdec, g.pmdec_error,
    g.radial_velocity, g.radial_velocity_error,
    g.phot_g_mean_mag,
    g.bp_rp,
    g.ruwe,
    g.visibility_periods_used,
    xmatch.original_ext_source_id AS tmass_id
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch
    ON g.source_id = xmatch.source_id
JOIN TAP_UPLOAD.tmass_ids AS tids
    ON xmatch.original_ext_source_id = tids.tmass_id
WHERE
    g.radial_velocity IS NOT NULL
    {quality_cuts}
"""


def build_query(
    l_min: Optional[float] = None,
    l_max: Optional[float] = None,
    b_min: Optional[float] = None,
    b_max: Optional[float] = None,
    ruwe_max: float = 1.6,
    vis_min: int = 8,
    max_stars: Optional[int] = None,
    use_random: bool = True,
) -> str:
    """Build ADQL query for 6D Gaia stars."""
    
    sky_cuts = []
    if l_min is not None and l_max is not None:
        # Handle l wrap-around at 0/360
        if l_min < l_max:
            sky_cuts.append(f"g.l BETWEEN {l_min} AND {l_max}")
        else:
            # Wraps around 360
            sky_cuts.append(f"(g.l >= {l_min} OR g.l <= {l_max})")
    if b_min is not None and b_max is not None:
        sky_cuts.append(f"g.b BETWEEN {b_min} AND {b_max}")
    
    sky_where = " AND ".join(sky_cuts) if sky_cuts else "1=1"
    
    quality_cuts = f"AND g.ruwe < {ruwe_max} AND g.visibility_periods_used >= {vis_min}"
    
    if max_stars is not None:
        limit_clause = f"TOP {max_stars}"
        order_by = "\nORDER BY g.random_index" if use_random else ""
    else:
        limit_clause = ""
        order_by = ""
    
    sky_clause = f"AND ({sky_where})" if sky_cuts else ""
    
    query = BASE_QUERY_6D.format(
        sky_cuts=sky_clause,
        quality_cuts=quality_cuts,
        limit_clause=limit_clause,
        order_by=order_by,
    )
    
    return query


def download_chunk(
    Gaia,
    query: str,
    output_path: Path,
    retry_max: int = 3,
) -> Optional[Table]:
    """Download one chunk from Gaia TAP with retries."""
    
    for attempt in range(1, retry_max + 1):
        try:
            print(f"  Submitting query (attempt {attempt}/{retry_max})...")
            job = Gaia.launch_job_async(
                query,
                dump_to_file=False,
                # Use default format (VOTable), we'll convert to FITS when saving
            )
            
            print(f"  Waiting for results...")
            results = job.get_results()
            
            if results is None or len(results) == 0:
                print(f"  No results returned")
                return None
            
            print(f"  Retrieved {len(results):,} stars")
            return results
            
        except Exception as e:
            if attempt == retry_max:
                print(f"  Failed after {retry_max} attempts: {e}")
                return None
            wait_time = 5 * attempt
            print(f"  Error (attempt {attempt}): {e}")
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return None


def download_gaia_for_brava(
    Gaia,
    brava_catalog_path: Path,
    output_path: Path,
    ruwe_max: float = 1.6,
    vis_min: int = 8,
    chunk_size: int = 5000,
) -> None:
    """Download Gaia 6D stars that match BRAVA catalog via 2MASS IDs.
    
    This uses the pre-computed Gaia-2MASS cross-match table to find
    Gaia stars with 6D information (radial velocity) that match BRAVA stars.
    """
    from astropy.io import ascii
    from astropy.table import Table
    
    print(f"\n{'='*70}")
    print("DOWNLOADING GAIA 6D STARS FOR BRAVA CATALOG")
    print(f"{'='*70}")
    
    # Load BRAVA catalog
    print(f"\nLoading BRAVA catalog: {brava_catalog_path}")
    try:
        brava = ascii.read(str(brava_catalog_path), format='ipac')
    except Exception as e:
        print(f"ERROR: Failed to load BRAVA catalog: {e}")
        return
    
    print(f"  Loaded {len(brava):,} BRAVA stars")
    
    # Extract 2MASS IDs
    if 'tmass_id' not in brava.colnames:
        print("ERROR: BRAVA catalog missing 'tmass_id' column")
        return
    
    tmass_ids = brava['tmass_id']
    # Filter out null/masked values and convert to strings
    if hasattr(tmass_ids, 'mask'):
        valid_mask = ~tmass_ids.mask
    else:
        # Try to handle different data types
        try:
            valid_mask = np.isfinite(tmass_ids.astype(float))
        except:
            valid_mask = np.array([str(x).strip() not in ['', 'None', 'nan', 'NULL'] for x in tmass_ids])
    
    tmass_ids_valid = tmass_ids[valid_mask]
    # Convert to strings (2MASS IDs are typically stored as strings)
    tmass_ids_valid = np.array([str(x).strip() for x in tmass_ids_valid])
    
    print(f"  Found {len(tmass_ids_valid):,} valid 2MASS IDs")
    
    if len(tmass_ids_valid) == 0:
        print("ERROR: No valid 2MASS IDs found")
        return
    
    # Process in chunks
    n_chunks = (len(tmass_ids_valid) + chunk_size - 1) // chunk_size
    print(f"\nProcessing {len(tmass_ids_valid):,} 2MASS IDs in {n_chunks} chunk(s)...")
    
    all_results = []
    
    for i in range(0, len(tmass_ids_valid), chunk_size):
        chunk_ids = tmass_ids_valid[i:min(i+chunk_size, len(tmass_ids_valid))]
        chunk_num = i // chunk_size + 1
        
        print(f"\n  Chunk {chunk_num}/{n_chunks}: {len(chunk_ids):,} 2MASS IDs")
        
        # Create upload table
        upload_table = Table({'tmass_id': chunk_ids})
        
        # Build query
        quality_cuts = f"AND g.ruwe < {ruwe_max} AND g.visibility_periods_used >= {vis_min}"
        query = BASE_QUERY_6D_2MASS.format(
            limit_clause="",
            quality_cuts=quality_cuts,
            order_by="",
        )
        
        # Download chunk
        for attempt in range(1, 4):
            try:
                print(f"    Submitting query (attempt {attempt}/3)...")
                job = Gaia.launch_job_async(
                    query,
                    upload_resource=upload_table,
                    upload_table_name="tmass_ids",
                    dump_to_file=False,
                )
                
                print(f"    Waiting for results...")
                results = job.get_results()
                
                if results is not None and len(results) > 0:
                    print(f"    Retrieved {len(results):,} Gaia stars with 6D info")
                    all_results.append(results)
                else:
                    print(f"    No matches found in this chunk")
                
                break  # Success
                
            except Exception as e:
                if attempt == 3:
                    print(f"    Failed after 3 attempts: {e}")
                else:
                    wait_time = 5 * attempt
                    print(f"    Error (attempt {attempt}): {e}")
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # Be nice to TAP server
        time.sleep(1)
    
    # Combine all results
    if not all_results:
        print("\nNo Gaia stars found matching BRAVA 2MASS IDs")
        return
    
    print(f"\nCombining {len(all_results)} chunks...")
    combined = vstack(all_results, metadata_conflicts='silent')
    
    print(f"  Total Gaia stars with 6D info: {len(combined):,}")
    print(f"  Match rate: {len(combined):,}/{len(brava):,} ({len(combined)/len(brava)*100:.1f}%)")
    
    # Save to FITS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fix string columns for FITS compatibility (FITS doesn't like variable-length strings)
    if 'tmass_id' in combined.colnames:
        # Convert tmass_id to fixed-length string array
        tmass_str = [str(x) for x in combined['tmass_id']]
        max_len = max(len(s) for s in tmass_str) if tmass_str else 20
        # Pad to fixed length
        tmass_fixed = [s.ljust(max_len)[:max_len] for s in tmass_str]
        combined['tmass_id'] = tmass_fixed
    
    combined.write(str(output_path), overwrite=True, format='fits')
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Saved: {output_path}")
    print(f"  Stars: {len(combined):,}")
    print(f"  Columns: {len(combined.colnames)}")
    print(f"{'='*70}")


def download_by_lb_bins(
    Gaia,
    output_dir: Path,
    l_bin_size: float = 30.0,
    b_bin_size: float = 10.0,
    ruwe_max: float = 1.6,
    vis_min: int = 8,
    max_stars_per_bin: Optional[int] = None,
) -> None:
    """Download 6D stars by splitting sky into l/b bins."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create l/b bins
    l_bins = np.arange(0, 360, l_bin_size)
    b_bins = np.arange(-90, 90, b_bin_size)
    
    total_bins = len(l_bins) * len(b_bins)
    print(f"\nDownloading 6D Gaia catalog in {total_bins} sky bins...")
    print(f"  l bins: {len(l_bins)} ({l_bin_size} deg each)")
    print(f"  b bins: {len(b_bins)} ({b_bin_size} deg each)")
    print(f"  Output directory: {output_dir}")
    
    all_files = []
    total_stars = 0
    
    pbar = tqdm(total=total_bins, desc="Downloading bins")
    
    for i, l_min in enumerate(l_bins):
        l_max = l_min + l_bin_size if i < len(l_bins) - 1 else 360.0
        
        for j, b_min in enumerate(b_bins):
            b_max = b_min + b_bin_size if j < len(b_bins) - 1 else 90.0
            
            # Build query for this bin
            query = build_query(
                l_min=l_min,
                l_max=l_max,
                b_min=b_min,
                b_max=b_max,
                ruwe_max=ruwe_max,
                vis_min=vis_min,
                max_stars=max_stars_per_bin,
                use_random=True,
            )
            
            # Download chunk
            output_file = output_dir / f"gaia_6d_l{l_min:06.1f}_{l_max:06.1f}_b{b_min:+06.1f}_{b_max:+06.1f}.fits"
            
            if output_file.exists():
                print(f"\n  Skipping {output_file.name} (already exists)")
                # Count existing file
                try:
                    existing = Table.read(str(output_file))
                    total_stars += len(existing)
                    all_files.append(output_file)
                except:
                    pass
                pbar.update(1)
                continue
            
            print(f"\n  Bin: l=[{l_min:.1f}, {l_max:.1f}], b=[{b_min:.1f}, {b_max:.1f}]")
            
            results = download_chunk(Gaia, query, output_file)
            
            if results is not None and len(results) > 0:
                # Save to FITS
                results.write(str(output_file), overwrite=True, format='fits')
                print(f"  Saved: {output_file.name} ({len(results):,} stars)")
                total_stars += len(results)
                all_files.append(output_file)
            else:
                print(f"  No stars found in this bin")
            
            pbar.update(1)
            
            # Be nice to TAP server
            time.sleep(1)
    
    pbar.close()
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total stars downloaded: {total_stars:,}")
    print(f"Files created: {len(all_files)}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo combine all files into one:")
    print(f"  python scripts/combine_gaia_fits.py {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download full Gaia DR3 6D catalog (470M stars with radial velocities)"
    )
    parser.add_argument(
        "--brava-catalog",
        type=str,
        default=None,
        help="Path to BRAVA catalog file. If provided, only downloads Gaia stars matching BRAVA via 2MASS IDs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output FITS file path (required when using --brava-catalog)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/gaia/6d_full",
        help="Output directory for FITS files (default: data/gaia/6d_full). Ignored if --brava-catalog is used.",
    )
    parser.add_argument(
        "--l-min",
        type=float,
        default=None,
        help="Minimum Galactic longitude (0-360 deg). If not set, downloads all sky in bins.",
    )
    parser.add_argument(
        "--l-max",
        type=float,
        default=None,
        help="Maximum Galactic longitude (0-360 deg)",
    )
    parser.add_argument(
        "--b-min",
        type=float,
        default=None,
        help="Minimum Galactic latitude (-90 to 90 deg)",
    )
    parser.add_argument(
        "--b-max",
        type=float,
        default=None,
        help="Maximum Galactic latitude (-90 to 90 deg)",
    )
    parser.add_argument(
        "--l-bin-size",
        type=float,
        default=30.0,
        help="Longitude bin size in degrees (default: 30.0)",
    )
    parser.add_argument(
        "--b-bin-size",
        type=float,
        default=10.0,
        help="Latitude bin size in degrees (default: 10.0)",
    )
    parser.add_argument(
        "--ruwe-max",
        type=float,
        default=1.6,
        help="Maximum RUWE for quality cut (default: 1.6)",
    )
    parser.add_argument(
        "--vis-min",
        type=int,
        default=8,
        help="Minimum visibility periods (default: 8)",
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=None,
        help="Maximum stars per bin (for testing). If not set, downloads all.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download only 10k stars total",
    )
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        args.max_stars = 10000
        args.l_bin_size = 360.0  # Single bin
        args.b_bin_size = 180.0  # Single bin
        print("TEST MODE: Downloading 10k stars only")
    
    # Initialize Gaia TAP
    print("Initializing Gaia TAP connection...")
    try:
        from astroquery.gaia import Gaia as GaiaClass
        GaiaClass.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        print("Connected to Gaia TAP")
    except Exception as e:
        print(f"ERROR: Failed to connect to Gaia TAP: {e}")
        print("Make sure astroquery is installed: pip install astroquery")
        sys.exit(1)
    
    # If BRAVA catalog provided, download only matching Gaia stars
    if args.brava_catalog is not None:
        brava_path = Path(args.brava_catalog)
        if not brava_path.exists():
            print(f"ERROR: BRAVA catalog not found: {brava_path}")
            sys.exit(1)
        
        if args.output is None:
            print("ERROR: --output is required when using --brava-catalog")
            sys.exit(1)
        
        output_path = Path(args.output)
        download_gaia_for_brava(
            GaiaClass,
            brava_path,
            output_path,
            ruwe_max=args.ruwe_max,
            vis_min=args.vis_min,
            chunk_size=5000,
        )
        return
    
    output_dir = Path(args.output_dir)
    
    # If specific region requested, download just that
    if args.l_min is not None and args.l_max is not None:
        print(f"\nDownloading specific region: l=[{args.l_min}, {args.l_max}], b=[{args.b_min}, {args.b_max}]")
        
        query = build_query(
            l_min=args.l_min,
            l_max=args.l_max,
            b_min=args.b_min,
            b_max=args.b_max,
            ruwe_max=args.ruwe_max,
            vis_min=args.vis_min,
            max_stars=args.max_stars,
        )
        
        output_file = output_dir / f"gaia_6d_l{args.l_min:.1f}_{args.l_max:.1f}_b{args.b_min:.1f}_{args.b_max:.1f}.fits"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = download_chunk(GaiaClass, query, output_file)
        
        if results is not None and len(results) > 0:
            results.write(str(output_file), overwrite=True, format='fits')
            print(f"\nSaved: {output_file} ({len(results):,} stars)")
        else:
            print("\nNo stars found")
    
    else:
        # Download full sky in bins
        download_by_lb_bins(
            GaiaClass,
            output_dir,
            l_bin_size=args.l_bin_size,
            b_bin_size=args.b_bin_size,
            ruwe_max=args.ruwe_max,
            vis_min=args.vis_min,
            max_stars_per_bin=args.max_stars,
        )


if __name__ == "__main__":
    main()

