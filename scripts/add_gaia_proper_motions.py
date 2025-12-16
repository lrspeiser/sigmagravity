#!/usr/bin/env python3
"""
Add Gaia DR3 proper motions to bulge kinematics catalogs.

Cross-matches existing catalogs (BRAVA, APOGEE, GIBS) with Gaia DR3
to add proper motions for full 6D phase space analysis.

Usage:
    python scripts/add_gaia_proper_motions.py --catalog BRAVA
    python scripts/add_gaia_proper_motions.py --catalog APOGEE
    python scripts/add_gaia_proper_motions.py --catalog GIBS
    python scripts/add_gaia_proper_motions.py --all
"""

import sys
from pathlib import Path
import argparse
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
BULGE_DIR = PROJECT_ROOT / "data" / "bulge_kinematics"
OUTPUT_DIR = BULGE_DIR / "crossmatched"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_catalog(catalog_name):
    """Load a bulge kinematics catalog."""
    if catalog_name.upper() == "BRAVA":
        catalog_path = BULGE_DIR / "BRAVA" / "brava_catalog.tbl"
        if not catalog_path.exists():
            raise FileNotFoundError(f"BRAVA catalog not found: {catalog_path}")
        
        # Read IPAC table format
        from astropy.io import ascii
        table = ascii.read(catalog_path, format='ipac')
        
        # Standardize column names
        if 'ra' in table.colnames and 'dec' in table.colnames:
            pass  # Already have ra, dec
        elif 'RA' in table.colnames and 'DEC' in table.colnames:
            table['ra'] = table['RA']
            table['dec'] = table['DEC']
        
        return table, "BRAVA"
    
    elif catalog_name.upper() == "APOGEE":
        # Try DR18 first, then DR17
        for dr in [18, 17]:
            catalog_path = BULGE_DIR / "APOGEE" / f"apogee_bulge_dr{dr}.fits"
            if catalog_path.exists():
                table = Table.read(catalog_path)
                return table, f"APOGEE_DR{dr}"
        
        raise FileNotFoundError("APOGEE catalog not found. Run download_apogee_bulge.py first.")
    
    elif catalog_name.upper() == "GIBS":
        # Load the main GIBS catalog (largest one)
        catalog_path = BULGE_DIR / "GIBS" / "gibs_catalog_1.fits"
        if not catalog_path.exists():
            raise FileNotFoundError(f"GIBS catalog not found: {catalog_path}")
        
        table = Table.read(catalog_path)
        return table, "GIBS"
    
    else:
        raise ValueError(f"Unknown catalog: {catalog_name}")


def crossmatch_gaia(catalog_table, max_sep_arcsec=2.0, max_stars=None):
    """
    Cross-match catalog with Gaia DR3 to get proper motions.
    
    Parameters
    ----------
    catalog_table : astropy.Table
        Input catalog with ra, dec columns
    max_sep_arcsec : float
        Maximum separation for cross-match (arcsec)
    max_stars : int, optional
        Maximum number of stars to query (for testing)
    
    Returns
    -------
    matched_table : astropy.Table
        Catalog with Gaia proper motions added
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        print("Installing astroquery...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "astroquery"])
        from astroquery.gaia import Gaia
    
    print(f"\nCross-matching {len(catalog_table)} stars with Gaia DR3...")
    print(f"Max separation: {max_sep_arcsec} arcsec")
    
    # Get coordinates
    if 'ra' not in catalog_table.colnames or 'dec' not in catalog_table.colnames:
        raise ValueError("Catalog must have 'ra' and 'dec' columns")
    
    # Convert to numpy arrays to avoid unit issues
    ra_values = np.array(catalog_table['ra'], dtype=float)
    dec_values = np.array(catalog_table['dec'], dtype=float)
    
    coords = SkyCoord(
        ra=ra_values * u.deg,
        dec=dec_values * u.deg,
        frame='icrs'
    )
    
    # Limit for testing
    if max_stars and len(coords) > max_stars:
        print(f"Limiting to first {max_stars} stars for testing...")
        coords = coords[:max_stars]
        catalog_table = catalog_table[:max_stars]
    
    # Query Gaia using ADQL with cross-match
    # Use smaller batches to avoid query timeouts
    batch_size = 100
    n_batches = (len(coords) + batch_size - 1) // batch_size
    
    all_results = []
    radius_deg = max_sep_arcsec / 3600.0
    
    print(f"  Querying in {n_batches} batches of {batch_size} stars each...")
    
    for i in tqdm(range(n_batches), desc="Querying Gaia"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(coords))
        batch_coords = coords[start_idx:end_idx]
        
        try:
            # Build ADQL query for this batch
            # Create a UNION of cone searches for each coordinate
            queries = []
            for coord in batch_coords:
                query = f"""
                SELECT TOP 1
                    g.source_id,
                    g.ra, g.dec,
                    g.parallax, g.parallax_error,
                    g.pmra, g.pmra_error,
                    g.pmdec, g.pmdec_error,
                    g.radial_velocity, g.radial_velocity_error,
                    g.phot_g_mean_mag,
                    g.ruwe,
                    g.visibility_periods_used,
                    {coord.ra.deg} AS query_ra,
                    {coord.dec.deg} AS query_dec,
                    DISTANCE(
                        POINT('ICRS', g.ra, g.dec),
                        POINT('ICRS', {coord.ra.deg}, {coord.dec.deg})
                    ) AS sep
                FROM gaiadr3.gaia_source AS g
                WHERE 
                    1 = CONTAINS(
                        POINT('ICRS', g.ra, g.dec),
                        CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius_deg})
                    )
                ORDER BY sep
                """
                queries.append(query)
            
            # Execute queries sequentially (Gaia rate limiting)
            batch_results = []
            for query in queries:
                try:
                    job = Gaia.launch_job_async(query)
                    result = job.get_results()
                    if len(result) > 0:
                        batch_results.append(result)
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(0.1)
                except Exception as e:
                    # Skip individual query errors
                    continue
            
            if batch_results:
                from astropy.table import vstack
                result = vstack(batch_results)
                all_results.append(result)
        
        except Exception as e:
            print(f"\n⚠ Error in batch {i+1}/{n_batches}: {e}")
            continue
    
    if not all_results:
        print("⚠ No Gaia matches found")
        return None
    
    # Combine results
    from astropy.table import vstack
    gaia_table = vstack(all_results)
    print(f"  Found {len(gaia_table)} Gaia matches")
    
    # Match back to original catalog using query_ra/query_dec
    if 'query_ra' not in gaia_table.colnames or 'query_dec' not in gaia_table.colnames:
        print("⚠ Warning: query coordinates not found, using position matching")
        gaia_coords = SkyCoord(
            ra=np.array(gaia_table['ra'], dtype=float) * u.deg,
            dec=np.array(gaia_table['dec'], dtype=float) * u.deg,
            frame='icrs'
        )
        idx, sep2d, _ = coords.match_to_catalog_sky(gaia_coords)
        sep_arcsec = sep2d.arcsec
        good = sep_arcsec < max_sep_arcsec
        matched_indices = np.where(good)[0]
        gaia_indices = idx[good]
    else:
        # Use query coordinates to match back
        query_ra = np.array(gaia_table['query_ra'], dtype=float)
        query_dec = np.array(gaia_table['query_dec'], dtype=float)
        catalog_ra = np.array(catalog_table['ra'], dtype=float)
        catalog_dec = np.array(catalog_table['dec'], dtype=float)
        
        # Find matches by comparing coordinates
        matched_indices = []
        gaia_indices = []
        for i, (qra, qdec) in enumerate(zip(query_ra, query_dec)):
            # Find closest catalog entry
            dra = catalog_ra - qra
            ddec = catalog_dec - qdec
            sep_deg = np.sqrt(dra**2 + ddec**2)
            min_idx = np.argmin(sep_deg)
            sep_arcsec_val = sep_deg[min_idx] * 3600
            
            if sep_arcsec_val < max_sep_arcsec:
                matched_indices.append(min_idx)
                gaia_indices.append(i)
        
        matched_indices = np.array(matched_indices)
        gaia_indices = np.array(gaia_indices)
        sep_arcsec = np.array([np.sqrt((catalog_ra[i] - query_ra[j])**2 + 
                                      (catalog_dec[i] - query_dec[j])**2) * 3600
                              for i, j in zip(matched_indices, gaia_indices)])
    
    print(f"  {len(matched_indices)} stars matched within {max_sep_arcsec} arcsec")
    
    # Create matched table
    matched_table = catalog_table[matched_indices].copy()
    
    # Add Gaia columns (rename to avoid conflicts)
    for col in ['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 
                'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                'radial_velocity', 'radial_velocity_error',
                'phot_g_mean_mag', 'ruwe', 'visibility_periods_used', 'sep']:
        if col in gaia_table.colnames:
            new_col = f'gaia_{col}' if col in matched_table.colnames else col
            matched_table[new_col] = gaia_table[col][gaia_indices]
    
    matched_table['gaia_sep_arcsec'] = sep_arcsec
    
    return matched_table


def process_catalog(catalog_name, max_sep_arcsec=2.0, max_stars=None):
    """Process a single catalog."""
    print("="*70)
    print(f"PROCESSING {catalog_name.upper()}")
    print("="*70)
    
    try:
        catalog_table, catalog_label = load_catalog(catalog_name)
        print(f"Loaded {len(catalog_table)} stars from {catalog_label}")
    except Exception as e:
        print(f"✗ Error loading catalog: {e}")
        return False
    
    # Cross-match with Gaia
    matched_table = crossmatch_gaia(
        catalog_table,
        max_sep_arcsec=max_sep_arcsec,
        max_stars=max_stars
    )
    
    if matched_table is None:
        return False
    
    # Save result
    output_path = OUTPUT_DIR / f"{catalog_label.lower()}_with_gaia.fits"
    matched_table.write(str(output_path), overwrite=True)
    print(f"\n✓ Saved cross-matched catalog: {output_path}")
    print(f"  {len(matched_table)} stars with Gaia proper motions")
    
    # Print statistics
    if 'gaia_pmra' in matched_table.colnames:
        valid_pm = ~matched_table['gaia_pmra'].mask if hasattr(matched_table['gaia_pmra'], 'mask') else True
        print(f"  Stars with proper motions: {valid_pm.sum() if hasattr(valid_pm, 'sum') else len(matched_table)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add Gaia DR3 proper motions to bulge kinematics catalogs"
    )
    parser.add_argument(
        "--catalog",
        choices=['BRAVA', 'APOGEE', 'GIBS'],
        help="Catalog to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available catalogs"
    )
    parser.add_argument(
        "--max-sep",
        type=float,
        default=2.0,
        help="Maximum separation for cross-match (arcsec, default: 2.0)"
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        help="Maximum number of stars to process (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        catalogs = ['BRAVA', 'APOGEE', 'GIBS']
    elif args.catalog:
        catalogs = [args.catalog]
    else:
        parser.print_help()
        sys.exit(1)
    
    results = {}
    for catalog in catalogs:
        results[catalog] = process_catalog(
            catalog,
            max_sep_arcsec=args.max_sep,
            max_stars=args.max_stars
        )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for catalog, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {catalog}")
    
    print(f"\nCross-matched catalogs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

