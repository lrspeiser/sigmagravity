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
    
    coords = SkyCoord(
        ra=catalog_table['ra'] * u.deg,
        dec=catalog_table['dec'] * u.deg,
        frame='icrs'
    )
    
    # Limit for testing
    if max_stars and len(coords) > max_stars:
        print(f"Limiting to first {max_stars} stars for testing...")
        coords = coords[:max_stars]
        catalog_table = catalog_table[:max_stars]
    
    # Query Gaia in batches (Gaia has limits on number of sources)
    batch_size = 5000
    n_batches = (len(coords) + batch_size - 1) // batch_size
    
    all_results = []
    
    for i in tqdm(range(n_batches), desc="Querying Gaia"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(coords))
        batch_coords = coords[start_idx:end_idx]
        
        try:
            # Use Gaia's cross-match functionality
            radius = u.Quantity(max_sep_arcsec, u.arcsec)
            j = Gaia.cone_search_async(
                coordinates=batch_coords,
                radius=radius,
                table="gaiadr3.gaia_source",
                columns=[
                    "source_id",
                    "ra", "dec",
                    "parallax", "parallax_error",
                    "pmra", "pmra_error",
                    "pmdec", "pmdec_error",
                    "radial_velocity", "radial_velocity_error",
                    "phot_g_mean_mag",
                    "ruwe",
                    "visibility_periods_used"
                ]
            )
            
            result = j.get_results()
            
            if len(result) > 0:
                all_results.append(result)
        
        except Exception as e:
            print(f"\n⚠ Error in batch {i+1}/{n_batches}: {e}")
            continue
    
    if not all_results:
        print("⚠ No Gaia matches found")
        return None
    
    # Combine results
    gaia_table = Table.vstack(all_results)
    print(f"  Found {len(gaia_table)} Gaia matches")
    
    # Cross-match with original catalog
    gaia_coords = SkyCoord(
        ra=gaia_table['ra'] * u.deg,
        dec=gaia_table['dec'] * u.deg,
        frame='icrs'
    )
    
    idx, sep2d, _ = coords.match_to_catalog_sky(gaia_coords)
    sep_arcsec = sep2d.arcsec
    
    # Only keep good matches
    good = sep_arcsec < max_sep_arcsec
    print(f"  {good.sum()} stars matched within {max_sep_arcsec} arcsec")
    
    # Create matched table
    matched_table = catalog_table[good].copy()
    
    # Add Gaia columns (rename to avoid conflicts)
    for col in ['source_id', 'parallax', 'parallax_error', 
                'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                'radial_velocity', 'radial_velocity_error',
                'phot_g_mean_mag', 'ruwe', 'visibility_periods_used']:
        if col in gaia_table.colnames:
            matched_table[f'gaia_{col}'] = gaia_table[col][idx[good]]
    
    matched_table['gaia_sep_arcsec'] = sep_arcsec[good]
    
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

