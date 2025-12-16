#!/usr/bin/env python3
"""
Cross-match multiple bulge kinematics catalogs.

Combines BRAVA, APOGEE, GIBS, and Gaia DR3 for comprehensive analysis.
Creates unified catalog with full 6D phase space where available.
"""

import sys
from pathlib import Path
import numpy as np
from astropy.table import Table, join, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import ascii

PROJECT_ROOT = Path(__file__).parent.parent
BULGE_DIR = PROJECT_ROOT / "data" / "bulge_kinematics"
OUTPUT_DIR = BULGE_DIR / "crossmatched"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_brava():
    """Load BRAVA catalog."""
    catalog_path = BULGE_DIR / "BRAVA" / "brava_catalog.tbl"
    if not catalog_path.exists():
        return None
    
    table = ascii.read(catalog_path, format='ipac')
    
    # Standardize column names
    if 'RA' in table.colnames:
        table['ra'] = table['RA']
    if 'DEC' in table.colnames:
        table['dec'] = table['DEC']
    if 'vhc' in table.colnames:
        table['vhelio'] = table['vhc']
    
    # Add catalog identifier
    table['catalog'] = 'BRAVA'
    
    return table


def load_apogee():
    """Load APOGEE catalog."""
    for dr in [18, 17]:
        catalog_path = BULGE_DIR / "APOGEE" / f"apogee_bulge_dr{dr}.fits"
        if catalog_path.exists():
            table = Table.read(catalog_path)
            table['catalog'] = f'APOGEE_DR{dr}'
            return table
    return None


def load_gibs():
    """Load GIBS catalog."""
    catalog_path = BULGE_DIR / "GIBS" / "gibs_catalog_1.fits"
    if not catalog_path.exists():
        return None
    
    table = Table.read(catalog_path)
    table['catalog'] = 'GIBS'
    
    return table


def crossmatch_catalogs(table1, table2, max_sep_arcsec=2.0, label1="Catalog 1", label2="Catalog 2"):
    """
    Cross-match two catalogs by position.
    
    Returns matched table with columns from both catalogs.
    """
    if 'ra' not in table1.colnames or 'dec' not in table1.colnames:
        print(f"⚠ {label1} missing ra/dec columns")
        return None
    if 'ra' not in table2.colnames or 'dec' not in table2.colnames:
        print(f"⚠ {label2} missing ra/dec columns")
        return None
    
    coords1 = SkyCoord(ra=table1['ra']*u.deg, dec=table1['dec']*u.deg)
    coords2 = SkyCoord(ra=table2['ra']*u.deg, dec=table2['dec']*u.deg)
    
    idx, sep2d, _ = coords1.match_to_catalog_sky(coords2)
    sep_arcsec = sep2d.arcsec
    
    good = sep_arcsec < max_sep_arcsec
    n_matched = good.sum()
    
    print(f"  {n_matched} stars matched between {label1} and {label2}")
    
    if n_matched == 0:
        return None
    
    # Create matched table
    matched = table1[good].copy()
    
    # Add columns from table2 (with prefix to avoid conflicts)
    for col in table2.colnames:
        if col not in ['ra', 'dec']:  # Don't duplicate coordinates
            new_col = f"{label2.lower()}_{col}" if col in matched.colnames else col
            matched[new_col] = table2[col][idx[good]]
    
    matched['sep_arcsec'] = sep_arcsec[good]
    
    return matched


def main():
    print("="*70)
    print("CROSS-MATCHING BULGE KINEMATICS CATALOGS")
    print("="*70)
    
    # Load catalogs
    print("\nLoading catalogs...")
    brava = load_brava()
    apogee = load_apogee()
    gibs = load_gibs()
    
    catalogs_loaded = {
        'BRAVA': brava is not None,
        'APOGEE': apogee is not None,
        'GIBS': gibs is not None
    }
    
    for name, loaded in catalogs_loaded.items():
        status = "✓" if loaded else "✗"
        count = len(brava) if name == 'BRAVA' and loaded else \
                len(apogee) if name == 'APOGEE' and loaded else \
                len(gibs) if name == 'GIBS' and loaded else 0
        print(f"{status} {name}: {count:,} stars" if loaded else f"{status} {name}: Not found")
    
    if not any(catalogs_loaded.values()):
        print("\n✗ No catalogs found. Please download catalogs first.")
        return
    
    # Cross-match BRAVA + APOGEE (recommended starting point)
    if brava is not None and apogee is not None:
        print("\n" + "-"*70)
        print("Cross-matching BRAVA + APOGEE...")
        print("-"*70)
        
        matched = crossmatch_catalogs(brava, apogee, max_sep_arcsec=2.0, 
                                     label1="BRAVA", label2="APOGEE")
        
        if matched is not None:
            output_path = OUTPUT_DIR / "brava_apogee_crossmatch.fits"
            matched.write(str(output_path), overwrite=True)
            print(f"✓ Saved: {output_path}")
            print(f"  {len(matched)} stars in cross-match")
    
    # Cross-match BRAVA + GIBS
    if brava is not None and gibs is not None:
        print("\n" + "-"*70)
        print("Cross-matching BRAVA + GIBS...")
        print("-"*70)
        
        matched = crossmatch_catalogs(brava, gibs, max_sep_arcsec=2.0,
                                     label1="BRAVA", label2="GIBS")
        
        if matched is not None:
            output_path = OUTPUT_DIR / "brava_gibs_crossmatch.fits"
            matched.write(str(output_path), overwrite=True)
            print(f"✓ Saved: {output_path}")
            print(f"  {len(matched)} stars in cross-match")
    
    # Create combined catalog (all unique stars)
    print("\n" + "-"*70)
    print("Creating combined catalog...")
    print("-"*70)
    
    all_catalogs = []
    if brava is not None:
        all_catalogs.append(brava)
    if apogee is not None:
        all_catalogs.append(apogee)
    if gibs is not None:
        all_catalogs.append(gibs)
    
    if len(all_catalogs) > 1:
        # Combine all catalogs (will have duplicate stars)
        combined = vstack(all_catalogs)
        output_path = OUTPUT_DIR / "all_bulge_catalogs_combined.fits"
        combined.write(str(output_path), overwrite=True)
        print(f"✓ Saved: {output_path}")
        print(f"  {len(combined)} total entries (may include duplicates)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Add Gaia DR3 proper motions:")
    print("   python scripts/add_gaia_proper_motions.py --catalog BRAVA")
    print("   python scripts/add_gaia_proper_motions.py --catalog APOGEE")
    print("\n2. Cross-matched catalogs saved to:")
    print(f"   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

