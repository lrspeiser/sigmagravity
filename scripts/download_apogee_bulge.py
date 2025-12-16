#!/usr/bin/env python3
"""
Download APOGEE inner bulge fields for R < 3 kpc analysis.

Queries APOGEE DR18/DR17 for stars in bulge region: |l| < 10°, |b| < 10°
Uses astroquery to access SDSS CasJobs database.
"""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bulge_kinematics" / "APOGEE"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_apogee_bulge(max_stars=50000, data_release=18):
    """
    Download APOGEE bulge fields.
    
    Parameters
    ----------
    max_stars : int
        Maximum number of stars to download (default 50,000)
    data_release : int
        APOGEE data release (17 or 18, default 18)
    """
    try:
        from astroquery.sdss import SDSS
    except ImportError:
        print("Installing astroquery...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "astroquery"])
        from astroquery.sdss import SDSS
    
    print("="*70)
    print("DOWNLOADING APOGEE BULGE FIELDS")
    print("="*70)
    print(f"Data Release: DR{data_release}")
    print(f"Max stars: {max_stars:,}")
    print(f"Coverage: |l| < 10°, |b| < 10°")
    print(f"Output: {DATA_DIR}")
    print()
    
    # APOGEE table name depends on data release
    if data_release == 18:
        table_name = "apogee_dr18.allStar"
    elif data_release == 17:
        table_name = "apogee_dr17.allStar"
    else:
        raise ValueError(f"Data release {data_release} not supported. Use 17 or 18.")
    
    # Query for bulge fields
    # Key columns: positions, velocities, chemistry, stellar parameters
    query = f"""
    SELECT TOP {max_stars}
        apogee_id,
        ra, dec,
        glon AS l,
        glat AS b,
        vhelio_avg AS vhelio,
        vscatter AS vhelio_err,
        fe_h,
        alpha_m AS alpha_fe,
        teff,
        logg,
        snr,
        nvisits,
        j, h, k AS ks,
        dist AS distance,
        dist_err AS distance_err
    FROM {table_name}
    WHERE 
        ABS(glon) < 10 
        AND ABS(glat) < 10
        AND vhelio_avg > -900  -- Valid radial velocity
        AND snr > 10  -- Good signal-to-noise
    ORDER BY snr DESC  -- Prioritize high-quality spectra
    """
    
    print("Query:")
    print("-"*70)
    print(query)
    print("-"*70)
    print()
    
    try:
        print("Submitting query to SDSS CasJobs...")
        print("(This may take several minutes)")
        
        result = SDSS.query_sql(query, data_release=data_release)
        
        if result is None or len(result) == 0:
            print("⚠ Query returned no results")
            print("  Possible reasons:")
            print("  - Network issue")
            print("  - Authentication required")
            print("  - Try using CasJobs web interface: https://skyserver.sdss.org/casjobs/")
            return False
        
        print(f"✓ Retrieved {len(result):,} stars")
        
        # Save to FITS
        output_path = DATA_DIR / f"apogee_bulge_dr{data_release}.fits"
        result.write(str(output_path), overwrite=True)
        print(f"✓ Saved to: {output_path}")
        
        # Also save summary statistics
        summary = {
            "dataset": "APOGEE",
            "data_release": data_release,
            "n_stars": len(result),
            "coverage": "|l| < 10°, |b| < 10°",
            "columns": list(result.colnames),
            "file": str(output_path),
            "query": query
        }
        
        import json
        summary_path = DATA_DIR / f"apogee_bulge_dr{data_release}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_path}")
        
        # Print basic statistics
        print("\nBasic Statistics:")
        print("-"*70)
        if 'vhelio' in result.colnames:
            valid_rv = result['vhelio'] > -900
            print(f"  Stars with valid RV: {valid_rv.sum():,}")
        if 'fe_h' in result.colnames:
            valid_fe = ~result['fe_h'].mask if hasattr(result['fe_h'], 'mask') else result['fe_h'] > -999
            print(f"  Stars with [Fe/H]: {valid_fe.sum() if hasattr(valid_fe, 'sum') else sum(valid_fe):,}")
        if 'snr' in result.colnames:
            print(f"  Median SNR: {result['snr'].median():.1f}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error querying APOGEE: {e}")
        print("\nAlternative: Use CasJobs web interface")
        print("1. Visit: https://skyserver.sdss.org/casjobs/")
        print("2. Create account and log in")
        print("3. Run the SQL query above")
        print("4. Download results as FITS")
        print(f"5. Save to: {DATA_DIR}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download APOGEE inner bulge fields"
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=50000,
        help="Maximum number of stars to download (default: 50000)"
    )
    parser.add_argument(
        "--dr",
        type=int,
        choices=[17, 18],
        default=18,
        help="APOGEE data release (default: 18)"
    )
    
    args = parser.parse_args()
    
    success = download_apogee_bulge(max_stars=args.max_stars, data_release=args.dr)
    
    if success:
        print("="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("1. Cross-match with BRAVA: python data/bulge_kinematics/crossmatch_brava_apogee_gaia.py")
        print("2. Add Gaia DR3 proper motions: python scripts/add_gaia_proper_motions.py")
    else:
        print("\n⚠ Download incomplete. See instructions above for manual download.")
        sys.exit(1)


if __name__ == "__main__":
    main()

