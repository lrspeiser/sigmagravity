"""
Fetch large Gaia DR3 sample for comprehensive MW test.
Targets multiple sky regions to get ~500k-1M quality stars.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def fetch_gaia_wedges():
    """
    Fetch multiple Gaia wedges to build comprehensive sample.
    Aims for ~500k-1M stars with good spatial coverage.
    """
    
    print("="*80)
    print("FETCHING COMPREHENSIVE GAIA DR3 SAMPLE")
    print("="*80)
    
    # Define wedges to fetch
    # Strategy: Get good coverage of inner disk, outer disk, and anticenter
    wedges = [
        # Inner disk (bulge region)
        {'region': 'inner', 'kind': 'pm', 'top_n': 300000, 'desc': 'Inner disk (PM)'},
        {'region': 'inner', 'kind': 'rvs', 'top_n': 100000, 'desc': 'Inner disk (RVs)'},
        
        # Anticenter (outer disk, away from bulge)
        {'region': 'anticenter', 'kind': 'pm', 'top_n': 200000, 'desc': 'Anticenter (PM)'},
        {'region': 'anticenter', 'kind': 'rvs', 'top_n': 50000, 'desc': 'Anticenter (RVs)'},
    ]
    
    print(f"\nPlanned wedges ({len(wedges)} total):")
    total_planned = 0
    for i, w in enumerate(wedges, 1):
        print(f"  {i}. {w['desc']}: {w['top_n']:,} stars")
        total_planned += w['top_n']
    
    print(f"\nTotal planned: {total_planned:,} stars")
    print("\nThis will take ~5-10 minutes to download from Gaia TAP.")
    
    response = input("\nProceed with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Fetch each wedge
    print("\n" + "="*80)
    print("DOWNLOADING WEDGES")
    print("="*80)
    
    for i, w in enumerate(wedges, 1):
        print(f"\n[{i}/{len(wedges)}] Fetching {w['desc']}...")
        
        cmd = [
            sys.executable,
            'scripts/fetch_gaia_wedges.py',
            '--region', w['region'],
            '--kind', w['kind'],
            '--top-n', str(w['top_n'])
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e}")
            print(e.stderr)
            continue
    
    print("\n" + "="*80)
    print("MERGING WEDGES")
    print("="*80)
    
    # Load and merge all wedges
    wedge_dir = Path('data/gaia/new')
    wedge_files = list(wedge_dir.glob('gaia_*.csv'))
    
    print(f"\nFound {len(wedge_files)} wedge files:")
    
    all_stars = []
    for f in wedge_files:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df):,} stars")
        all_stars.append(df)
    
    combined = pd.concat(all_stars, ignore_index=True)
    
    # Remove duplicates (some stars may appear in multiple wedges)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset='source_id', keep='first')
    n_after = len(combined)
    
    if n_before > n_after:
        print(f"\nRemoved {n_before - n_after:,} duplicate stars")
    
    print(f"\nTotal unique stars: {n_after:,}")
    
    # Save merged sample
    output_path = 'data/gaia/gaia_large_sample.csv'
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"\nNext: python GravityWaveTest/prepare_large_gaia_sample.py")

if __name__ == "__main__":
    print("\n⚠️  NOTE: This will download ~500k-1M stars from Gaia TAP")
    print("⚠️  Requires internet connection and ~5-10 minutes")
    print("\nAlternatively, run with existing 144k sample for faster testing.")
    
    choice = input("\nFetch large sample (L) or use existing 144k (E)? [L/e]: ").strip().lower()
    
    if choice == 'e':
        print("\nUsing existing 144k sample. Run:")
        print("  python GravityWaveTest/test_multicomponent_mw.py")
    else:
        fetch_gaia_wedges()

