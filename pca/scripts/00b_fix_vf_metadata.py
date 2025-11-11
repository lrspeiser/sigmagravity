#!/usr/bin/env python3
"""
Fix missing/zero Vf values by estimating from outer rotation curve.
For galaxies with Vf <= 0 or NaN, compute Vf as median V over R/Rd ∈ [4, 6].
"""
import numpy as np
import pandas as pd
from pathlib import Path

def estimate_vf_from_curve(curve_file):
    """Estimate Vf from outer rotation curve (median V at R/Rd >= 4)"""
    try:
        df = pd.read_csv(curve_file)
        # We need Rd to compute R/Rd, but at this stage we're working with raw R_kpc
        # So instead, just take median of outer 30% of available radii
        if len(df) < 5:
            return np.nan
        
        # Sort by radius
        df = df.sort_values('R_kpc')
        # Take outer 30% of points
        n_outer = max(3, int(len(df) * 0.3))
        outer_points = df.tail(n_outer)
        
        # Return median velocity from outer region
        vf_est = outer_points['V_obs'].median()
        return vf_est if np.isfinite(vf_est) and vf_est > 0 else np.nan
    except Exception as e:
        print(f"Warning: Could not estimate Vf from {curve_file}: {e}")
        return np.nan

def main():
    repo_root = Path(__file__).parent.parent.parent
    meta_file = repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'
    curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    
    print("Loading metadata...")
    meta = pd.read_csv(meta_file)
    
    # Find problematic Vf values
    bad_mask = (meta['Vf'].isna()) | (meta['Vf'] <= 0.0)
    n_bad = bad_mask.sum()
    print(f"Found {n_bad} galaxies with Vf <= 0 or NaN")
    
    if n_bad == 0:
        print("All Vf values are valid!")
        return
    
    # Estimate Vf from curves
    print("\nEstimating Vf from outer rotation curves...")
    n_fixed = 0
    for idx in meta[bad_mask].index:
        name = meta.loc[idx, 'name']
        curve_file = curves_dir / f"{name}.csv"
        
        if not curve_file.exists():
            print(f"  {name:12s}: No curve file found")
            continue
        
        vf_est = estimate_vf_from_curve(curve_file)
        
        if np.isfinite(vf_est) and vf_est > 0:
            old_vf = meta.loc[idx, 'Vf']
            meta.loc[idx, 'Vf'] = vf_est
            print(f"  {name:12s}: Vf {old_vf:.1f} -> {vf_est:.1f} km/s (estimated)")
            n_fixed += 1
        else:
            print(f"  {name:12s}: Could not estimate Vf")
    
    # Save updated metadata
    meta.to_csv(meta_file, index=False)
    print(f"\nFixed {n_fixed}/{n_bad} galaxies")
    print(f"Saved updated metadata to {meta_file}")
    
    # Print new statistics
    print("\nUpdated Vf statistics:")
    print(meta['Vf'].describe())
    print(f"\nRemaining problematic Vf: {((meta['Vf'].isna()) | (meta['Vf'] <= 0.0)).sum()}")

if __name__ == '__main__':
    main()


