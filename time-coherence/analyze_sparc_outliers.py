"""
Analyze SPARC outliers with morphology and quality flags.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    sparc_cohs = Path("time-coherence/sparc_coherence_test.csv")
    sparc_meta = Path("data/sparc/sparc_combined.csv")
    out_csv = Path("time-coherence/sparc_outlier_morphology.csv")
    
    if not sparc_cohs.exists():
        print(f"Error: {sparc_cohs} not found. Run test_sparc_coherence.py first.")
        return
    
    if not sparc_meta.exists():
        print(f"Error: {sparc_meta} not found.")
        return
    
    # Load data
    coh = pd.read_csv(sparc_cohs)
    meta = pd.read_csv(sparc_meta)
    
    # Merge on galaxy name
    # Try different column name variations
    galaxy_col_coh = None
    galaxy_col_meta = None
    
    for col in coh.columns:
        if "galaxy" in col.lower():
            galaxy_col_coh = col
            break
    
    for col in meta.columns:
        if "galaxy" in col.lower() or "name" in col.lower():
            galaxy_col_meta = col
            break
    
    if galaxy_col_coh and galaxy_col_meta:
        df = coh.merge(meta, left_on=galaxy_col_coh, right_on=galaxy_col_meta, how="left")
    else:
        print("Warning: Could not find galaxy name columns, using coh data only")
        df = coh.copy()
    
    # Rank by delta_rms
    if "delta_rms" not in df.columns and "rms_coherence" in df.columns and "rms_gr" in df.columns:
        df["delta_rms"] = df["rms_coherence"] - df["rms_gr"]
    
    worst = df.sort_values("delta_rms", ascending=False).head(30)
    
    # Select useful columns
    cols_to_keep = ["galaxy"] if "galaxy" in worst.columns else []
    if "delta_rms" in worst.columns:
        cols_to_keep.append("delta_rms")
    if "sigma_v_kms" in worst.columns:
        cols_to_keep.append("sigma_v_kms")
    elif "sigma_velocity" in worst.columns:
        cols_to_keep.append("sigma_velocity")
    
    # Add any morphology columns that exist
    morphology_cols = ["bar_flag", "warp_flag", "inclination", "bulge_frac", 
                       "morphology_code", "Q_flag", "HI_asymmetry"]
    for col in morphology_cols:
        if col in worst.columns:
            cols_to_keep.append(col)
    
    # Keep only columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in worst.columns]
    
    worst_filtered = worst[cols_to_keep].copy()
    worst_filtered.to_csv(out_csv, index=False)
    
    print("=" * 80)
    print("SPARC OUTLIER MORPHOLOGY ANALYSIS")
    print("=" * 80)
    print(f"\nWorst 30 galaxies by delta_RMS:")
    print(f"  Mean delta_RMS: {worst['delta_rms'].mean():.2f} km/s")
    print(f"  Range: [{worst['delta_rms'].min():.2f}, {worst['delta_rms'].max():.2f}] km/s")
    
    if "sigma_v_kms" in worst.columns or "sigma_velocity" in worst.columns:
        sigma_col = "sigma_v_kms" if "sigma_v_kms" in worst.columns else "sigma_velocity"
        print(f"\n  Mean sigma_v: {worst[sigma_col].mean():.2f} km/s")
        print(f"  Overall mean sigma_v: {df[sigma_col].mean():.2f} km/s")
        print(f"  Ratio: {worst[sigma_col].mean() / df[sigma_col].mean():.2f}x")
    
    # Check morphology flags if available
    if "bar_flag" in worst.columns:
        n_bars = worst["bar_flag"].sum()
        print(f"\n  Galaxies with bars: {n_bars}/{len(worst)} ({n_bars/len(worst)*100:.1f}%)")
    
    if "warp_flag" in worst.columns:
        n_warps = worst["warp_flag"].sum()
        print(f"  Galaxies with warps: {n_warps}/{len(worst)} ({n_warps/len(worst)*100:.1f}%)")
    
    if "bulge_frac" in worst.columns:
        mean_bulge = worst["bulge_frac"].mean()
        print(f"  Mean bulge fraction: {mean_bulge:.3f}")
    
    print(f"\nSaved to {out_csv}")

if __name__ == "__main__":
    main()

