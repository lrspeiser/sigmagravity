"""
Identify and analyze SPARC outliers with large positive ΔRMS.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_sparc_summary():
    """Load SPARC summary with sigma_v and other properties."""
    summary_path = Path("data/sparc/sparc_combined.csv")
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found")
        return None
    return pd.read_csv(summary_path)

def analyze_outliers(csv_path: str = "time-coherence/sparc_coherence_test.csv", 
                     n_outliers: int = 20):
    """
    Identify worst-performing galaxies and analyze patterns.
    
    Parameters:
    -----------
    csv_path : str
        Path to SPARC coherence test results CSV
    n_outliers : int
        Number of worst galaxies to analyze
    """
    df = pd.read_csv(csv_path)
    summary = load_sparc_summary()
    
    # Sort by delta_rms descending (worst first)
    df_sorted = df.sort_values("delta_rms", ascending=False)
    worst = df_sorted.head(n_outliers).copy()
    
    print("=" * 80)
    print(f"SPARC OUTLIER ANALYSIS (worst {n_outliers} galaxies)")
    print("=" * 80)
    
    print(f"\nOverall statistics:")
    print(f"  Total galaxies: {len(df)}")
    print(f"  Mean delta_RMS: {df['delta_rms'].mean():.3f} km/s")
    print(f"  Median delta_RMS: {df['delta_rms'].median():.3f} km/s")
    print(f"  Improved: {(df['delta_rms'] < 0).sum()}/{len(df)} ({(df['delta_rms'] < 0).sum()/len(df)*100:.1f}%)")
    
    print(f"\nWorst {n_outliers} galaxies:")
    print(f"  Mean delta_RMS: {worst['delta_rms'].mean():.3f} km/s")
    print(f"  Median delta_RMS: {worst['delta_rms'].median():.3f} km/s")
    print(f"  Range: [{worst['delta_rms'].min():.2f}, {worst['delta_rms'].max():.2f}] km/s")
    
    # Merge with summary if available
    if summary is not None:
        # Try to match galaxy names - check what column name exists
        galaxy_col_summary = None
        for col in summary.columns:
            if col.lower() in ['galaxy', 'name', 'gal']:
                galaxy_col_summary = col
                break
        
        if galaxy_col_summary:
            worst_with_summary = worst.merge(
                summary, 
                left_on="galaxy", 
                right_on=galaxy_col_summary, 
                how="left",
                suffixes=("", "_summary")
            )
        else:
            worst_with_summary = worst
        
        # Analyze patterns
        print(f"\n--- Pattern Analysis ---")
        
        # Check sigma_v
        if "sigma_v" in worst.columns or "sigma_v_kms" in worst.columns:
            sigma_col = "sigma_v" if "sigma_v" in worst.columns else "sigma_v_kms"
            print(f"\nsigma_v distribution (worst {n_outliers}):")
            print(f"  Mean: {worst[sigma_col].mean():.2f} km/s")
            print(f"  Median: {worst[sigma_col].median():.2f} km/s")
            print(f"  Range: [{worst[sigma_col].min():.2f}, {worst[sigma_col].max():.2f}] km/s")
            
            # Compare to overall
            if sigma_col in df.columns:
                print(f"\n  Overall mean: {df[sigma_col].mean():.2f} km/s")
                print(f"  Ratio (worst/overall): {worst[sigma_col].mean() / df[sigma_col].mean():.2f}x")
        
        # Check R_max if available
        if "R_max_kpc" in worst.columns:
            print(f"\nR_max distribution (worst {n_outliers}):")
            print(f"  Mean: {worst['R_max_kpc'].mean():.2f} kpc")
            print(f"  Median: {worst['R_max_kpc'].median():.2f} kpc")
            print(f"  Range: [{worst['R_max_kpc'].min():.2f}, {worst['R_max_kpc'].max():.2f}] kpc")
        
        # Check ell_coh
        if "ell_coh_mean_kpc" in worst.columns:
            print(f"\nell_coh distribution (worst {n_outliers}):")
            print(f"  Mean: {worst['ell_coh_mean_kpc'].mean():.2f} kpc")
            print(f"  Median: {worst['ell_coh_mean_kpc'].median():.2f} kpc")
            print(f"  Range: [{worst['ell_coh_mean_kpc'].min():.2f}, {worst['ell_coh_mean_kpc'].max():.2f}] kpc")
        
        # Check K_max if available
        if "K_max" in worst.columns:
            print(f"\nK_max distribution (worst {n_outliers}):")
            print(f"  Mean: {worst['K_max'].mean():.3f}")
            print(f"  Median: {worst['K_max'].median():.3f}")
            print(f"  Range: [{worst['K_max'].min():.3f}, {worst['K_max'].max():.3f}]")
    
    # List worst galaxies
    print(f"\n--- Worst {n_outliers} galaxies (delta_RMS > +25 km/s) ---")
    worst_25 = worst[worst['delta_rms'] > 25.0]
    if len(worst_25) > 0:
        for idx, row in worst_25.iterrows():
            sigma_str = ""
            if "sigma_v" in row.index:
                sigma_str = f", sigma_v={row['sigma_v']:.1f}"
            elif "sigma_v_kms" in row.index:
                sigma_str = f", sigma_v={row['sigma_v_kms']:.1f}"
            
            rmax_str = ""
            if "R_max_kpc" in row.index:
                rmax_str = f", R_max={row['R_max_kpc']:.1f} kpc"
            
            galaxy_name = str(row['galaxy']).encode('ascii', 'replace').decode('ascii')
            print(f"  {galaxy_name}: delta_RMS={row['delta_rms']:.2f} km/s{sigma_str}{rmax_str}")
    else:
        print(f"  (No galaxies with delta_RMS > +25 km/s in worst {n_outliers})")
    
    # Save to CSV
    output_path = Path("time-coherence/sparc_outliers_analysis.csv")
    worst.to_csv(output_path, index=False)
    print(f"\nSaved worst {n_outliers} galaxies to {output_path}")
    
    return worst


if __name__ == "__main__":
    worst = analyze_outliers(n_outliers=20)
