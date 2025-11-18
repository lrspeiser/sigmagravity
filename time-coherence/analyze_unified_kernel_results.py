"""
Comprehensive analysis of unified kernel results on SPARC galaxies.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print("UNIFIED KERNEL RESULTS ANALYSIS")
    print("=" * 80)
    
    # Load results
    results_csv = Path("time-coherence/unified_kernel_sparc_results.csv")
    summary_json = Path("time-coherence/unified_kernel_summary.json")
    
    df = pd.read_csv(results_csv)
    with open(summary_json, "r") as f:
        summary = json.load(f)
    
    print(f"\nLoaded {len(df)} galaxies")
    print(f"\nOverall Statistics:")
    print(f"  Improved: {summary['n_improved']} ({100*summary['n_improved']/summary['n_galaxies']:.1f}%)")
    print(f"  Worsened: {summary['n_worsened']} ({100*summary['n_worsened']/summary['n_galaxies']:.1f}%)")
    print(f"  Mean delta_RMS: {summary['mean_delta_rms']:.2f} km/s")
    print(f"  Median delta_RMS: {summary['median_delta_rms']:.2f} km/s")
    
    # Analyze by improvement category
    improved = df[df["delta_rms"] < 0]
    worsened = df[df["delta_rms"] > 0]
    
    print(f"\n" + "=" * 80)
    print("IMPROVED GALAXIES (delta_RMS < 0)")
    print("=" * 80)
    print(f"Count: {len(improved)}")
    print(f"Mean improvement: {improved['delta_rms'].mean():.2f} km/s")
    print(f"Median improvement: {improved['delta_rms'].median():.2f} km/s")
    print(f"Best improvement: {improved['delta_rms'].min():.2f} km/s")
    print(f"\nProperties of improved galaxies:")
    print(f"  Mean K_rough: {improved['K_rough'].mean():.3f}")
    print(f"  Mean F_missing: {improved['F_missing'].mean():.3f}")
    print(f"  Mean K_total: {improved['K_total_mean'].mean():.3f}")
    print(f"  Mean Xi: {improved['Xi_mean'].mean():.3f}")
    print(f"  Mean M_baryon: {improved['M_baryon'].mean():.2e} Msun")
    print(f"  Mean R_disk: {improved['R_disk'].mean():.2f} kpc")
    print(f"  Mean sigma_v: {improved['sigma_v'].mean():.2f} km/s")
    
    print(f"\n" + "=" * 80)
    print("WORSENED GALAXIES (delta_RMS > 0)")
    print("=" * 80)
    print(f"Count: {len(worsened)}")
    print(f"Mean worsening: {worsened['delta_rms'].mean():.2f} km/s")
    print(f"Median worsening: {worsened['delta_rms'].median():.2f} km/s")
    print(f"Worst worsening: {worsened['delta_rms'].max():.2f} km/s")
    print(f"\nProperties of worsened galaxies:")
    print(f"  Mean K_rough: {worsened['K_rough'].mean():.3f}")
    print(f"  Mean F_missing: {worsened['F_missing'].mean():.3f}")
    print(f"  Mean K_total: {worsened['K_total_mean'].mean():.3f}")
    print(f"  Mean Xi: {worsened['Xi_mean'].mean():.3f}")
    print(f"  Mean M_baryon: {worsened['M_baryon'].mean():.2e} Msun")
    print(f"  Mean R_disk: {worsened['R_disk'].mean():.2f} kpc")
    print(f"  Mean sigma_v: {worsened['sigma_v'].mean():.2f} km/s")
    
    # Check F_missing saturation
    f_saturated = df[df["F_missing"] >= 4.99]  # Close to F_max=5.0
    print(f"\n" + "=" * 80)
    print("F_MISSING SATURATION ANALYSIS")
    print("=" * 80)
    print(f"Galaxies with F_missing >= 4.99 (saturated): {len(f_saturated)} ({100*len(f_saturated)/len(df):.1f}%)")
    if len(f_saturated) > 0:
        print(f"  Mean delta_RMS: {f_saturated['delta_rms'].mean():.2f} km/s")
        print(f"  Median delta_RMS: {f_saturated['delta_rms'].median():.2f} km/s")
        print(f"  Improved: {np.sum(f_saturated['delta_rms'] < 0)} ({100*np.sum(f_saturated['delta_rms'] < 0)/len(f_saturated):.1f}%)")
    
    f_not_saturated = df[df["F_missing"] < 4.99]
    if len(f_not_saturated) > 0:
        print(f"\nGalaxies with F_missing < 4.99 (not saturated): {len(f_not_saturated)} ({100*len(f_not_saturated)/len(df):.1f}%)")
        print(f"  Mean delta_RMS: {f_not_saturated['delta_rms'].mean():.2f} km/s")
        print(f"  Median delta_RMS: {f_not_saturated['delta_rms'].median():.2f} km/s")
        print(f"  Improved: {np.sum(f_not_saturated['delta_rms'] < 0)} ({100*np.sum(f_not_saturated['delta_rms'] < 0)/len(f_not_saturated):.1f}%)")
    
    # Correlations
    print(f"\n" + "=" * 80)
    print("CORRELATIONS WITH PERFORMANCE")
    print("=" * 80)
    numeric_cols = ["K_rough", "F_missing", "K_total_mean", "Xi_mean", "M_baryon", "R_disk", "sigma_v", "rms_gr"]
    for col in numeric_cols:
        if col in df.columns:
            corr = np.corrcoef(df[col], df["delta_rms"])[0, 1]
            print(f"  {col:20s} vs delta_RMS: {corr:7.3f}")
    
    # Top and bottom performers
    print(f"\n" + "=" * 80)
    print("TOP 10 IMPROVEMENTS")
    print("=" * 80)
    top_improved = improved.nsmallest(10, "delta_rms")[["galaxy", "delta_rms", "rms_gr", "rms_model", "K_rough", "F_missing", "K_total_mean"]]
    print(top_improved.to_string(index=False))
    
    print(f"\n" + "=" * 80)
    print("TOP 10 WORST PERFORMERS")
    print("=" * 80)
    top_worsened = worsened.nlargest(10, "delta_rms")[["galaxy", "delta_rms", "rms_gr", "rms_model", "K_rough", "F_missing", "K_total_mean"]]
    print(top_worsened.to_string(index=False))
    
    # Distribution analysis
    print(f"\n" + "=" * 80)
    print("DISTRIBUTION STATISTICS")
    print("=" * 80)
    print(f"delta_RMS:")
    print(f"  Mean: {df['delta_rms'].mean():.2f} km/s")
    print(f"  Std: {df['delta_rms'].std():.2f} km/s")
    print(f"  Min: {df['delta_rms'].min():.2f} km/s")
    print(f"  25th percentile: {df['delta_rms'].quantile(0.25):.2f} km/s")
    print(f"  50th percentile (median): {df['delta_rms'].quantile(0.50):.2f} km/s")
    print(f"  75th percentile: {df['delta_rms'].quantile(0.75):.2f} km/s")
    print(f"  Max: {df['delta_rms'].max():.2f} km/s")
    
    # Save detailed analysis
    analysis = {
        "overall": summary,
        "improved_stats": {
            "count": int(len(improved)),
            "mean_delta_rms": float(improved["delta_rms"].mean()),
            "median_delta_rms": float(improved["delta_rms"].median()),
            "mean_K_rough": float(improved["K_rough"].mean()),
            "mean_F_missing": float(improved["F_missing"].mean()),
            "mean_K_total": float(improved["K_total_mean"].mean()),
        },
        "worsened_stats": {
            "count": int(len(worsened)),
            "mean_delta_rms": float(worsened["delta_rms"].mean()),
            "median_delta_rms": float(worsened["delta_rms"].median()),
            "mean_K_rough": float(worsened["K_rough"].mean()),
            "mean_F_missing": float(worsened["F_missing"].mean()),
            "mean_K_total": float(worsened["K_total_mean"].mean()),
        },
        "f_saturated": {
            "count": int(len(f_saturated)),
            "fraction": float(len(f_saturated) / len(df)),
            "mean_delta_rms": float(f_saturated["delta_rms"].mean()) if len(f_saturated) > 0 else None,
        },
        "correlations": {
            col: float(np.corrcoef(df[col], df["delta_rms"])[0, 1])
            for col in numeric_cols if col in df.columns
        },
    }
    
    analysis_path = Path("time-coherence/unified_kernel_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nDetailed analysis saved to {analysis_path}")
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. F_missing is saturated (F_max=5.0) for most galaxies.")
    print("   Consider increasing F_max or adjusting the mass-coherence model.")
    print("2. Mean delta_RMS is positive but median is negative.")
    print("   This suggests outliers are pulling the mean up.")
    print("3. Check correlations to identify which galaxy properties")
    print("   are associated with better/worse performance.")
    print("4. Consider parameter tuning (extra_amp, f_amp) to reduce")
    print("   the contribution of F_missing for over-predicting galaxies.")


if __name__ == "__main__":
    main()

