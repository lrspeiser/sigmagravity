"""
Analyze F_missing = A_empirical / K_rough correlations.

This script correlates F_missing with system properties to identify
what physical mechanism might explain the "missing" enhancement.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_sparc_metadata(sparc_summary_csv: str) -> pd.DataFrame:
    """Load SPARC summary with metadata."""
    df = pd.read_csv(sparc_summary_csv)
    return df


def analyze_correlations(df: pd.DataFrame) -> dict:
    """
    Correlate F_missing with system properties.
    
    Properties to check:
    - sigma_v (velocity dispersion)
    - R_d (disc scale length)
    - gas fraction
    - morphology flags (bar, warp, bulge)
    - environment
    """
    results = {}
    
    # Filter valid F_missing
    valid = df["F_missing"].dropna()
    if len(valid) < 10:
        return {"error": "Insufficient data"}
    
    df_valid = df[df["F_missing"].notna()].copy()
    F_missing = df_valid["F_missing"].values
    
    # Properties to check
    properties = {
        "sigma_v": "sigma_velocity",
        "R_d": ["R_disk", "R_d", "disk_scale_length"],
        "gas_fraction": ["gas_fraction", "f_gas", "M_gas_M_star"],
        "bar_flag": ["bar_flag", "bar", "Bar"],
        "warp_flag": ["warp_flag", "warp", "Warp"],
        "bulge_frac": ["bulge_frac", "bulge_fraction", "B_T"],
    }
    
    for prop_name, col_names in properties.items():
        if isinstance(col_names, str):
            col_names = [col_names]
        
        # Find column
        col = None
        for cn in col_names:
            if cn in df_valid.columns:
                col = cn
                break
        
        if col is None:
            continue
        
        values = df_valid[col].values
        valid_mask = np.isfinite(values) & np.isfinite(F_missing)
        
        if np.sum(valid_mask) < 10:
            continue
        
        x = values[valid_mask]
        y = F_missing[valid_mask]
        
        # Pearson correlation
        r_pearson, p_pearson = pearsonr(x, y)
        
        # Spearman correlation (rank-based, more robust)
        r_spearman, p_spearman = spearmanr(x, y)
        
        results[prop_name] = {
            "column": col,
            "n_points": int(np.sum(valid_mask)),
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "mean_x": float(np.mean(x)),
            "mean_y": float(np.mean(y)),
        }
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze F_missing correlations"
    )
    parser.add_argument(
        "--roughness-csv",
        type=str,
        default="time-coherence/sparc_roughness_amplitude.csv",
        help="CSV from roughness amplitude test",
    )
    parser.add_argument(
        "--sparc-summary-csv",
        type=str,
        default="data/sparc/sparc_combined.csv",
        help="SPARC summary CSV with metadata",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="time-coherence/F_missing_correlations.json",
        help="Output JSON",
    )
    args = parser.parse_args()
    
    # Load data
    roughness_df = pd.read_csv(args.roughness_csv)
    summary_df = load_sparc_metadata(args.sparc_summary_csv)
    
    # Merge
    galaxy_col_rough = "galaxy"
    galaxy_col_summary = None
    for col in ["galaxy_name", "Galaxy", "name", "gal"]:
        if col in summary_df.columns:
            galaxy_col_summary = col
            break
    
    if galaxy_col_summary is None:
        print("Error: Could not find galaxy name column in summary")
        return
    
    # Merge on galaxy name
    merged = roughness_df.merge(
        summary_df,
        left_on=galaxy_col_rough,
        right_on=galaxy_col_summary,
        how="inner",
    )
    
    if len(merged) == 0:
        print("Error: No matching galaxies after merge")
        return
    
    print("=" * 80)
    print("F_MISSING CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"\nMerged {len(merged)} galaxies")
    print(f"Mean F_missing: {merged['F_missing'].mean():.3f}")
    print(f"Median F_missing: {merged['F_missing'].median():.3f}")
    
    # Analyze correlations
    results = analyze_correlations(merged)
    
    if "error" in results:
        print(f"\nError: {results['error']}")
        return
    
    # Save results
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("CORRELATION RESULTS")
    print("=" * 80)
    
    # Sort by absolute Spearman correlation
    sorted_results = sorted(
        results.items(),
        key=lambda x: abs(x[1]["spearman_r"]),
        reverse=True,
    )
    
    for prop_name, stats in sorted_results:
        print(f"\n{prop_name}:")
        print(f"  Column: {stats['column']}")
        print(f"  N points: {stats['n_points']}")
        print(f"  Pearson r: {stats['pearson_r']:.3f} (p={stats['pearson_p']:.3e})")
        print(f"  Spearman r: {stats['spearman_r']:.3f} (p={stats['spearman_p']:.3e})")
        print(f"  Mean {prop_name}: {stats['mean_x']:.3f}")
        print(f"  Mean F_missing: {stats['mean_y']:.3f}")
        
        if abs(stats['spearman_r']) > 0.3 and stats['spearman_p'] < 0.05:
            print(f"  *** SIGNIFICANT CORRELATION ***")
    
    print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()

