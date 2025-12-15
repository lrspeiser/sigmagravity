#!/usr/bin/env python3
"""Analyze SPARC pointwise residuals to find missing physics patterns.

Usage:
    python scripts/analyze_sparc_pointwise.py <sparc_pointwise.csv>
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def rankcorr(a, b):
    """Rank correlation (Spearman-like)."""
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main(in_csv: str):
    df = pd.read_csv(in_csv)
    
    # 1) How much of the dataset requires Σ<1?
    p_lt1_all = (df["Sigma_req"] < 1.0).mean()
    
    # Local bulge dominance bins
    df["bulge_bin"] = pd.cut(
        df["f_bulge_r"], 
        bins=[0, 0.1, 0.3, 0.6, 1.0], 
        include_lowest=True,
        labels=["f_bulge<0.1", "0.1≤f_bulge<0.3", "0.3≤f_bulge<0.6", "f_bulge≥0.6"]
    )
    
    summary = df.groupby("bulge_bin").agg({
        "Sigma_req": ["size", "mean", "median"],
        "need_sigma_lt_1": "mean",
        "Sigma_pred": "median",
        "dSigma": "median",
    })
    
    print("\n" + "="*70)
    print("SPARC POINTWISE RESIDUAL ANALYSIS")
    print("="*70)
    
    print(f"\n=== GLOBAL STATISTICS ===")
    print(f"N = {len(df):,} points")
    print(f"P(Σ_req < 1) = {p_lt1_all:.3f} ({p_lt1_all*100:.1f}%)")
    print(f"Median Σ_req = {df['Sigma_req'].median():.3f}")
    print(f"Median Σ_pred = {df['Sigma_pred'].median():.3f}")
    print(f"Median ΔΣ = {df['dSigma'].median():.3f}")
    
    print(f"\n=== BY LOCAL BULGE FRACTION f_bulge_r ===")
    print(f"{'Bin':<20} {'N':>8} {'P(Σ<1)':>10} {'Σ_req_med':>12} {'Σ_pred_med':>12} {'ΔΣ_med':>12}")
    print("-" * 70)
    for bin_name in summary.index:
        n = summary.loc[bin_name, ("Sigma_req", "size")]
        p_lt1 = summary.loc[bin_name, ("need_sigma_lt_1", "mean")]
        sigma_req_med = summary.loc[bin_name, ("Sigma_req", "median")]
        sigma_pred_med = summary.loc[bin_name, ("Sigma_pred", "median")]
        dsigma_med = summary.loc[bin_name, ("dSigma", "median")]
        print(f"{str(bin_name):<20} {n:>8,} {p_lt1:>10.3f} {sigma_req_med:>12.3f} {sigma_pred_med:>12.3f} {dsigma_med:>12.3f}")
    
    # 2) Top bulge galaxies by P(Σ_req<1)
    print(f"\n=== TOP 10 BULGE GALAXIES BY P(Σ_req<1) ===")
    galaxy_stats = df.groupby("galaxy").agg({
        "need_sigma_lt_1": ["mean", "size"],
        "f_bulge_global": "first",
        "Sigma_req": "median",
    })
    galaxy_stats.columns = ["p_sigma_lt1", "n_points", "f_bulge_global", "sigma_req_med"]
    galaxy_stats = galaxy_stats[galaxy_stats["f_bulge_global"] > 0.3].sort_values("p_sigma_lt1", ascending=False)
    
    print(f"{'Galaxy':<20} {'f_bulge':>10} {'N':>6} {'P(Σ<1)':>10} {'Σ_req_med':>12}")
    print("-" * 70)
    for galaxy, row in galaxy_stats.head(10).iterrows():
        print(f"{galaxy:<20} {row['f_bulge_global']:>10.3f} {int(row['n_points']):>6} {row['p_sigma_lt1']:>10.3f} {row['sigma_req_med']:>12.3f}")
    
    # 3) Rank correlations with dSigma
    print(f"\n=== RANK CORRELATIONS WITH ΔΣ ===")
    features = [
        "f_bulge_r", "f_gas_r", "R_over_Rd", 
        "g_bar_SI", "Omega_bar_SI", "tau_dyn_Myr",
        "dlnVbar_dlnR", "dlnGbar_dlnR"
    ]
    
    print(f"{'Feature':<20} {'Correlation':>12}")
    print("-" * 35)
    for f in features:
        if f in df.columns:
            c = rankcorr(df[f], df["dSigma"])
            print(f"{f:<20} {c:>12.3f}")
    
    # 4) Critical question: high-bulge regions
    high_bulge = df[df["f_bulge_r"] > 0.6]
    if len(high_bulge) > 0:
        p_lt1_high_bulge = (high_bulge["Sigma_req"] < 1.0).mean()
        print(f"\n=== HIGH-BULGE REGIONS (f_bulge_r > 0.6) ===")
        print(f"N = {len(high_bulge):,} points")
        print(f"P(Σ_req < 1) = {p_lt1_high_bulge:.3f} ({p_lt1_high_bulge*100:.1f}%)")
        print(f"Median Σ_req = {high_bulge['Sigma_req'].median():.3f}")
        print(f"Median Σ_pred = {high_bulge['Sigma_pred'].median():.3f}")
        print(f"Median ΔΣ = {high_bulge['dSigma'].median():.3f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)
    if p_lt1_high_bulge > 0.3:
        print(f"⚠️  HIGH overshoot in bulge regions ({p_lt1_high_bulge*100:.1f}% require Σ<1)")
        print("   → Current model CANNOT fix these (Σ ≥ 1 always)")
        print("   → Need either: (i) Σ < 1 capability, or (ii) baryon corrections")
    elif p_lt1_high_bulge > 0.1:
        print(f"⚠️  MODERATE overshoot in bulge regions ({p_lt1_high_bulge*100:.1f}% require Σ<1)")
        print("   → Some points need Σ<1, but not dominant")
    else:
        print(f"✓ LOW overshoot in bulge regions ({p_lt1_high_bulge*100:.1f}% require Σ<1)")
        print("   → Problem is likely elsewhere (coherence proxy, σ model, etc.)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_sparc_pointwise.py <sparc_pointwise.csv>")
        sys.exit(1)
    
    main(sys.argv[1])


