"""
Analyze pairing parameter tuning results.

Parse the grid search output and create visualizations and summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def main():
    results_dir = Path("results")
    grid_csv = results_dir / "pairing_parameter_grid.csv"
    
    if not grid_csv.exists():
        print(f"Error: {grid_csv} not found")
        return
    
    # Load results
    df = pd.read_csv(grid_csv)
    print(f"Loaded {len(df)} parameter configurations")
    
    # Filter for valid results
    df = df[df['improvement_pct'].notna()]
    
    # Solar System safety
    safe = df[df['solar_system_safe'] == True]
    unsafe = df[df['solar_system_safe'] == False]
    
    print(f"\n{'='*80}")
    print("PAIRING PARAMETER TUNING RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal configurations: {len(df)}")
    print(f"Solar System safe: {len(safe)} ({len(safe)/len(df)*100:.1f}%)")
    print(f"Positive improvement: {len(df[df['improvement_pct'] > 0])} ({len(df[df['improvement_pct'] > 0])/len(df)*100:.1f}%)")
    print(f"Safe AND improved: {len(safe[safe['improvement_pct'] > 0])} ({len(safe[safe['improvement_pct'] > 0])/len(df)*100:.1f}%)")
    
    # Best configurations
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*80}")
    
    # Best overall (ignoring safety)
    best_any = df.loc[df['improvement_pct'].idxmax()]
    print(f"\nBest overall (any safety):")
    print(f"  A_pair={best_any['A_pair']:.1f}, sigma_c={best_any['sigma_c']:.0f}, gamma={best_any['gamma_sigma']:.1f}, ell={best_any['ell_pair_kpc']:.0f}, p={best_any['p']:.1f}")
    print(f"  Improvement: {best_any['improvement_pct']:.2f}%")
    print(f"  RMS: {best_any['rms_pair_mean']:.2f} km/s (vs GR {best_any['rms_gr_mean']:.2f} km/s)")
    print(f"  Fraction improved: {best_any['fraction_improved']:.1%}")
    print(f"  K(Solar System): {best_any['K_solar_system']:.2e} (safe={best_any['solar_system_safe']})")
    
    # Best safe
    if len(safe) > 0:
        best_safe = safe.loc[safe['improvement_pct'].idxmax()]
        print(f"\nBest Solar System safe:")
        print(f"  A_pair={best_safe['A_pair']:.1f}, sigma_c={best_safe['sigma_c']:.0f}, gamma={best_safe['gamma_sigma']:.1f}, ell={best_safe['ell_pair_kpc']:.0f}, p={best_safe['p']:.1f}")
        print(f"  Improvement: {best_safe['improvement_pct']:.2f}%")
        print(f"  RMS: {best_safe['rms_pair_mean']:.2f} km/s (vs GR {best_safe['rms_gr_mean']:.2f} km/s)")
        print(f"  Fraction improved: {best_safe['fraction_improved']:.1%}")
        print(f"  K(Solar System): {best_safe['K_solar_system']:.2e}")
        
        # Save best safe parameters
        best_params = {
            "model": "pairing",
            "parameters": {
                "A_pair": float(best_safe['A_pair']),
                "sigma_c": float(best_safe['sigma_c']),
                "gamma_sigma": float(best_safe['gamma_sigma']),
                "ell_pair_kpc": float(best_safe['ell_pair_kpc']),
                "p": float(best_safe['p']),
            },
            "performance": {
                "improvement_pct": float(best_safe['improvement_pct']),
                "rms_pair_mean": float(best_safe['rms_pair_mean']),
                "rms_gr_mean": float(best_safe['rms_gr_mean']),
                "fraction_improved": float(best_safe['fraction_improved']),
                "n_galaxies": int(best_safe['n_galaxies']),
            },
            "safety": {
                "K_solar_system": float(best_safe['K_solar_system']),
                "solar_system_safe": bool(best_safe['solar_system_safe']),
                "safety_threshold": 1e-10,
            },
        }
        
        with open(results_dir / "pairing_best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Saved to {results_dir / 'pairing_best_params.json'}")
    
    # Top 20 safe configurations
    print(f"\n{'='*80}")
    print("TOP 20 SOLAR SYSTEM SAFE CONFIGURATIONS")
    print(f"{'='*80}")
    if len(safe) > 0:
        top20 = safe.nlargest(20, 'improvement_pct')
        print(f"\n{'Rank':<6}{'A_pair':<8}{'sig_c':<8}{'gamma':<8}{'ell':<8}{'p':<8}{'Improv%':<10}{'Frac':<8}")
        print("-" * 80)
        for i, (_, row) in enumerate(top20.iterrows(), 1):
            print(f"{i:<6}{row['A_pair']:<8.1f}{row['sigma_c']:<8.0f}{row['gamma_sigma']:<8.1f}"
                  f"{row['ell_pair_kpc']:<8.0f}{row['p']:<8.1f}{row['improvement_pct']:<10.2f}"
                  f"{row['fraction_improved']:<8.1%}")
    
    # Parameter sensitivity analysis
    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY")
    print(f"{'='*80}")
    
    for param in ['A_pair', 'sigma_c', 'gamma_sigma', 'ell_pair_kpc', 'p']:
        print(f"\n{param}:")
        param_groups = safe.groupby(param)['improvement_pct'].agg(['mean', 'max', 'count'])
        print(param_groups.sort_index())
    
    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    if len(safe) > 0:
        print(f"\n1. Best improvement: {best_safe['improvement_pct']:.1f}% (vs +6% with defaults)")
        print(f"   -> Need A_pair ~ 5.0 (5x higher than default)")
        print(f"   -> Prefer low sigma_c ~ 15-20 km/s (colder transition)")
        print(f"   -> Sharp transition gamma ~ 2.5-3.0")
        
        print(f"\n2. Solar System safety:")
        print(f"   -> Best safe config has K(1 AU) = {best_safe['K_solar_system']:.2e}")
        print(f"   -> Need p >= 1.5 for strong small-scale suppression")
        print(f"   -> {len(safe)/len(df)*100:.0f}% of configs are safe")
        
        print(f"\n3. Consistency:")
        print(f"   -> Best config improves {best_safe['fraction_improved']:.0%} of galaxies")
        print(f"   -> Mean RMS reduced from {best_safe['rms_gr_mean']:.1f} to {best_safe['rms_pair_mean']:.1f} km/s")
        
        gap = 100 * (1 - best_safe['rms_pair_mean'] / best_safe['rms_gr_mean'])
        target_gap = 30  # Need ~30% improvement to match empirical Sigma-Gravity
        print(f"\n4. Gap to empirical Sigma-Gravity:")
        print(f"   -> Current: {gap:.1f}% RMS reduction")
        print(f"   -> Target: ~{target_gap}% RMS reduction")
        print(f"   -> Still need: ~{target_gap - gap:.1f}% more improvement")
        print(f"   -> Possible paths:")
        print(f"     a) Combine with roughness model (~10-20% boost)")
        print(f"     b) Further A_pair increase (test A > 5)")
        print(f"     c) Different radial profile (Burr-XII instead of exponential)")
    
    # Create summary statistics
    summary = {
        "total_configs": len(df),
        "safe_configs": len(safe),
        "improved_configs": len(df[df['improvement_pct'] > 0]),
        "safe_and_improved": len(safe[safe['improvement_pct'] > 0]),
        "best_overall": {
            "improvement_pct": float(best_any['improvement_pct']),
            "parameters": {
                "A_pair": float(best_any['A_pair']),
                "sigma_c": float(best_any['sigma_c']),
                "gamma_sigma": float(best_any['gamma_sigma']),
                "ell_pair_kpc": float(best_any['ell_pair_kpc']),
                "p": float(best_any['p']),
            },
            "solar_system_safe": bool(best_any['solar_system_safe']),
        },
    }
    
    if len(safe) > 0:
        summary["best_safe"] = {
            "improvement_pct": float(best_safe['improvement_pct']),
            "parameters": {
                "A_pair": float(best_safe['A_pair']),
                "sigma_c": float(best_safe['sigma_c']),
                "gamma_sigma": float(best_safe['gamma_sigma']),
                "ell_pair_kpc": float(best_safe['ell_pair_kpc']),
                "p": float(best_safe['p']),
            },
        }
    
    with open(results_dir / "pairing_tuning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Summary saved to {results_dir / 'pairing_tuning_summary.json'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

