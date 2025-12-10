"""
Analyze the results of the unified kernel parameter sweep.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent.parent
sweep_results = project_root / "time-coherence" / "unified_kernel_sweep_results.csv"

# Load sweep results
df = pd.read_csv(sweep_results)

print("=" * 80)
print("UNIFIED KERNEL PARAMETER SWEEP ANALYSIS")
print("=" * 80)
print()

# Overall statistics
print("OVERALL STATISTICS")
print("-" * 80)
print(f"Total parameter combinations tested: {len(df)}")
print(f"Galaxies per combination: {df['n_galaxies'].iloc[0]}")
print()

# Best results by different metrics
print("TOP 10 PARAMETER SETS")
print("-" * 80)

print("\n1. By Mean Delta RMS (most negative = best):")
print("-" * 40)
top_mean = df.nsmallest(10, 'mean_delta_rms')
for idx, row in top_mean.iterrows():
    print(f"  gamma_sigma={row['gamma_sigma']:.2f}, F_max={row['F_max']:.1f}, "
          f"extra_amp={row['extra_amp']:.3f}, sigma_gate_ref={row['sigma_gate_ref']:.1f}")
    print(f"    mean_delta={row['mean_delta_rms']:.2f} km/s, "
          f"median_delta={row['median_delta_rms']:.2f} km/s, "
          f"frac_improved={row['frac_improved']:.3f}")

print("\n2. By Median Delta RMS (most negative = best):")
print("-" * 40)
top_median = df.nsmallest(10, 'median_delta_rms')
for idx, row in top_median.iterrows():
    print(f"  gamma_sigma={row['gamma_sigma']:.2f}, F_max={row['F_max']:.1f}, "
          f"extra_amp={row['extra_amp']:.3f}, sigma_gate_ref={row['sigma_gate_ref']:.1f}")
    print(f"    mean_delta={row['mean_delta_rms']:.2f} km/s, "
          f"median_delta={row['median_delta_rms']:.2f} km/s, "
          f"frac_improved={row['frac_improved']:.3f}")

print("\n3. By Fraction Improved (highest = best):")
print("-" * 40)
top_frac = df.nlargest(10, 'frac_improved')
for idx, row in top_frac.iterrows():
    print(f"  gamma_sigma={row['gamma_sigma']:.2f}, F_max={row['F_max']:.1f}, "
          f"extra_amp={row['extra_amp']:.3f}, sigma_gate_ref={row['sigma_gate_ref']:.1f}")
    print(f"    mean_delta={row['mean_delta_rms']:.2f} km/s, "
          f"median_delta={row['median_delta_rms']:.2f} km/s, "
          f"frac_improved={row['frac_improved']:.3f}")

# Parameter sensitivity analysis
print("\n\nPARAMETER SENSITIVITY ANALYSIS")
print("-" * 80)

# Group by each parameter and show statistics
print("\n1. Effect of gamma_sigma (sigma-gating strength):")
for gamma in sorted(df['gamma_sigma'].unique()):
    subset = df[df['gamma_sigma'] == gamma]
    print(f"  gamma_sigma={gamma:.2f}:")
    print(f"    Mean delta_RMS: {subset['mean_delta_rms'].mean():.3f} +/- {subset['mean_delta_rms'].std():.3f} km/s")
    print(f"    Median delta_RMS: {subset['median_delta_rms'].mean():.3f} +/- {subset['median_delta_rms'].std():.3f} km/s")
    print(f"    Fraction improved: {subset['frac_improved'].mean():.3f} +/- {subset['frac_improved'].std():.3f}")

print("\n2. Effect of F_max (maximum F_missing clamp):")
for fmax in sorted(df['F_max'].unique()):
    subset = df[df['F_max'] == fmax]
    print(f"  F_max={fmax:.1f}:")
    print(f"    Mean delta_RMS: {subset['mean_delta_rms'].mean():.3f} +/- {subset['mean_delta_rms'].std():.3f} km/s")
    print(f"    Median delta_RMS: {subset['median_delta_rms'].mean():.3f} +/- {subset['median_delta_rms'].std():.3f} km/s")
    print(f"    Fraction improved: {subset['frac_improved'].mean():.3f} +/- {subset['frac_improved'].std():.3f}")

print("\n3. Effect of extra_amp (F_missing lever arm):")
for amp in sorted(df['extra_amp'].unique()):
    subset = df[df['extra_amp'] == amp]
    print(f"  extra_amp={amp:.3f}:")
    print(f"    Mean delta_RMS: {subset['mean_delta_rms'].mean():.3f} +/- {subset['mean_delta_rms'].std():.3f} km/s")
    print(f"    Median delta_RMS: {subset['median_delta_rms'].mean():.3f} +/- {subset['median_delta_rms'].std():.3f} km/s")
    print(f"    Fraction improved: {subset['frac_improved'].mean():.3f} +/- {subset['frac_improved'].std():.3f}")

print("\n4. Effect of sigma_gate_ref (gating reference velocity):")
for ref in sorted(df['sigma_gate_ref'].unique()):
    subset = df[df['sigma_gate_ref'] == ref]
    print(f"  sigma_gate_ref={ref:.1f}:")
    print(f"    Mean delta_RMS: {subset['mean_delta_rms'].mean():.3f} +/- {subset['mean_delta_rms'].std():.3f} km/s")
    print(f"    Median delta_RMS: {subset['median_delta_rms'].mean():.3f} +/- {subset['median_delta_rms'].std():.3f} km/s")
    print(f"    Fraction improved: {subset['frac_improved'].mean():.3f} +/- {subset['frac_improved'].std():.3f}")

# Recommended parameter sets
print("\n\nRECOMMENDED PARAMETER SETS")
print("-" * 80)

# Best balanced: good mean delta, good fraction improved
balanced = df[(df['mean_delta_rms'] < -1.0) & (df['frac_improved'] > 0.73)]
if len(balanced) > 0:
    best_balanced = balanced.loc[balanced['mean_delta_rms'].idxmin()]
    print("\n1. Best Balanced (mean improvement + high fraction improved):")
    print(f"  gamma_sigma: {best_balanced['gamma_sigma']:.2f}")
    print(f"  F_max: {best_balanced['F_max']:.1f}")
    print(f"  extra_amp: {best_balanced['extra_amp']:.3f}")
    print(f"  sigma_gate_ref: {best_balanced['sigma_gate_ref']:.1f}")
    print(f"  Performance:")
    print(f"    Mean delta_RMS: {best_balanced['mean_delta_rms']:.2f} km/s")
    print(f"    Median delta_RMS: {best_balanced['median_delta_rms']:.2f} km/s")
    print(f"    Fraction improved: {best_balanced['frac_improved']:.3f}")
    
    # Save this as recommended
    rec_params = {
        "gamma_sigma": float(best_balanced['gamma_sigma']),
        "F_max": float(best_balanced['F_max']),
        "extra_amp": float(best_balanced['extra_amp']),
        "sigma_gate_ref": float(best_balanced['sigma_gate_ref']),
        "performance": {
            "mean_delta_rms": float(best_balanced['mean_delta_rms']),
            "median_delta_rms": float(best_balanced['median_delta_rms']),
            "frac_improved": float(best_balanced['frac_improved']),
        }
    }
    out_json = project_root / "time-coherence" / "recommended_unified_params.json"
    with open(out_json, 'w') as f:
        json.dump(rec_params, f, indent=2)
    print(f"\n  Saved to: {out_json}")

# Best for dwarfs (median performance)
print("\n2. Best for Dwarf Galaxies (median delta RMS):")
best_dwarf = df.loc[df['median_delta_rms'].idxmin()]
print(f"  gamma_sigma: {best_dwarf['gamma_sigma']:.2f}")
print(f"  F_max: {best_dwarf['F_max']:.1f}")
print(f"  extra_amp: {best_dwarf['extra_amp']:.3f}")
print(f"  sigma_gate_ref: {best_dwarf['sigma_gate_ref']:.1f}")
print(f"  Performance:")
print(f"    Mean delta_RMS: {best_dwarf['mean_delta_rms']:.2f} km/s")
print(f"    Median delta_RMS: {best_dwarf['median_delta_rms']:.2f} km/s")
print(f"    Fraction improved: {best_dwarf['frac_improved']:.3f}")

# Sigma bin analysis (if available)
sigma_cols = [col for col in df.columns if col.startswith('mean_delta_')]
if len(sigma_cols) > 1:  # More than just mean_delta_rms
    print("\n\nPERFORMANCE BY SIGMA_V BIN")
    print("-" * 80)
    
    best_overall = df.loc[df['mean_delta_rms'].idxmin()]
    print("\nFor best overall parameter set:")
    print(f"  (gamma_sigma={best_overall['gamma_sigma']:.2f}, F_max={best_overall['F_max']:.1f}, "
          f"extra_amp={best_overall['extra_amp']:.3f})")
    for col in sigma_cols:
        if col != 'mean_delta_rms' and pd.notna(best_overall.get(col)):
            bin_name = col.replace('mean_delta_', '')
            n_col = f"n_{bin_name}"
            n_gal = best_overall.get(n_col, 0)
            print(f"  {bin_name:12s}: {best_overall[col]:7.2f} km/s  (n={n_gal:.0f})")

print("\n" + "=" * 80)
print("Analysis complete.")
print("=" * 80)

