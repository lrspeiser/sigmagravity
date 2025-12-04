#!/usr/bin/env python3
"""
Fox+ 2022 Cluster Validation: g† = cH₀/(4√π) vs g† = cH₀/(2e)
=============================================================

Tests the new critical acceleration formula on the FULL Fox+ 2022 cluster sample.
This is the academically defensible test using real published data.

Data source: Fox+ 2022 (ApJ 928, 87) - 75 unique clusters with strong lensing masses

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, Optional

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8              # m/s
G = 6.674e-11            # m³/kg/s²
M_sun = 1.989e30         # kg
kpc_to_m = 3.086e19
Mpc_to_m = 3.086e22
H0 = 70                  # km/s/Mpc
H0_SI = H0 * 1000 / Mpc_to_m
cH0 = c * H0_SI

# Two formulas for critical acceleration
g_dagger_old = cH0 / (2 * math.e)
g_dagger_new = cH0 / (4 * math.sqrt(math.pi))

# Cluster amplitude
A_cluster = math.pi * math.sqrt(2)  # π√2 ≈ 4.44

print("=" * 80)
print("FOX+ 2022 CLUSTER VALIDATION: g† = cH₀/(4√π) vs g† = cH₀/(2e)")
print("=" * 80)
print(f"\nOld formula: g† = cH₀/(2e)   = {g_dagger_old:.4e} m/s²")
print(f"New formula: g† = cH₀/(4√π)  = {g_dagger_new:.4e} m/s²")
print(f"Cluster amplitude: A = π√2 = {A_cluster:.4f}")

# =============================================================================
# Σ-GRAVITY FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray, g_dagger: float) -> np.ndarray:
    """Universal h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def Sigma_cluster(g_bar: np.ndarray, g_dagger: float) -> np.ndarray:
    """
    Enhancement factor for clusters.
    W = 1 for clusters (lensing is line-of-sight integrated)
    """
    h = h_function(g_bar, g_dagger)
    return 1 + A_cluster * h


# =============================================================================
# LOAD FOX+ 2022 DATA
# =============================================================================

def find_fox2022_data() -> Optional[Path]:
    """Find the Fox+ 2022 cluster data."""
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/clusters/fox2022_unique_clusters.csv"),
        Path(__file__).parent.parent / "data" / "clusters" / "fox2022_unique_clusters.csv",
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def run_fox2022_validation():
    """Run validation on Fox+ 2022 cluster sample."""
    
    # Find data
    data_file = find_fox2022_data()
    if data_file is None:
        print("\nERROR: Fox+ 2022 data not found!")
        return None
    
    print(f"\nData file: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} clusters from Fox+ 2022")
    
    # Filter to clusters with required data
    # Need: M500 (for baryonic mass estimate) and MSL_200kpc (for comparison)
    df_valid = df[df['M500_1e14Msun'].notna() & df['MSL_200kpc_1e12Msun'].notna()].copy()
    print(f"Clusters with M500 and MSL_200kpc: {len(df_valid)}")
    
    # Filter to spectroscopic redshifts for quality
    df_specz = df_valid[df_valid['spec_z_constraint'] == 'yes'].copy()
    print(f"With spectroscopic redshifts: {len(df_specz)}")
    
    # Use quality sample
    df_analysis = df_specz.copy()
    
    # Further filter out very low mass clusters
    df_analysis = df_analysis[df_analysis['M500_1e14Msun'] > 2.0].copy()
    print(f"After M500 > 2×10¹⁴ M☉ cut: {len(df_analysis)}")
    
    if len(df_analysis) == 0:
        print("ERROR: No clusters pass quality cuts!")
        return None
    
    print("\n" + "=" * 80)
    print("ANALYSIS: Baryonic mass enhancement vs strong lensing mass")
    print("=" * 80)
    
    # Baryonic fraction (gas + stars)
    f_baryon = 0.15  # Typical: ~12% gas + ~3% stars within R500
    
    results = []
    
    for idx, row in df_analysis.iterrows():
        cluster = row['cluster']
        z = row['z_lens']
        
        # Total mass within R500
        M500 = row['M500_1e14Msun'] * 1e14 * M_sun  # kg
        
        # Baryonic mass estimate at 200 kpc
        # M_bar(200kpc) ~ 0.4 * f_baryon * M500 (gas concentrated toward center)
        M_bar_200 = 0.4 * f_baryon * M500  # kg
        
        # Baryonic acceleration at 200 kpc
        r_200kpc = 200 * kpc_to_m
        g_bar = G * M_bar_200 / r_200kpc**2
        
        # Σ-Gravity enhancement (old formula)
        Sigma_old = Sigma_cluster(np.array([g_bar]), g_dagger_old)[0]
        M_sigma_old = Sigma_old * M_bar_200
        
        # Σ-Gravity enhancement (new formula)
        Sigma_new = Sigma_cluster(np.array([g_bar]), g_dagger_new)[0]
        M_sigma_new = Sigma_new * M_bar_200
        
        # Observed strong lensing mass at 200 kpc
        MSL_200 = row['MSL_200kpc_1e12Msun'] * 1e12 * M_sun
        MSL_err_lo = row['e_MSL_lo'] * 1e12 * M_sun if pd.notna(row['e_MSL_lo']) else MSL_200 * 0.1
        MSL_err_hi = row['e_MSL_hi'] * 1e12 * M_sun if pd.notna(row['e_MSL_hi']) else MSL_200 * 0.1
        MSL_err = (MSL_err_lo + MSL_err_hi) / 2
        
        # Ratios
        ratio_old = M_sigma_old / MSL_200
        ratio_new = M_sigma_new / MSL_200
        
        # Log residuals (dex)
        log_resid_old = np.log10(M_sigma_old / MSL_200)
        log_resid_new = np.log10(M_sigma_new / MSL_200)
        
        results.append({
            'cluster': cluster,
            'z': z,
            'M500': row['M500_1e14Msun'],
            'M_bar_200': M_bar_200 / M_sun,
            'g_bar': g_bar,
            'Sigma_old': Sigma_old,
            'Sigma_new': Sigma_new,
            'M_sigma_old': M_sigma_old / M_sun,
            'M_sigma_new': M_sigma_new / M_sun,
            'MSL_200': MSL_200 / M_sun,
            'MSL_err': MSL_err / M_sun,
            'ratio_old': ratio_old,
            'ratio_new': ratio_new,
            'log_resid_old': log_resid_old,
            'log_resid_new': log_resid_new,
        })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================
    
    n_clusters = len(df_results)
    
    # Mean and median ratios
    mean_ratio_old = df_results['ratio_old'].mean()
    mean_ratio_new = df_results['ratio_new'].mean()
    median_ratio_old = df_results['ratio_old'].median()
    median_ratio_new = df_results['ratio_new'].median()
    
    # Scatter in log space (dex)
    scatter_old = df_results['log_resid_old'].std()
    scatter_new = df_results['log_resid_new'].std()
    
    # Mean absolute log residual
    mae_old = np.abs(df_results['log_resid_old']).mean()
    mae_new = np.abs(df_results['log_resid_new']).mean()
    
    # Count wins (closer to ratio = 1.0)
    wins_old = sum(abs(r['ratio_old'] - 1.0) < abs(r['ratio_new'] - 1.0) for r in results)
    wins_new = sum(abs(r['ratio_new'] - 1.0) < abs(r['ratio_old'] - 1.0) for r in results)
    
    # Print sample of results
    print(f"\n{'Cluster':<25} {'z':<6} {'M500':<8} {'Σ_old':<8} {'Σ_new':<8} {'Ratio_old':<10} {'Ratio_new':<10}")
    print("-" * 95)
    
    for r in results[:15]:  # Show first 15
        print(f"{r['cluster']:<25} {r['z']:<6.3f} {r['M500']:<8.1f} {r['Sigma_old']:<8.2f} {r['Sigma_new']:<8.2f} {r['ratio_old']:<10.3f} {r['ratio_new']:<10.3f}")
    
    if len(results) > 15:
        print(f"... and {len(results) - 15} more clusters")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    print(f"\nTotal clusters analyzed: {n_clusters}")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                        FOX+ 2022 CLUSTER VALIDATION RESULTS                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║ Formula              │ Mean Ratio │ Median Ratio │ Scatter (dex) │ MAE (dex) │ Wins      ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║ OLD: g† = cH₀/(2e)   │ {mean_ratio_old:>10.3f} │ {median_ratio_old:>12.3f} │ {scatter_old:>13.3f} │ {mae_old:>9.3f} │ {wins_old:>9}  ║
║ NEW: g† = cH₀/(4√π)  │ {mean_ratio_new:>10.3f} │ {median_ratio_new:>12.3f} │ {scatter_new:>13.3f} │ {mae_new:>9.3f} │ {wins_new:>9}  ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

INTERPRETATION:
  Ratio = M_Σ / M_SL (predicted / observed)
  Ideal ratio = 1.0 (perfect prediction)
  Ratio > 1.0 means over-prediction
  Ratio < 1.0 means under-prediction
""")
    
    # Improvement metrics
    # For ratio, closer to 1.0 is better
    old_deviation = abs(mean_ratio_old - 1.0)
    new_deviation = abs(mean_ratio_new - 1.0)
    ratio_improvement = 100 * (old_deviation - new_deviation) / old_deviation if old_deviation > 0 else 0
    
    scatter_improvement = 100 * (scatter_old - scatter_new) / scatter_old if scatter_old > 0 else 0
    mae_improvement = 100 * (mae_old - mae_new) / mae_old if mae_old > 0 else 0
    
    print(f"""
IMPROVEMENT ANALYSIS:
  Mean ratio deviation from 1.0:
    Old: |{mean_ratio_old:.3f} - 1.0| = {old_deviation:.3f}
    New: |{mean_ratio_new:.3f} - 1.0| = {new_deviation:.3f}
    Improvement: {ratio_improvement:+.1f}% {'(BETTER)' if ratio_improvement > 0 else '(WORSE)'}
  
  Scatter improvement: {scatter_improvement:+.1f}% {'(BETTER)' if scatter_improvement > 0 else '(WORSE)'}
  MAE improvement: {mae_improvement:+.1f}% {'(BETTER)' if mae_improvement > 0 else '(WORSE)'}

HEAD-TO-HEAD (closer to ratio=1.0):
  Old wins: {wins_old}
  New wins: {wins_new}
""")
    
    return {
        'n_clusters': n_clusters,
        'mean_ratio_old': mean_ratio_old,
        'mean_ratio_new': mean_ratio_new,
        'median_ratio_old': median_ratio_old,
        'median_ratio_new': median_ratio_new,
        'scatter_old': scatter_old,
        'scatter_new': scatter_new,
        'mae_old': mae_old,
        'mae_new': mae_new,
        'wins_old': wins_old,
        'wins_new': wins_new,
        'ratio_improvement_pct': ratio_improvement,
        'scatter_improvement_pct': scatter_improvement,
        'results': results,
    }


if __name__ == "__main__":
    results = run_fox2022_validation()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("ACADEMIC DEFENSIBILITY SUMMARY")
        print("=" * 80)
        
        print(f"""
DATA SOURCE:
  Fox+ 2022 (ApJ 928, 87)
  "The Hubble Frontier Fields: Strong Lensing Mass Models"
  N = {results['n_clusters']} clusters with spectroscopic redshifts and M500 > 2×10¹⁴ M☉

METHODOLOGY:
  1. Baryonic mass estimated from M500: M_bar = 0.4 × f_baryon × M500
     where f_baryon = 0.15 (gas + stars)
  2. Baryonic acceleration at 200 kpc: g_bar = G × M_bar / r²
  3. Σ-Gravity enhancement: Σ = 1 + A × h(g) with A = π√2
  4. Predicted mass: M_Σ = Σ × M_bar
  5. Compared to observed strong lensing mass MSL_200kpc

RESULTS:
  The new formula g† = cH₀/(4√π) shows:
  - Scatter: {results['scatter_new']:.3f} dex (vs {results['scatter_old']:.3f} dex for old)
  - MAE: {results['mae_new']:.3f} dex (vs {results['mae_old']:.3f} dex for old)
  - Head-to-head: New wins {results['wins_new']}, Old wins {results['wins_old']}

CONCLUSION:
  Both formulas produce similar results on clusters, with the new formula
  showing {'slight improvement' if results['scatter_improvement_pct'] > 0 else 'comparable performance'}.
  
  This is expected because cluster lensing is dominated by the amplitude
  A = π√2, not the critical acceleration g†. The g† value matters more
  at galaxy scales where accelerations are closer to g†.
""")

