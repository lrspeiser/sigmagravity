"""
Cross-Catalog Σ-Gravity Consistency Test
==========================================

This script analyzes mass-distance correlations and coherence parameter ε
across different LIGO/Virgo observing runs to test whether ε is stable
(as Σ-Gravity predicts).

Uses the cumulative GWTC CSV from GWOSC API.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

LAMBDA_COH_KPC = 2.2
M_BAR = 60.0  # Assumed intrinsic mass for gap events

def load_gwtc_csv():
    """Load the cumulative GWTC catalog from GWOSC."""
    csv_path = Path(__file__).parent / "GWTC_all_events.csv"
    df = pd.read_csv(csv_path)
    print(f"[load_gwtc_csv] Loaded {len(df)} rows from GWTC cumulative catalog")
    return df

def categorize_by_run(df):
    """Categorize events by observing run based on catalog name."""
    run_map = {
        'O1+O2': ['GWTC-1-confident', 'GWTC-1-marginal'],
        'O3a': ['GWTC-2-confident', 'GWTC-2-marginal', 'GWTC-2.1-confident', 'GWTC-2.1-marginal'],
        'O3b': ['GWTC-3-confident', 'GWTC-3-marginal'],
        'O4a': ['GWTC-4.0', 'GWTC-4.0-confident'],
    }
    
    def get_run(cat_name):
        if pd.isna(cat_name):
            return None
        for run, cats in run_map.items():
            if cat_name in cats:
                return run
        # Fallback based on naming
        if 'GWTC-1' in str(cat_name):
            return 'O1+O2'
        elif 'GWTC-2' in str(cat_name):
            return 'O3a'
        elif 'GWTC-3' in str(cat_name):
            return 'O3b'
        elif 'GWTC-4' in str(cat_name):
            return 'O4a'
        return None
    
    df['run'] = df['catalog.shortName'].apply(get_run)
    return df

def deduplicate_events(df):
    """Keep only one row per unique event (latest version)."""
    # Sort by version descending, then take first occurrence of each commonName
    df_sorted = df.sort_values('version', ascending=False)
    df_dedup = df_sorted.drop_duplicates(subset='commonName', keep='first')
    n_removed = len(df) - len(df_dedup)
    print(f"[deduplicate] {len(df)} rows → {len(df_dedup)} unique events ({n_removed} duplicates removed)")
    return df_dedup

def compute_epsilon(mass, dist_mpc):
    """Compute coherence parameter ε for a gap event."""
    C = mass / M_BAR
    N = dist_mpc * 1000 / LAMBDA_COH_KPC  # Convert Mpc to kpc
    if N > 0 and C > 1:
        return np.power(C, 1/N) - 1
    return np.nan

def analyze_run(df_run, run_name):
    """Analyze a single observing run."""
    # Filter for valid data
    valid = (
        df_run['total_mass_source'].notna() & 
        df_run['luminosity_distance'].notna() &
        df_run['network_matched_filter_snr'].notna() &
        (df_run['network_matched_filter_snr'] > 8)
    )
    df_valid = df_run[valid].copy()
    n_events = len(df_valid)
    
    if n_events < 5:
        return None
    
    mass = df_valid['total_mass_source'].values
    dist = df_valid['luminosity_distance'].values
    chi_eff = df_valid['chi_eff'].values
    
    # Mass-distance correlation
    corr, p_val = stats.pearsonr(dist, mass)
    
    # Gap events
    gap_mask = mass >= 100
    n_gap = np.sum(gap_mask)
    
    # Compute ε for gap events
    epsilons = []
    for m, d in zip(mass[gap_mask], dist[gap_mask]):
        eps = compute_epsilon(m, d)
        if not np.isnan(eps):
            epsilons.append(eps)
    
    eps_median = np.median(epsilons) if epsilons else np.nan
    eps_cv = (np.std(epsilons) / np.mean(epsilons) * 100) if len(epsilons) > 1 else np.nan
    
    # Spin analysis for gap events
    gap_spins = chi_eff[gap_mask & ~np.isnan(chi_eff)]
    normal_spins = chi_eff[~gap_mask & ~np.isnan(chi_eff)]
    
    spin_median_gap = np.median(gap_spins) if len(gap_spins) > 0 else np.nan
    spin_median_normal = np.median(normal_spins) if len(normal_spins) > 0 else np.nan
    
    # High-spin fraction (χ_eff > 0.3, signed)
    high_spin_gap = np.mean(gap_spins > 0.3) * 100 if len(gap_spins) > 0 else np.nan
    high_spin_normal = np.mean(normal_spins > 0.3) * 100 if len(normal_spins) > 0 else np.nan
    
    result = {
        'run': run_name,
        'n_events': n_events,
        'n_gap': n_gap,
        'correlation_r': corr,
        'correlation_p': p_val,
        'epsilon_median': eps_median,
        'epsilon_cv': eps_cv,
        'spin_median_gap': spin_median_gap,
        'spin_median_normal': spin_median_normal,
        'high_spin_gap_pct': high_spin_gap,
        'high_spin_normal_pct': high_spin_normal,
    }
    
    return result

def main():
    print("=" * 70)
    print("CROSS-CATALOG Σ-GRAVITY CONSISTENCY TEST")
    print("=" * 70)
    
    df = load_gwtc_csv()
    df = categorize_by_run(df)
    df = deduplicate_events(df)
    
    # Analyze each run
    runs = ['O1+O2', 'O3a', 'O3b', 'O4a']
    results = []
    
    for run in runs:
        df_run = df[df['run'] == run]
        if len(df_run) == 0:
            print(f"\n{run}: No events found")
            continue
        
        result = analyze_run(df_run, run)
        if result:
            results.append(result)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("CROSS-CATALOG RESULTS")
    print("=" * 70)
    
    print(f"\n{'Run':<10} {'N':<6} {'Gap':<5} {'r':<8} {'p-value':<12} {'ε median':<12} {'ε CV%':<8}")
    print("-" * 70)
    
    epsilons_all = []
    for r in results:
        print(f"{r['run']:<10} {r['n_events']:<6} {r['n_gap']:<5} "
              f"{r['correlation_r']:<8.3f} {r['correlation_p']:<12.2e} "
              f"{r['epsilon_median']:<12.2e} {r['epsilon_cv']:<8.1f}" if not np.isnan(r['epsilon_cv']) 
              else f"{r['run']:<10} {r['n_events']:<6} {r['n_gap']:<5} "
              f"{r['correlation_r']:<8.3f} {r['correlation_p']:<12.2e} "
              f"{r['epsilon_median']:<12.2e} {'N/A':<8}")
        if not np.isnan(r['epsilon_median']):
            epsilons_all.append(r['epsilon_median'])
    
    # Cross-catalog ε consistency
    if len(epsilons_all) >= 2:
        eps_mean = np.mean(epsilons_all)
        eps_std = np.std(epsilons_all)
        eps_cv_cross = eps_std / eps_mean * 100
        
        print(f"\n{'CROSS-CATALOG ε CONSISTENCY':}")
        print(f"  Runs with gap events: {len(epsilons_all)}")
        print(f"  ε range: [{min(epsilons_all):.2e}, {max(epsilons_all):.2e}]")
        print(f"  ε mean: {eps_mean:.2e}")
        print(f"  ε CV (cross-catalog): {eps_cv_cross:.1f}%")
        
        if eps_cv_cross < 50:
            print(f"  ✓ CONSISTENT ε across catalogs! (CV < 50%)")
            print(f"    This strongly supports Σ-Gravity universality.")
        else:
            print(f"  ⚠ Moderate variation in ε across catalogs")
    
    # Spin analysis summary
    print(f"\n{'SPIN DISTRIBUTION SUMMARY':<40}")
    print(f"{'Run':<10} {'Gap χ_eff':<12} {'Normal χ_eff':<14} {'Gap >0.3':<10} {'Normal >0.3':<12}")
    print("-" * 70)
    
    for r in results:
        gap_spin = f"{r['spin_median_gap']:.3f}" if not np.isnan(r['spin_median_gap']) else "N/A"
        norm_spin = f"{r['spin_median_normal']:.3f}" if not np.isnan(r['spin_median_normal']) else "N/A"
        gap_high = f"{r['high_spin_gap_pct']:.1f}%" if not np.isnan(r['high_spin_gap_pct']) else "N/A"
        norm_high = f"{r['high_spin_normal_pct']:.1f}%" if not np.isnan(r['high_spin_normal_pct']) else "N/A"
        print(f"{r['run']:<10} {gap_spin:<12} {norm_spin:<14} {gap_high:<10} {norm_high:<12}")
    
    # Overall interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Check if correlation grows with distance/sensitivity
    if len(results) >= 3:
        correlations = [r['correlation_r'] for r in results]
        print(f"\nMass-distance correlations by run: {[f'{r:.3f}' for r in correlations]}")
        
        if all(r > 0 for r in correlations):
            print("  ✓ ALL runs show POSITIVE mass-distance correlation")
            print("    This is consistent with Σ-Gravity coherence accumulation.")
        
        # Check spin consistency
        gap_spins = [r['spin_median_gap'] for r in results if not np.isnan(r['spin_median_gap'])]
        if gap_spins and all(s < 0.2 for s in gap_spins):
            print(f"\n  ✓ ALL runs show LOW gap event spins (median < 0.2)")
            print("    This is INCONSISTENT with hierarchical mergers (expect ~0.5)")
            print("    This SUPPORTS Σ-Gravity (gap events are enhanced normal BBH)")
    
    return results

if __name__ == "__main__":
    results = main()
