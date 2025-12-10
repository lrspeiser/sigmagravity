#!/usr/bin/env python3
"""
Cluster Holdout Validation for Σ-Gravity

This script implements a proper train/test split for cluster parameter calibration
to address the concern that L_0 and n were calibrated on the same 42 clusters
used for validation.

Strategy:
- Split 42 clusters into calibration (30) and holdout (12) sets
- Calibrate L_0 and n on calibration set only
- Report performance on holdout set as true out-of-sample validation
- Use stratified split by redshift to ensure representative samples

The holdout results can then be reported as genuine predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G = 6.674e-11  # m^3 kg^-1 s^-2
c = 2.998e8    # m/s
H0 = 70 * 1e3 / 3.086e22  # s^-1 (70 km/s/Mpc)
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19

# Canonical parameters (derived, not calibrated)
g_dagger = c * H0 / (4 * np.sqrt(np.pi))  # ~9.6e-11 m/s^2
A0 = np.exp(1 / (2 * np.pi))  # ~1.173


def h_function(g_N: np.ndarray) -> np.ndarray:
    """Acceleration-dependent enhancement function."""
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)


def predict_cluster_mass(M_bar: float, r_kpc: float, L_0: float, n: float) -> float:
    """
    Predict enhanced mass for a cluster.
    
    For clusters (D=1): A = A0 × (L/L_0)^n
    where L is the path length through baryons (~r_kpc for clusters)
    """
    # Path length through baryons (approximated as radius for spherical systems)
    L = r_kpc  # kpc
    
    # Amplitude for dispersion-dominated system
    A = A0 * (L / L_0) ** n
    
    # Acceleration at measurement radius
    r_m = r_kpc * kpc_to_m
    g_N = G * M_bar * M_sun / r_m**2
    
    # Enhancement (full coherence C=1 for this simplified model)
    h = h_function(g_N)
    Sigma = 1 + A * h
    
    return M_bar * Sigma


def load_clusters() -> pd.DataFrame:
    """Load cluster data from Fox+ 2022."""
    data_dir = Path(__file__).parent.parent / "data"
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    
    df = pd.read_csv(cluster_file)
    
    # Apply quality filters
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes') &
        (df['M500_1e14Msun'] > 2.0)
    ].copy()
    
    # Compute derived quantities
    f_baryon = 0.15
    df_valid['M_bar'] = 0.4 * f_baryon * df_valid['M500_1e14Msun'] * 1e14  # M_sun
    df_valid['M_lens'] = df_valid['MSL_200kpc_1e12Msun'] * 1e12  # M_sun
    df_valid['r_kpc'] = 200.0
    
    return df_valid


def stratified_split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split clusters into train/test sets, stratified by redshift.
    
    This ensures both sets have similar redshift distributions.
    """
    np.random.seed(seed)
    
    # Create redshift bins
    df = df.copy()
    df['z_bin'] = pd.qcut(df['z_lens'], q=4, labels=['low', 'mid-low', 'mid-high', 'high'])
    
    train_idx = []
    test_idx = []
    
    for bin_name in df['z_bin'].unique():
        bin_df = df[df['z_bin'] == bin_name]
        n_test = max(1, int(len(bin_df) * test_frac))
        
        shuffled = bin_df.sample(frac=1, random_state=seed)
        test_idx.extend(shuffled.index[:n_test].tolist())
        train_idx.extend(shuffled.index[n_test:].tolist())
    
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def calibrate_parameters(train_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calibrate L_0 and n on training set.
    
    Returns: (L_0, n, train_scatter)
    """
    def objective(params):
        L_0, n = params
        if L_0 <= 0 or n <= 0:
            return 1e10
        
        log_ratios = []
        for _, row in train_df.iterrows():
            M_pred = predict_cluster_mass(row['M_bar'], row['r_kpc'], L_0, n)
            M_obs = row['M_lens']
            log_ratios.append(np.log10(M_pred / M_obs))
        
        # Minimize scatter (RMS of log ratios)
        return np.std(log_ratios)
    
    # Initial guess
    x0 = [0.4, 0.27]
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    L_0, n = result.x
    train_scatter = result.fun
    
    return L_0, n, train_scatter


def evaluate_on_holdout(test_df: pd.DataFrame, L_0: float, n: float) -> Dict:
    """
    Evaluate model on holdout set.
    
    Returns dict with performance metrics.
    """
    ratios = []
    log_ratios = []
    
    results = []
    for _, row in test_df.iterrows():
        M_pred = predict_cluster_mass(row['M_bar'], row['r_kpc'], L_0, n)
        M_obs = row['M_lens']
        ratio = M_pred / M_obs
        ratios.append(ratio)
        log_ratios.append(np.log10(ratio))
        results.append({
            'cluster': row['cluster'],
            'z': row['z_lens'],
            'M_pred': M_pred,
            'M_obs': M_obs,
            'ratio': ratio
        })
    
    return {
        'median_ratio': np.median(ratios),
        'mean_ratio': np.mean(ratios),
        'scatter_dex': np.std(log_ratios),
        'min_ratio': np.min(ratios),
        'max_ratio': np.max(ratios),
        'n_clusters': len(test_df),
        'results': results
    }


def run_holdout_validation(n_splits: int = 10):
    """
    Run holdout validation with multiple random splits.
    
    This tests robustness of the calibration.
    """
    print("=" * 70)
    print("CLUSTER HOLDOUT VALIDATION FOR Σ-GRAVITY")
    print("=" * 70)
    print()
    
    # Load data
    df = load_clusters()
    print(f"Loaded {len(df)} clusters from Fox+ 2022")
    print()
    
    # Run multiple splits
    all_holdout_metrics = []
    all_L0 = []
    all_n = []
    
    print("Running holdout validation with multiple random splits...")
    print()
    
    for seed in range(n_splits):
        train_df, test_df = stratified_split(df, test_frac=0.3, seed=seed)
        
        # Calibrate on training set
        L_0, n, train_scatter = calibrate_parameters(train_df)
        all_L0.append(L_0)
        all_n.append(n)
        
        # Evaluate on holdout
        holdout_metrics = evaluate_on_holdout(test_df, L_0, n)
        all_holdout_metrics.append(holdout_metrics)
        
        print(f"Split {seed+1}: Train={len(train_df)}, Test={len(test_df)} | "
              f"L_0={L_0:.3f}, n={n:.3f} | "
              f"Holdout median={holdout_metrics['median_ratio']:.3f}, "
              f"scatter={holdout_metrics['scatter_dex']:.3f} dex")
    
    print()
    print("=" * 70)
    print("SUMMARY ACROSS ALL SPLITS")
    print("=" * 70)
    
    # Aggregate results
    median_ratios = [m['median_ratio'] for m in all_holdout_metrics]
    scatters = [m['scatter_dex'] for m in all_holdout_metrics]
    
    print(f"\nCalibrated parameters:")
    print(f"  L_0 = {np.mean(all_L0):.3f} ± {np.std(all_L0):.3f} kpc")
    print(f"  n   = {np.mean(all_n):.3f} ± {np.std(all_n):.3f}")
    
    print(f"\nHoldout performance (out-of-sample):")
    print(f"  Median ratio (pred/obs) = {np.mean(median_ratios):.3f} ± {np.std(median_ratios):.3f}")
    print(f"  Scatter = {np.mean(scatters):.3f} ± {np.std(scatters):.3f} dex")
    print(f"  Range of ratios: {np.min([m['min_ratio'] for m in all_holdout_metrics]):.2f} - "
          f"{np.max([m['max_ratio'] for m in all_holdout_metrics]):.2f}")
    
    print()
    print("=" * 70)
    print("CANONICAL SPLIT (seed=42) - FOR PAPER")
    print("=" * 70)
    
    # Run canonical split for paper
    train_df, test_df = stratified_split(df, test_frac=0.3, seed=42)
    L_0, n, train_scatter = calibrate_parameters(train_df)
    holdout_metrics = evaluate_on_holdout(test_df, L_0, n)
    
    print(f"\nTraining set: {len(train_df)} clusters")
    print(f"Holdout set: {len(test_df)} clusters")
    print(f"\nCalibrated on training set:")
    print(f"  L_0 = {L_0:.3f} kpc")
    print(f"  n   = {n:.3f}")
    print(f"  Training scatter = {train_scatter:.3f} dex")
    
    print(f"\nHoldout set performance (TRUE OUT-OF-SAMPLE):")
    print(f"  Median ratio = {holdout_metrics['median_ratio']:.3f}")
    print(f"  Mean ratio   = {holdout_metrics['mean_ratio']:.3f}")
    print(f"  Scatter      = {holdout_metrics['scatter_dex']:.3f} dex")
    print(f"  Range        = {holdout_metrics['min_ratio']:.2f} - {holdout_metrics['max_ratio']:.2f}")
    
    print(f"\nHoldout cluster details:")
    for r in holdout_metrics['results']:
        status = "✓" if 0.5 < r['ratio'] < 2.0 else "✗"
        print(f"  {status} {r['cluster']:30s} z={r['z']:.3f}  ratio={r['ratio']:.3f}")
    
    # Compare with full-sample calibration
    print()
    print("=" * 70)
    print("COMPARISON: HOLDOUT vs FULL-SAMPLE CALIBRATION")
    print("=" * 70)
    
    # Full sample calibration (current approach)
    L_0_full, n_full, _ = calibrate_parameters(df)
    full_metrics = evaluate_on_holdout(df, L_0_full, n_full)
    
    print(f"\nFull-sample calibration (current paper):")
    print(f"  L_0 = {L_0_full:.3f} kpc, n = {n_full:.3f}")
    print(f"  Median ratio = {full_metrics['median_ratio']:.3f}")
    print(f"  Scatter = {full_metrics['scatter_dex']:.3f} dex")
    
    print(f"\nHoldout validation (recommended):")
    print(f"  L_0 = {L_0:.3f} kpc, n = {n:.3f}")
    print(f"  Holdout median ratio = {holdout_metrics['median_ratio']:.3f}")
    print(f"  Holdout scatter = {holdout_metrics['scatter_dex']:.3f} dex")
    
    # Key finding
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    holdout_success = 0.8 < holdout_metrics['median_ratio'] < 1.2 and holdout_metrics['scatter_dex'] < 0.2
    if holdout_success:
        print("\n✓ HOLDOUT VALIDATION SUCCESSFUL")
        print("  The model generalizes well to unseen clusters.")
        print("  Recommended to update paper with holdout results.")
    else:
        print("\n✗ HOLDOUT VALIDATION SHOWS DEGRADATION")
        print("  Consider Option B: acknowledge calibration on clusters,")
        print("  emphasize that SPARC/MW are independent validation.")
    
    return {
        'holdout_metrics': holdout_metrics,
        'L_0': L_0,
        'n': n,
        'train_df': train_df,
        'test_df': test_df
    }


if __name__ == "__main__":
    results = run_holdout_validation(n_splits=10)

