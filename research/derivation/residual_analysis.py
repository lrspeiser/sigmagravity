#!/usr/bin/env python3
"""
RESIDUAL ANALYSIS FOR K(R)
==========================

The tanh model gives R² = 0.33 on individual stars but R² = 0.99 on bins.
This script analyzes what's causing the scatter.

Questions:
1. Is it random noise or systematic?
2. Does residual correlate with other variables?
3. Can we identify distinct populations?
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def analyze_residuals(
    filepath: str = None,
    model_params: dict = None,
    output_prefix: str = None
):
    """
    Comprehensive residual analysis.
    """
    
    # Default tanh parameters from your fit
    if model_params is None:
        model_params = {'A': 0.95, 'R1': 6.75, 'w': 1.78, 'c': 1.02}
    
    print("="*70)
    print("  RESIDUAL ANALYSIS")
    print("="*70)
    
    # Load data
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "gaia" / "outputs" / "mw_rar_starlevel_full.csv"
    
    print(f"\n  Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Compute K
    if 'K' not in df.columns:
        df['K'] = 10**(df['log10_g_obs'] - df['log10_g_bar']) - 1
    
    # Standardize columns
    rename_map = {'R_kpc': 'R', 'z_kpc': 'z', 'log10_g_bar': 'g_bar', 'log10_g_obs': 'g_obs'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Quality cuts
    valid = (df['K'] > -0.5) & (df['K'] < 50) & (df['R'] > 0) & (df['R'] < 25)
    df = df[valid].copy()
    
    print(f"  Stars after cuts: {len(df):,}")
    
    # Compute model prediction
    p = model_params
    df['K_pred'] = p['A'] * np.tanh((df['R'] - p['R1']) / p['w']) + p['c']
    
    # Compute residuals
    df['residual'] = df['K'] - df['K_pred']
    df['residual_frac'] = df['residual'] / np.maximum(df['K_pred'], 0.1)
    df['residual_abs'] = np.abs(df['residual'])
    
    # === BASIC STATS ===
    print(f"\n  RESIDUAL STATISTICS:")
    print(f"    Mean: {df['residual'].mean():.4f}")
    print(f"    Std:  {df['residual'].std():.4f}")
    print(f"    Median: {df['residual'].median():.4f}")
    print(f"    IQR: [{df['residual'].quantile(0.25):.4f}, {df['residual'].quantile(0.75):.4f}]")
    print(f"    Skew: {stats.skew(df['residual']):.3f}")
    print(f"    Kurtosis: {stats.kurtosis(df['residual']):.3f}")
    
    # Test for normality
    sample = df['residual'].sample(min(5000, len(df)), random_state=42)
    stat, pval = stats.normaltest(sample)
    print(f"    Normality test p-value: {pval:.2e} ({'Normal' if pval > 0.05 else 'Non-normal'})")
    
    # === CORRELATION ANALYSIS ===
    print(f"\n  RESIDUAL CORRELATIONS:")
    
    correlations = {}
    for col in ['R', 'z', 'g_bar', 'g_obs', 'v_obs_kms', 'v_err_kms', 'Sigma_loc_Msun_pc2']:
        if col in df.columns:
            valid_both = df[col].notna() & df['residual'].notna()
            if valid_both.sum() > 100:
                corr, pval = stats.spearmanr(df.loc[valid_both, col], df.loc[valid_both, 'residual'])
                correlations[col] = {'spearman_r': float(corr), 'p_value': float(pval)}
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"    {col:25s}: r = {corr:+.4f} {sig}")
    
    # === RANDOM FOREST FEATURE IMPORTANCE ===
    print(f"\n  FEATURE IMPORTANCE (RandomForest on residuals):")
    
    feature_cols = [c for c in ['R', 'z', 'g_bar', 'v_obs_kms', 'v_err_kms'] if c in df.columns and df[c].notna().sum() > len(df)*0.5]
    
    importances = {}
    r2_residual = 0
    
    if len(feature_cols) >= 2:
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            X = df[feature_cols].fillna(0).values
            y = df['residual'].values
            
            # Subsample for speed
            idx = np.random.choice(len(X), min(30000, len(X)), replace=False)
            X_sub, y_sub = X[idx], y[idx]
            
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_sub, y_sub)
            
            importances = dict(zip(feature_cols, rf.feature_importances_))
            for col, imp in sorted(importances.items(), key=lambda x: -x[1]):
                print(f"    {col:25s}: {imp:.4f}")
            
            # R² of residual prediction
            y_pred_residual = rf.predict(X_sub)
            ss_res = np.sum((y_sub - y_pred_residual)**2)
            ss_tot = np.sum((y_sub - np.mean(y_sub))**2)
            r2_residual = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            print(f"\n    R² of residual prediction: {r2_residual:.4f}")
            print(f"    → {r2_residual*100:.1f}% of scatter is SYSTEMATIC")
            print(f"    → {(1-r2_residual)*100:.1f}% of scatter is RANDOM NOISE")
            
        except ImportError:
            print("    (sklearn not available)")
    
    # === BINNED RESIDUAL ANALYSIS ===
    print(f"\n  BINNED RESIDUAL ANALYSIS:")
    
    R_bins = [0, 3, 5, 6, 7, 8, 9, 10, 12, 15, 25]
    
    binned_stats = []
    for i in range(len(R_bins)-1):
        mask = (df['R'] >= R_bins[i]) & (df['R'] < R_bins[i+1])
        sub = df[mask]
        if len(sub) > 10:
            stat = {
                'R_min': R_bins[i],
                'R_max': R_bins[i+1],
                'N': len(sub),
                'K_mean': float(sub['K'].mean()),
                'K_std': float(sub['K'].std()),
                'K_pred_mean': float(sub['K_pred'].mean()),
                'residual_mean': float(sub['residual'].mean()),
                'residual_std': float(sub['residual'].std()),
                'CV': float(sub['K'].std() / sub['K'].mean() * 100),
            }
            binned_stats.append(stat)
            bias = 'HIGH' if stat['residual_mean'] > 0.05 else 'LOW' if stat['residual_mean'] < -0.05 else 'OK'
            print(f"    R={R_bins[i]:2d}-{R_bins[i+1]:2d}: N={stat['N']:6,}, "
                  f"K={stat['K_mean']:.2f}±{stat['K_std']:.2f}, "
                  f"resid={stat['residual_mean']:+.3f}±{stat['residual_std']:.3f} [{bias}]")
    
    # === z-BINNED ANALYSIS ===
    if 'z' in df.columns:
        print(f"\n  z-HEIGHT ANALYSIS:")
        z_bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0]
        for i in range(len(z_bins)-1):
            mask = (np.abs(df['z']) >= z_bins[i]) & (np.abs(df['z']) < z_bins[i+1])
            sub = df[mask]
            if len(sub) > 100:
                print(f"    |z|={z_bins[i]:.1f}-{z_bins[i+1]:.1f} kpc: N={len(sub):6,}, "
                      f"K={sub['K'].mean():.3f}±{sub['K'].std():.3f}, resid={sub['residual'].mean():+.3f}")
    
    # === OUTLIER ANALYSIS ===
    print(f"\n  OUTLIER ANALYSIS:")
    
    q95 = df['K'].quantile(0.95)
    q05 = df['K'].quantile(0.05)
    
    high_K = df[df['K'] > q95]
    low_K = df[df['K'] < q05]
    middle_K = df[(df['K'] >= q05) & (df['K'] <= q95)]
    
    print(f"    High-K outliers (K > {q95:.2f}, top 5%): N={len(high_K)}")
    if 'z' in df.columns:
        print(f"      Mean |z|: {np.abs(high_K['z']).mean():.3f} kpc")
    print(f"      Mean R: {high_K['R'].mean():.2f} kpc")
    print(f"      Mean residual: {high_K['residual'].mean():+.3f}")
    
    print(f"    Low-K outliers (K < {q05:.2f}, bottom 5%): N={len(low_K)}")
    if 'z' in df.columns:
        print(f"      Mean |z|: {np.abs(low_K['z']).mean():.3f} kpc")
    print(f"      Mean R: {low_K['R'].mean():.2f} kpc")
    print(f"      Mean residual: {low_K['residual'].mean():+.3f}")
    
    # === MEASUREMENT ERROR ANALYSIS ===
    print(f"\n  MEASUREMENT ERROR ANALYSIS:")
    if 'v_err_kms' in df.columns:
        err_q = df['v_err_kms'].quantile([0.1, 0.5, 0.9])
        print(f"    Velocity error: 10%={err_q.iloc[0]:.1f}, 50%={err_q.iloc[1]:.1f}, 90%={err_q.iloc[2]:.1f} km/s")
        
        # Correlation between error and residual magnitude
        corr, _ = stats.spearmanr(df['v_err_kms'], df['residual_abs'])
        print(f"    Correlation (v_err vs |residual|): r = {corr:+.4f}")
        
        # High vs low error comparison
        low_err = df[df['v_err_kms'] < df['v_err_kms'].quantile(0.25)]
        high_err = df[df['v_err_kms'] > df['v_err_kms'].quantile(0.75)]
        print(f"    Low-error stars:  |residual| = {low_err['residual_abs'].mean():.3f}")
        print(f"    High-error stars: |residual| = {high_err['residual_abs'].mean():.3f}")
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    systematic_frac = r2_residual
    random_frac = 1 - r2_residual
    
    print(f"\n  The K(R) = tanh model explains {0.33*100:.0f}% of total variance")
    print(f"  Of the remaining {67:.0f}% scatter:")
    print(f"    - {systematic_frac*67:.1f}% ({systematic_frac*100:.1f}% of scatter) is SYSTEMATIC")
    print(f"    - {random_frac*67:.1f}% ({random_frac*100:.1f}% of scatter) is RANDOM NOISE")
    
    if systematic_frac < 0.1:
        conclusion = "RANDOM_NOISE_DOMINATED"
        recommendation = "The tanh model captures all available signal. Scatter is observational noise."
    elif systematic_frac < 0.3:
        conclusion = "MOSTLY_RANDOM"
        recommendation = "Small systematic component exists but tanh model is nearly optimal."
    else:
        conclusion = "SYSTEMATIC_COMPONENT"
        recommendation = "Significant systematic scatter - consider additional variables or model refinement."
    
    print(f"\n  CONCLUSION: {conclusion}")
    print(f"  {recommendation}")
    
    # === SAVE RESULTS ===
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if output_prefix is None:
        output_prefix = str(output_dir / 'residual_analysis')
    
    results = {
        'model_params': model_params,
        'n_stars': len(df),
        'residual_stats': {
            'mean': float(df['residual'].mean()),
            'std': float(df['residual'].std()),
            'median': float(df['residual'].median()),
            'skew': float(stats.skew(df['residual'])),
            'kurtosis': float(stats.kurtosis(df['residual'])),
        },
        'correlations': correlations,
        'feature_importance': {k: float(v) for k, v in importances.items()},
        'binned_stats': binned_stats,
        'conclusion': {
            'systematic_fraction': systematic_frac,
            'random_fraction': random_frac,
            'type': conclusion,
            'recommendation': recommendation,
        }
    }
    
    with open(f'{output_prefix}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to {output_prefix}.json")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to Gaia CSV')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    analyze_residuals(args.data, output_prefix=args.output)
