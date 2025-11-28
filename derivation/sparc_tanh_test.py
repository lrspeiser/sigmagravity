#!/usr/bin/env python3
"""
SPARC TANH TEST
===============

Test if the tanh K(R) form discovered from MW holds for external galaxies.

Key questions:
1. Does K = A × tanh((R - R_c) / w) + c fit SPARC galaxies?
2. Does R_c scale with galaxy size (R_disk)?
3. Is the transition universal or MW-specific?

MW result: K = 0.95 × tanh((R - 6.75) / 1.78) + 1.02
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
import glob
import os
from pathlib import Path
import json
import time


def load_rotmod_galaxy(filepath):
    """Load a single galaxy rotation curve from Rotmod_LTG."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract distance from comment
    distance = None
    for line in lines:
        if 'Distance' in line:
            try:
                distance = float(line.split('=')[1].split('Mpc')[0].strip())
            except:
                pass
    
    # Load data
    df = pd.read_csv(filepath, sep=r'\s+', comment='#',
                     names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'],
                     on_bad_lines='skip')
    
    # Remove any NaN rows
    df = df.dropna(subset=['Rad', 'Vobs', 'Vgas', 'Vdisk'])
    
    return {
        'name': Path(filepath).stem.replace('_rotmod', ''),
        'r': df['Rad'].values,
        'v_obs': df['Vobs'].values,
        'v_err': df['errV'].values,
        'v_disk': df['Vdisk'].values,
        'v_gas': df['Vgas'].values,
        'v_bulge': df['Vbul'].values if 'Vbul' in df else np.zeros(len(df)),
        'distance_Mpc': distance,
    }


def compute_K(gal):
    """Compute enhancement factor K = g_obs/g_bar - 1."""
    r = gal['r']
    v_obs = gal['v_obs']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal['v_bulge'] if 'v_bulge' in gal else np.zeros_like(r)
    
    # g_obs = v_obs^2 / R
    g_obs = v_obs**2 / np.maximum(r, 0.01)
    
    # v_bar^2 = v_disk^2 + v_gas^2 + v_bulge^2 (signed)
    v_bar_sq = np.sign(v_disk) * v_disk**2 + np.sign(v_gas) * v_gas**2 + v_bulge**2
    v_bar = np.sqrt(np.maximum(v_bar_sq, 0))
    
    # g_bar = v_bar^2 / R
    g_bar = v_bar**2 / np.maximum(r, 0.01)
    
    # K = g_obs/g_bar - 1
    K = g_obs / np.maximum(g_bar, 1e-10) - 1
    
    # Filter extreme values
    valid = np.isfinite(K) & (K > -0.5) & (K < 100) & (r > 0)
    
    return r[valid], K[valid], g_bar[valid], g_obs[valid]


def tanh_model(R, A, R_c, w, c):
    """Tanh transition model for K(R)."""
    return A * np.tanh((R - R_c) / w) + c


def linear_model(R, m, b):
    """Simple linear model."""
    return m * R + b


def fit_tanh_to_galaxy(r, K, R_disk=None):
    """Fit tanh model to galaxy K(R) data."""
    if len(r) < 4:
        return None, None, None
    
    # Initial guess based on data
    R_max = r.max()
    K_min, K_max = K.min(), K.max()
    
    # Bounds
    bounds = [
        (0.01, 20),       # A: amplitude
        (0.1, R_max * 2), # R_c: transition radius
        (0.1, R_max),     # w: transition width
        (-5, 20),         # c: offset
    ]
    
    def objective(params):
        try:
            K_pred = tanh_model(r, *params)
            return np.mean((K - K_pred)**2)
        except:
            return 1e30
    
    result = differential_evolution(objective, bounds, maxiter=200, seed=42, polish=True)
    
    A, R_c, w, c = result.x
    K_pred = tanh_model(r, A, R_c, w, c)
    
    # Compute R²
    ss_res = np.sum((K - K_pred)**2)
    ss_tot = np.sum((K - np.mean(K))**2)
    r2_tanh = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Also fit linear for comparison
    try:
        popt_lin, _ = curve_fit(linear_model, r, K, p0=[0.1, 0])
        K_pred_lin = linear_model(r, *popt_lin)
        ss_res_lin = np.sum((K - K_pred_lin)**2)
        r2_linear = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0
    except:
        r2_linear = 0
    
    return {
        'A': A, 'R_c': R_c, 'w': w, 'c': c,
        'r2_tanh': r2_tanh,
        'r2_linear': r2_linear,
        'n_points': len(r),
        'R_max': R_max,
    }


def run_sparc_test(data_dir: str = None, output_file: str = None):
    """Run tanh fit on all SPARC galaxies."""
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "Rotmod_LTG"
    
    print("="*70)
    print("  SPARC TANH K(R) TEST")
    print("="*70)
    
    # Load galaxy summary for R_disk
    summary_file = Path(__file__).parent.parent / "data" / "sparc" / "sparc_combined.csv"
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        summary_dict = dict(zip(summary_df['galaxy_name'], summary_df.to_dict('records')))
    else:
        summary_dict = {}
    
    # Find all rotation curve files
    rotmod_files = glob.glob(str(data_dir / "*_rotmod.dat"))
    print(f"\n  Found {len(rotmod_files)} galaxies in Rotmod_LTG")
    
    results = []
    
    print("\n  Fitting tanh model to each galaxy...")
    print("  " + "-"*60)
    
    for filepath in sorted(rotmod_files):
        try:
            gal = load_rotmod_galaxy(filepath)
            r, K, g_bar, g_obs = compute_K(gal)
            
            if len(r) < 5:
                continue
            
            # Get R_disk from summary
            gal_name = gal['name']
            R_disk = summary_dict.get(gal_name, {}).get('R_disk', None)
            
            fit = fit_tanh_to_galaxy(r, K, R_disk)
            
            if fit is None:
                continue
            
            result = {
                'name': gal_name,
                'R_disk': R_disk,
                **fit,
            }
            results.append(result)
            
            # Print progress
            better = '✓' if fit['r2_tanh'] > fit['r2_linear'] else ' '
            print(f"    {gal_name:15s}: R²_tanh={fit['r2_tanh']:.3f}, R²_lin={fit['r2_linear']:.3f} "
                  f"R_c={fit['R_c']:.2f}, A={fit['A']:.2f} {better}")
            
        except Exception as e:
            print(f"    {Path(filepath).stem}: FAILED ({e})")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    
    n_total = len(df)
    n_tanh_better = (df['r2_tanh'] > df['r2_linear']).sum()
    n_good_fit = (df['r2_tanh'] > 0.5).sum()
    
    print(f"\n  Total galaxies fitted: {n_total}")
    print(f"  Tanh better than linear: {n_tanh_better}/{n_total} ({n_tanh_better/n_total*100:.0f}%)")
    print(f"  Good tanh fit (R² > 0.5): {n_good_fit}/{n_total} ({n_good_fit/n_total*100:.0f}%)")
    
    print(f"\n  R² statistics:")
    print(f"    Tanh:   mean={df['r2_tanh'].mean():.3f}, median={df['r2_tanh'].median():.3f}")
    print(f"    Linear: mean={df['r2_linear'].mean():.3f}, median={df['r2_linear'].median():.3f}")
    
    # R_c scaling analysis
    valid_Rc = df[df['R_disk'].notna() & (df['r2_tanh'] > 0.3)]
    
    if len(valid_Rc) > 5:
        from scipy.stats import pearsonr
        
        corr, pval = pearsonr(valid_Rc['R_disk'], valid_Rc['R_c'])
        print(f"\n  R_c vs R_disk correlation:")
        print(f"    Pearson r = {corr:.3f} (p = {pval:.3e})")
        print(f"    N = {len(valid_Rc)} galaxies")
        
        # Linear fit R_c = a × R_disk + b
        slope = np.polyfit(valid_Rc['R_disk'], valid_Rc['R_c'], 1)[0]
        print(f"    R_c ≈ {slope:.2f} × R_disk")
        
        # MW comparison
        MW_R_disk = 2.5  # kpc (approximate)
        MW_R_c_predicted = slope * MW_R_disk
        print(f"\n  MW check: R_disk ≈ {MW_R_disk} kpc → R_c_predicted ≈ {MW_R_c_predicted:.1f} kpc")
        print(f"            Actual MW R_c = 6.75 kpc")
    
    # Best fits
    print("\n  TOP 10 TANH FITS:")
    top10 = df.nlargest(10, 'r2_tanh')
    for _, row in top10.iterrows():
        print(f"    {row['name']:15s}: R²={row['r2_tanh']:.3f}, R_c={row['R_c']:.2f}, A={row['A']:.2f}")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if output_file is None:
        output_file = str(output_dir / 'sparc_tanh_results.json')
    
    output = {
        'summary': {
            'n_galaxies': n_total,
            'tanh_better_frac': n_tanh_better / n_total,
            'good_fit_frac': n_good_fit / n_total,
            'mean_r2_tanh': float(df['r2_tanh'].mean()),
            'mean_r2_linear': float(df['r2_linear'].mean()),
        },
        'per_galaxy': df.to_dict('records'),
        'MW_comparison': {
            'MW_R_c': 6.75,
            'MW_A': 0.95,
            'MW_w': 1.78,
            'MW_c': 1.02,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to {output_file}")
    
    # Final conclusion
    print("\n" + "="*70)
    print("  CONCLUSION")
    print("="*70)
    
    if n_tanh_better / n_total > 0.6:
        print("\n  ✓ TANH TRANSITION IS UNIVERSAL")
        print("    The phase transition form fits most SPARC galaxies")
    elif n_tanh_better / n_total > 0.4:
        print("\n  ~ MIXED RESULTS")
        print("    Tanh works for some galaxies but not all")
    else:
        print("\n  ✗ TANH MAY BE MW-SPECIFIC")
        print("    Linear or other forms may be better for external galaxies")
    
    return df


if __name__ == '__main__':
    run_sparc_test()
