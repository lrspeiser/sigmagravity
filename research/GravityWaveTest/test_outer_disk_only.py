"""
OUTER DISK ONLY TEST - Using ONLY real Gaia stars, NO analytical components.
Tests Σ-Gravity in disk-dominated region (R > 5 kpc) where we have actual data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy available - using GPU acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False

import sys
sys.path.insert(0, 'GravityWaveTest')
from test_star_by_star_mw import StarByStarCalculator
import json
import os

def test_outer_disk_only(R_min=5.0, R_max=15.0):
    """
    Pure validation test using ONLY real Gaia stars.
    NO analytical components, NO parameter tuning.
    
    Parameters:
    -----------
    R_min : float
        Minimum radius for test (default 5 kpc - beyond bulge)
    R_max : float  
        Maximum radius for test (default 15 kpc - where we have data)
    """
    
    print("="*80)
    print("OUTER DISK ONLY TEST (REAL DATA ONLY!)")
    print("="*80)
    print("\n✓ Using ONLY real Gaia stars")
    print("✓ NO analytical bulge")
    print("✓ NO parameter fitting")
    print(f"✓ Test region: {R_min} < R < {R_max} kpc")
    
    # Load REAL Gaia data
    print(f"\nLoading real Gaia data...")
    gaia = pd.read_csv('data/gaia/mw/gaia_mw_real.csv')
    
    # Select outer disk stars
    mask_outer = (gaia['R_kpc'] >= R_min) & (gaia['R_kpc'] <= R_max)
    stars_outer = gaia[mask_outer].copy()
    
    print(f"Selected {len(stars_outer):,} stars in outer disk ({R_min}-{R_max} kpc)")
    print(f"  Excluded {(gaia.R_kpc < R_min).sum():,} inner stars (R < {R_min} kpc)")
    print(f"  Excluded {(gaia.R_kpc > R_max).sum():,} outer stars (R > {R_max} kpc)")
    
    # Add required columns
    stars_outer['phi'] = np.random.uniform(0, 2*np.pi, len(stars_outer))
    stars_outer['M_star'] = 1.0
    stars_outer['pmra'] = 0.0
    stars_outer['pmdec'] = 0.0
    stars_outer['distance_pc'] = stars_outer['R_kpc'] * 1000
    
    # Rename columns
    stars_outer = stars_outer.rename(columns={'R_kpc': 'R_cyl', 'z_kpc': 'z'})
    
    # Observed rotation curve in this region
    R_bins = np.linspace(R_min, R_max, 20)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    v_obs_binned = []
    v_err_binned = []
    
    for i in range(len(R_bins)-1):
        mask_bin = (stars_outer['R_cyl'] >= R_bins[i]) & (stars_outer['R_cyl'] < R_bins[i+1])
        if mask_bin.sum() > 10:
            v_obs_binned.append(np.median(stars_outer.loc[mask_bin, 'vphi']))
            v_err_binned.append(np.std(stars_outer.loc[mask_bin, 'vphi']) / np.sqrt(mask_bin.sum()))
        else:
            v_obs_binned.append(np.nan)
            v_err_binned.append(np.nan)
    
    v_obs_binned = np.array(v_obs_binned)
    v_err_binned = np.array(v_err_binned)
    
    print(f"\nObserved rotation curve:")
    print(f"  Median v_phi: {np.nanmedian(v_obs_binned):.1f} km/s")
    print(f"  Range: {np.nanmin(v_obs_binned):.1f} - {np.nanmax(v_obs_binned):.1f} km/s")
    
    # Initialize calculator
    calc = StarByStarCalculator(stars_outer, use_gpu=GPU_AVAILABLE)
    
    # Test different λ hypotheses
    R_obs = R_centers[~np.isnan(v_obs_binned)]
    v_obs = v_obs_binned[~np.isnan(v_obs_binned)]
    
    # Estimate total disk mass from outer disk
    # v² ≈ GM/R → M ≈ v²R/G
    G_kpc = 4.30091e-6
    M_disk_estimate = np.median(v_obs**2 * R_obs / G_kpc)
    
    print(f"\nDisk mass estimate from outer rotation: {M_disk_estimate:.2e} M_☉")
    print("(This includes any Σ-Gravity enhancement already!)")
    
    # For pure test, use standard M_disk
    M_disk = 5.0e10
    A = 0.591
    
    hypotheses = {
        'universal': {
            'func': lambda: calc.compute_lambda_universal(4.993),
            'kwargs': {},
            'name': 'Universal λ = 4.993 kpc',
            'color': 'blue'
        },
        'h_R': {
            'func': lambda: calc.compute_lambda_local_disk(calc.R_stars),
            'kwargs': {},
            'name': 'λ = h(R) (local disk height)',
            'color': 'green'
        },
        'hybrid': {
            'func': lambda M_weights: calc.compute_lambda_hybrid(M_weights, calc.R_stars),
            'kwargs': {'M_weights': None},
            'name': 'λ ~ M^0.3 × R^0.3 (SPARC)',
            'color': 'orange'
        }
    }
    
    print("\n" + "="*80)
    print("TESTING PURE DISK MODELS (NO BULGE)")
    print("="*80)
    
    results = {}
    
    for hyp_name, hyp_data in hypotheses.items():
        print(f"\nTesting: {hyp_data['name']}")
        
        v_circ, lambda_stars = calc.test_hypothesis(
            R_obs, hyp_data['func'],
            A=A, M_disk=M_disk,
            **hyp_data['kwargs']
        )
        
        # Compute metrics against REAL observations
        residuals = v_circ - v_obs
        chi2 = np.sum(residuals**2 / v_err_binned[~np.isnan(v_obs_binned)]**2) / len(v_obs)
        rms = np.sqrt(np.mean(residuals**2))
        
        results[hyp_name] = {
            'R': R_obs,
            'v_pred': v_circ,
            'v_obs': v_obs,
            'residuals': residuals,
            'chi2': chi2,
            'rms': rms,
            'lambda_median': np.median(lambda_stars),
            'lambda_std': np.std(lambda_stars)
        }
        
        print(f"  λ median: {results[hyp_name]['lambda_median']:.2f} ± {results[hyp_name]['lambda_std']:.2f} kpc")
        print(f"  RMS residual: {rms:.1f} km/s")
        print(f"  χ²/dof: {chi2:.2f}")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Outer Disk Test: {len(stars_outer):,} Real Gaia Stars, R={R_min}-{R_max} kpc', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Rotation curves
    ax = axes[0, 0]
    
    # Plot observed with error bars
    ax.errorbar(R_obs, v_obs, yerr=v_err_binned[~np.isnan(v_obs_binned)], 
                fmt='ko', capsize=3, markersize=6, label='Gaia observations', zorder=10)
    
    # Plot models
    for hyp_name, hyp_data in hypotheses.items():
        ax.plot(results[hyp_name]['R'], results[hyp_name]['v_pred'],
                label=hyp_data['name'], color=hyp_data['color'], linewidth=2)
    
    ax.axhline(220, color='gray', linestyle=':', alpha=0.5, label='MW expected (220 km/s)')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Rotation Curves (Real Data)')
    ax.set_ylim(50, 350)
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    for hyp_name, hyp_data in hypotheses.items():
        ax.plot(results[hyp_name]['R'], results[hyp_name]['residuals'],
                label=hyp_data['name'], color=hyp_data['color'], linewidth=2, marker='o')
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Δv [km/s]', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Residuals vs Observations')
    
    # Plot 3: χ² comparison
    ax = axes[1, 0]
    hyp_names = list(hypotheses.keys())
    chi2_values = [results[h]['chi2'] for h in hyp_names]
    rms_values = [results[h]['rms'] for h in hyp_names]
    colors = [hypotheses[h]['color'] for h in hyp_names]
    
    x = np.arange(len(hyp_names))
    ax.bar(x, chi2_values, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([hypotheses[h]['name'] for h in hyp_names],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('χ² / dof', fontsize=12)
    ax.set_title('Goodness of Fit (Real Data)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Stellar distribution in test region
    ax = axes[1, 1]
    ax.hist(stars_outer['R_cyl'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Star count', fontsize=12)
    ax.set_title(f'Real Gaia Distribution ({len(stars_outer):,} stars)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = "GravityWaveTest/outer_disk_validation"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/outer_disk_pure_validation.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_dir}/outer_disk_pure_validation.png")
    plt.close()
    
    # Save results
    results_export = {
        'test_type': 'OUTER_DISK_ONLY_PURE_VALIDATION',
        'n_stars': int(len(stars_outer)),
        'R_range': [float(R_min), float(R_max)],
        'M_disk': float(M_disk),
        'A': float(A),
        'note': 'Uses ONLY real Gaia stars, NO analytical components',
        'R_obs': R_obs.tolist(),
        'v_obs': v_obs.tolist(),
        'hypotheses': {}
    }
    
    for hyp_name in hypotheses:
        results_export['hypotheses'][hyp_name] = {
            'name': hypotheses[hyp_name]['name'],
            'v_pred': results[hyp_name]['v_pred'].tolist(),
            'lambda_median': float(results[hyp_name]['lambda_median']),
            'lambda_std': float(results[hyp_name]['lambda_std']),
            'chi2': float(results[hyp_name]['chi2']),
            'rms': float(results[hyp_name]['rms'])
        }
    
    with open(f"{output_dir}/pure_validation_results.json", 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/pure_validation_results.json")
    
    # Summary
    print("\n" + "="*80)
    print("PURE VALIDATION RESULTS (REAL DATA ONLY)")
    print("="*80)
    
    best_hyp = min(hypotheses.keys(), key=lambda h: results[h]['chi2'])
    
    print(f"\nBest fit: {hypotheses[best_hyp]['name']}")
    print(f"  χ²/dof: {results[best_hyp]['chi2']:.2f}")
    print(f"  RMS: {results[best_hyp]['rms']:.1f} km/s")
    
    print("\nAll results:")
    for hyp_name in sorted(hypotheses.keys(), key=lambda h: results[h]['chi2']):
        print(f"  {hypotheses[hyp_name]['name']}")
        print(f"    χ²={results[hyp_name]['chi2']:.2f}, RMS={results[hyp_name]['rms']:.1f} km/s")
    
    print("\n" + "="*80)
    print("CRITICAL NOTE:")
    print("="*80)
    print("\nThis test uses ONLY real Gaia stars - no analytical components!")
    print("We're testing the DISK component of Σ-Gravity in isolation.")
    print("Inner regions (R < 5 kpc) excluded due to lack of bulge stars.")
    print("\nFor full MW model, bulge must be added (analytically or from IR data).")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CHOOSE TEST REGION")
    print("="*80)
    print("\nOptions:")
    print("  1. R > 5 kpc (disk-dominated, 143k stars)")
    print("  2. R > 8 kpc (pure outer disk, ~93k stars)")
    print("  3. R > 10 kpc (far outer disk, ~3k stars)")
    
    choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip()
    
    if choice == '2':
        R_min = 8.0
    elif choice == '3':
        R_min = 10.0
    else:
        R_min = 5.0
    
    results = test_outer_disk_only(R_min=R_min, R_max=15.0)

