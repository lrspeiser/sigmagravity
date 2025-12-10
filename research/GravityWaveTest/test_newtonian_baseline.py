"""
Step 1: Test NEWTONIAN baseline (A=0) to verify basic physics.
Uses original Gaia data with proper velocities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'GravityWaveTest')
from test_star_by_star_mw import StarByStarCalculator, G_KPC
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

def load_original_gaia_with_proper_velocities():
    """
    Load ORIGINAL Gaia data with properly computed velocities.
    """
    
    print("="*80)
    print("LOADING ORIGINAL GAIA DATA (PROPER VELOCITIES)")
    print("="*80)
    
    gaia_orig = pd.read_csv('data/gaia/mw/gaia_mw_real.csv')
    
    print(f"\nLoaded {len(gaia_orig):,} stars")
    print(f"  R range: {gaia_orig['R_kpc'].min():.2f} - {gaia_orig['R_kpc'].max():.2f} kpc")
    print(f"  z range: {gaia_orig['z_kpc'].min():.2f} - {gaia_orig['z_kpc'].max():.2f} kpc")
    print(f"\nVelocities (properly transformed):")
    print(f"  v_phi median: {gaia_orig['vphi'].median():.1f} km/s")
    print(f"  v_phi range: {gaia_orig['vphi'].min():.1f} - {gaia_orig['vphi'].max():.1f} km/s")
    
    # Convert to format needed for calculator
    # Add phi (azimuthal angle) - random for axisymmetric disk
    gaia_orig['phi'] = np.random.uniform(0, 2*np.pi, len(gaia_orig))
    gaia_orig['M_star'] = 1.0  # Placeholder
    gaia_orig['pmra'] = 0.0
    gaia_orig['pmdec'] = 0.0
    gaia_orig['distance_pc'] = gaia_orig['R_kpc'] * 1000
    
    # Rename columns
    gaia_for_calc = gaia_orig.rename(columns={'R_kpc': 'R_cyl', 'z_kpc': 'z', 'vR': 'v_rad'})
    
    return gaia_for_calc, gaia_orig

def get_observed_rotation_curve(gaia_orig):
    """
    Extract observed rotation curve from Gaia v_phi.
    """
    
    R_bins = np.linspace(4, 16, 25)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    
    v_median = []
    v_err = []
    n_stars = []
    
    for i in range(len(R_bins)-1):
        mask = (gaia_orig['R_kpc'] >= R_bins[i]) & (gaia_orig['R_kpc'] < R_bins[i+1])
        
        if mask.sum() > 10:
            v_median.append(np.median(gaia_orig.loc[mask, 'vphi']))
            v_err.append(np.std(gaia_orig.loc[mask, 'vphi']) / np.sqrt(mask.sum()))
            n_stars.append(mask.sum())
        else:
            v_median.append(np.nan)
            v_err.append(np.nan)
            n_stars.append(0)
    
    # Remove NaN
    mask_valid = ~np.isnan(v_median)
    
    return R_centers[mask_valid], np.array(v_median)[mask_valid], np.array(v_err)[mask_valid]

def test_newtonian_baseline():
    """
    Test pure Newtonian prediction (A=0) as sanity check.
    """
    
    print("\n" + "="*80)
    print("STEP 1: NEWTONIAN BASELINE TEST (A=0)")
    print("="*80)
    print("\nThis tests basic physics WITHOUT Σ-Gravity enhancement")
    print("Expected result: v ~ 210 km/s at R=8.2 kpc (baryons only)")
    
    # Load data
    gaia_for_calc, gaia_orig = load_original_gaia_with_proper_velocities()
    
    # Get observed curve
    R_obs, v_obs, v_err = get_observed_rotation_curve(gaia_orig)
    
    print(f"\nObserved rotation curve (from Gaia v_phi):")
    print(f"  Median: {np.median(v_obs):.1f} km/s")
    print(f"  At R=8.2 kpc: {np.interp(8.2, R_obs, v_obs):.1f} ± {np.interp(8.2, R_obs, v_err):.1f} km/s")
    print(f"  Range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
    
    # Initialize calculator
    calc = StarByStarCalculator(gaia_for_calc, use_gpu=GPU_AVAILABLE)
    
    # Test Newtonian (A=0)
    print("\n" + "="*80)
    print("COMPUTING NEWTONIAN PREDICTION (A=0)")
    print("="*80)
    
    # Use universal lambda (doesn't matter since A=0)
    lambda_func = lambda: calc.compute_lambda_universal(4.993)
    
    v_newtonian, _ = calc.test_hypothesis(
        R_obs,
        lambda_func,
        A=0.0,  # NO ENHANCEMENT
        M_disk=5.0e10
    )
    
    print(f"\nNewtonian predictions (A=0):")
    print(f"  v @ R=8.2 kpc: {np.interp(8.2, R_obs, v_newtonian):.1f} km/s")
    print(f"  vs observed: {np.interp(8.2, R_obs, v_obs):.1f} km/s")
    print(f"  Deficit: {np.interp(8.2, R_obs, v_obs) - np.interp(8.2, R_obs, v_newtonian):.1f} km/s")
    
    # Test with Σ-Gravity (A=0.591)
    print("\n" + "="*80)
    print("COMPUTING WITH Σ-GRAVITY (A=0.591)")
    print("="*80)
    
    v_sigma_gravity, _ = calc.test_hypothesis(
        R_obs,
        lambda_func,
        A=0.591,  # With enhancement
        M_disk=5.0e10
    )
    
    print(f"\nΣ-Gravity predictions (A=0.591):")
    print(f"  v @ R=8.2 kpc: {np.interp(8.2, R_obs, v_sigma_gravity):.1f} km/s")
    print(f"  vs observed: {np.interp(8.2, R_obs, v_obs):.1f} km/s")
    print(f"  Boost from Newtonian: {np.interp(8.2, R_obs, v_sigma_gravity) / np.interp(8.2, R_obs, v_newtonian):.2f}×")
    
    # Compute RMS
    rms_newt = np.sqrt(np.mean((v_newtonian - v_obs)**2))
    rms_sigma = np.sqrt(np.mean((v_sigma_gravity - v_obs)**2))
    
    print(f"\nRMS residuals:")
    print(f"  Newtonian (A=0): {rms_newt:.1f} km/s")
    print(f"  Σ-Gravity (A=0.591): {rms_sigma:.1f} km/s")
    print(f"  Improvement: {(rms_newt - rms_sigma):.1f} km/s ({100*(rms_newt-rms_sigma)/rms_newt:.1f}%)")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Newtonian Baseline Test (Original Gaia, Proper Velocities)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Rotation curves
    ax = axes[0]
    ax.errorbar(R_obs, v_obs, yerr=v_err, fmt='ko', capsize=3, markersize=6,
                label='Observed (Gaia v_phi)', zorder=10)
    ax.plot(R_obs, v_newtonian, 'b-', linewidth=2, label='Newtonian (A=0)')
    ax.plot(R_obs, v_sigma_gravity, 'r-', linewidth=2, label='Σ-Gravity (A=0.591)')
    ax.axhline(220, color='gray', linestyle=':', alpha=0.5, label='MW expected')
    ax.axvline(8.2, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Rotation Curves')
    ax.set_ylim(0, 400)
    
    # Plot 2: Residuals
    ax = axes[1]
    ax.plot(R_obs, v_newtonian - v_obs, 'b-', linewidth=2, marker='o', label='Newtonian')
    ax.plot(R_obs, v_sigma_gravity - v_obs, 'r-', linewidth=2, marker='o', label='Σ-Gravity')
    ax.axhline(0, color='k', linestyle='--')
    ax.fill_between(R_obs, -v_err, v_err, alpha=0.2, color='gray')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_pred - v_obs [km/s]', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Residuals')
    
    plt.tight_layout()
    plt.savefig('GravityWaveTest/newtonian_baseline_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to GravityWaveTest/newtonian_baseline_test.png")
    plt.close()
    
    return {
        'R': R_obs,
        'v_obs': v_obs,
        'v_err': v_err,
        'v_newtonian': v_newtonian,
        'v_sigma_gravity': v_sigma_gravity,
        'rms_newtonian': rms_newt,
        'rms_sigma_gravity': rms_sigma
    }

if __name__ == "__main__":
    results = test_newtonian_baseline()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("\nIf Newtonian gives v~210 km/s: ✓ Physics is correct")
    print("If Newtonian gives v>>300 km/s: ✗ Mass or integration error")
    print("\nIf Σ-Gravity improves on Newtonian: ✓ Enhancement works")
    print("If Σ-Gravity makes it worse: ✗ Enhancement application error")

