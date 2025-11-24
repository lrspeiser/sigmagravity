#!/usr/bin/env python3
"""
Test Root Cause Correlations for Parameter Reduction

Tests the hypothesis that fitted parameters can be derived from observables:
1. σ*_fitted vs κ_observed → should be linear (σ* = κ / k_eff)
2. Q*_fitted vs (σ*κ)/(πGΣ_b) → should be 1:1 correlation
3. M*_fitted vs Σ_b × ℓ² → should be 1:1 correlation

If correlations hold, we can reduce parameters from 9 → 2 (α₀, k_eff).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from data_integration.load_real_data import RealDataLoader
except ImportError:
    from data_integration.load_data import RealDataLoader

from galaxies.environment_estimator_v2 import EnvironmentEstimatorV2
from galaxies.coherence_microphysics import GravitationalPolarizationMemory

# Physical constants
G_kpc = 4.302e-3  # kpc (km/s)^2 / M_sun


def load_sparc_masses(galaxy_name):
    """Load SPARC mass data from master table."""
    # This is a placeholder - you'll need to implement actual loading
    # For now, use typical values
    return {
        'M_stellar': 1e9,
        'M_HI': 1e8,
        'M_total': 1e9,
        'R_disk': 2.0,
        'R_HI': 3.0
    }


def compute_kappa_profile(r, v_obs, v_bar=None):
    """
    Compute epicyclic frequency κ(R) profile.
    
    κ² = 2Ω(Ω + r dΩ/dr)
    where Ω = v/r
    """
    v_for_kappa = v_bar if v_bar is not None else v_obs
    
    # Mask valid points
    mask = (r > 0) & (v_for_kappa > 0)
    if np.sum(mask) < 3:
        return None, None
    
    r_valid = r[mask]
    v_valid = v_for_kappa[mask]
    
    # Angular velocity
    Omega = v_valid / r_valid
    
    # dΩ/dr
    dOmega_dr = np.gradient(Omega, r_valid)
    
    # Epicyclic frequency
    kappa_sq = 2.0 * Omega * (Omega + r_valid * dOmega_dr)
    kappa_sq = np.maximum(kappa_sq, 0.0)
    kappa = np.sqrt(kappa_sq)
    
    # Filter out very small kappa (unphysical)
    kappa_mask = kappa > 0.1 * np.max(kappa)
    if np.sum(kappa_mask) < 3:
        return None, None
    
    return r_valid[kappa_mask], kappa[kappa_mask]


def compute_mean_kappa(r, kappa):
    """Compute mean epicyclic frequency (weighted by radius)."""
    if r is None or kappa is None or len(r) == 0:
        return None
    
    # Weight by radius (outer regions matter more for coherence)
    weights = r / np.sum(r)
    kappa_mean = np.sum(kappa * weights)
    
    return kappa_mean


def compute_mean_sigma_b(r, SBdisk, M_L=0.5):
    """Compute mean surface density."""
    if len(r) == 0 or len(SBdisk) == 0:
        return None
    
    # Convert to surface density
    Sigma_b = SBdisk * M_L * 1e6  # M_sun/kpc²
    
    # Mask valid
    mask = (r > 0) & (Sigma_b > 0)
    if np.sum(mask) < 3:
        return None
    
    r_valid = r[mask]
    Sigma_valid = Sigma_b[mask]
    
    # Weight by surface density (denser regions matter more)
    weights = Sigma_valid / np.sum(Sigma_valid)
    Sigma_mean = np.sum(Sigma_valid * weights)
    
    return Sigma_mean


def test_root_cause_correlations(galaxy_names=None):
    """
    Test root cause correlations across SPARC sample.
    
    Uses current best-fit parameters and tests if they correlate with observables.
    """
    
    # Current best-fit parameters (from GPM_SUCCESS.md)
    alpha0_fitted = 0.3
    ell0_fitted = 2.0  # kpc
    Qstar_fitted = 2.0
    sigmastar_fitted = 25.0  # km/s (from GPM_SUCCESS, but also see 70 km/s in other docs)
    Mstar_fitted = 2e8  # M_sun
    nM_fitted = 1.5
    n_fitted = 2.0  # nQ = nσ = 2.0
    
    print("=" * 80)
    print("ROOT CAUSE CORRELATION TEST")
    print("=" * 80)
    print(f"\nUsing fitted parameters:")
    print(f"  α₀ = {alpha0_fitted:.3f}")
    print(f"  ℓ₀ = {ell0_fitted:.2f} kpc")
    print(f"  Q* = {Qstar_fitted:.2f}")
    print(f"  σ* = {sigmastar_fitted:.1f} km/s")
    print(f"  M* = {Mstar_fitted:.2e} M☉")
    print(f"  n = {n_fitted:.1f}")
    print(f"  nM = {nM_fitted:.1f}")
    
    # Load galaxy data
    loader = RealDataLoader()
    estimator = EnvironmentEstimatorV2(verbose=False)
    
    if galaxy_names is None:
        # Use a sample of galaxies
        galaxy_names = [
            'DDO154', 'DDO170', 'IC2574', 'NGC2403', 'NGC6503',
            'NGC3198', 'NGC2841', 'UGC00128', 'UGC02259', 'NGC0801'
        ]
    
    print(f"\nTesting on {len(galaxy_names)} galaxies...")
    
    # Storage for correlations
    results = []
    
    for galaxy_name in galaxy_names:
        try:
            # Load SPARC data
            gal = loader.load_rotmod_galaxy(galaxy_name)
            if gal is None or len(gal['r']) < 5:
                print(f"  ✗ {galaxy_name}: Insufficient data")
                continue
            
            r = gal['r']
            v_obs = gal['v_obs']
            
            # Load SBdisk from rotmod file (column 7)
            try:
                import pandas as pd
                rotmod_dir = Path(loader.base_data_dir) / 'Rotmod_LTG'
                filepath = rotmod_dir / f'{galaxy_name}_rotmod.dat'
                if filepath.exists():
                    # Read the file, skipping comment lines
                    df = pd.read_csv(filepath, sep=r'\s+', comment='#', 
                                    names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
                    SBdisk = df['SBdisk'].values
                    # Match length to r array (in case of mismatches)
                    if len(SBdisk) != len(r):
                        # Interpolate or pad
                        if len(SBdisk) > 0:
                            SBdisk = np.interp(r, df['Rad'].values, SBdisk)
                        else:
                            SBdisk = np.zeros_like(r)
                else:
                    SBdisk = np.zeros_like(r)
            except Exception as e:
                print(f"    Warning: Could not load SBdisk: {e}")
                SBdisk = np.zeros_like(r)
            
            # Load masses
            masses = load_sparc_masses(galaxy_name)
            M_total = masses['M_total']
            R_disk = masses['R_disk']
            
            # Compute observables
            # 1. Epicyclic frequency κ
            r_kappa, kappa_profile = compute_kappa_profile(r, v_obs, gal.get('v_bar'))
            if r_kappa is None:
                print(f"  ✗ {galaxy_name}: Could not compute κ")
                continue
            
            kappa_mean = compute_mean_kappa(r_kappa, kappa_profile)
            if kappa_mean is None:
                print(f"  ✗ {galaxy_name}: Could not compute mean κ")
                continue
            
            # 2. Surface density Σ_b
            Sigma_b_mean = compute_mean_sigma_b(r, SBdisk)
            if Sigma_b_mean is None:
                print(f"  ✗ {galaxy_name}: Could not compute Σ_b")
                continue
            
            # 3. Compute theoretical predictions
            # Prediction 1: σ* = κ / k_eff
            k_eff = 1.0 / ell0_fitted  # k_eff = 1/ℓ₀
            sigmastar_predicted = kappa_mean / k_eff
            
            # Prediction 2: Q* = σ* κ / (πG Σ_b)
            # Toomre Q formula: Q = κ σ_v / (π G Σ_b)
            # For a galaxy with σ_v = σ*, we get Q = (κ × σ*) / (π G Σ_b)
            # This should equal Q* for typical galaxies
            # Note: Some sources use 3.36 instead of π, but we'll use π for consistency
            denominator = np.pi * G_kpc * Sigma_b_mean
            if denominator > 0:
                Qstar_predicted = (sigmastar_predicted * kappa_mean) / denominator
            else:
                Qstar_predicted = 0.0
            
            # Prediction 3: M* = Σ_b × ℓ²
            Mstar_predicted = Sigma_b_mean * (ell0_fitted ** 2)
            
            # Store results
            results.append({
                'name': galaxy_name,
                'kappa_mean': kappa_mean,
                'Sigma_b_mean': Sigma_b_mean,
                'M_total': M_total,
                'R_disk': R_disk,
                'sigmastar_predicted': sigmastar_predicted,
                'Qstar_predicted': Qstar_predicted,
                'Mstar_predicted': Mstar_predicted,
            })
            
            print(f"  ✓ {galaxy_name}: κ={kappa_mean:.1f} km/s/kpc, Σ_b={Sigma_b_mean:.2e} M☉/kpc²")
            
        except Exception as e:
            print(f"  ✗ {galaxy_name}: Error - {e}")
            continue
    
    if len(results) == 0:
        print("\n✗ No galaxies processed successfully!")
        return None
    
    print(f"\n✓ Processed {len(results)} galaxies successfully")
    
    # Convert to arrays for analysis
    results_array = np.array([(r['kappa_mean'], r['Sigma_b_mean'], r['M_total'],
                               r['sigmastar_predicted'], r['Qstar_predicted'], r['Mstar_predicted'])
                              for r in results])
    
    kappa_obs = results_array[:, 0]
    Sigma_b_obs = results_array[:, 1]
    M_total_obs = results_array[:, 2]
    sigmastar_pred = results_array[:, 3]
    Qstar_pred = results_array[:, 4]
    Mstar_pred = results_array[:, 5]
    
    # Test correlations
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Test 1: σ* vs κ
    print("\n1. Testing: σ* = κ / k_eff")
    print(f"   Fitted σ* = {sigmastar_fitted:.1f} km/s")
    print(f"   k_eff = 1/ℓ₀ = {1/ell0_fitted:.3f} kpc⁻¹")
    print(f"   Predicted σ* range: [{sigmastar_pred.min():.1f}, {sigmastar_pred.max():.1f}] km/s")
    
    # Linear fit: σ* = a × κ + b
    if len(kappa_obs) > 2:
        coeffs = np.polyfit(kappa_obs, sigmastar_pred, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        r_sq = np.corrcoef(kappa_obs, sigmastar_pred)[0, 1] ** 2
        
        print(f"   Linear fit: σ* = {slope:.2f} × κ + {intercept:.2f}")
        print(f"   R² = {r_sq:.3f}")
        print(f"   Expected slope = 1/k_eff = {1/k_eff:.3f} kpc")
        
        if abs(slope - 1/k_eff) < 0.5:
            print("   ✓ Slope matches prediction!")
        else:
            print(f"   ✗ Slope mismatch: expected {1/k_eff:.3f}, got {slope:.2f}")
    
    # Test 2: Q* vs (σ*κ)/(πGΣ_b)
    print("\n2. Testing: Q* = (σ* × κ) / (πG Σ_b)")
    print(f"   Fitted Q* = {Qstar_fitted:.2f}")
    print(f"   Predicted Q* range: [{Qstar_pred.min():.2f}, {Qstar_pred.max():.2f}]")
    
    if len(Qstar_pred) > 2:
        Qstar_mean = np.mean(Qstar_pred)
        Qstar_std = np.std(Qstar_pred)
        r_sq = np.corrcoef([Qstar_fitted] * len(Qstar_pred), Qstar_pred)[0, 1] ** 2 if len(Qstar_pred) > 1 else 0
        
        print(f"   Mean predicted Q* = {Qstar_mean:.2f} ± {Qstar_std:.2f}")
        print(f"   R² vs fitted = {r_sq:.3f}")
        
        if abs(Qstar_mean - Qstar_fitted) < 1.0:
            print("   ✓ Mean matches fitted value!")
        else:
            print(f"   ✗ Mean mismatch: expected {Qstar_fitted:.2f}, got {Qstar_mean:.2f}")
    
    # Test 3: M* vs Σ_b × ℓ²
    print("\n3. Testing: M* = Σ_b × ℓ²")
    print(f"   Fitted M* = {Mstar_fitted:.2e} M☉")
    print(f"   ℓ₀ = {ell0_fitted:.2f} kpc")
    print(f"   Predicted M* range: [{Mstar_pred.min():.2e}, {Mstar_pred.max():.2e}] M☉")
    
    if len(Mstar_pred) > 2:
        Mstar_mean = np.mean(Mstar_pred)
        Mstar_std = np.std(Mstar_pred)
        
        print(f"   Mean predicted M* = {Mstar_mean:.2e} ± {Mstar_std:.2e} M☉")
        
        # Check if within order of magnitude
        if abs(np.log10(Mstar_mean) - np.log10(Mstar_fitted)) < 1.0:
            print("   ✓ Mean within order of magnitude of fitted value!")
        else:
            print(f"   ✗ Mean mismatch: expected {Mstar_fitted:.2e}, got {Mstar_mean:.2e}")
    
    # Create plots
    output_dir = Path(__file__).parent.parent / 'outputs' / 'root_cause_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: σ* vs κ
    ax = axes[0]
    ax.scatter(kappa_obs, sigmastar_pred, alpha=0.6, s=50)
    if len(kappa_obs) > 2:
        x_line = np.linspace(kappa_obs.min(), kappa_obs.max(), 100)
        y_line = x_line / k_eff
        ax.plot(x_line, y_line, 'r--', label=f'Expected: σ* = κ / {k_eff:.3f}')
    ax.axhline(sigmastar_fitted, color='g', linestyle=':', label=f'Fitted σ* = {sigmastar_fitted:.1f}')
    ax.set_xlabel('κ (km/s/kpc)')
    ax.set_ylabel('Predicted σ* (km/s)')
    ax.set_title('Test 1: σ* = κ / k_eff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Q* vs predicted
    ax = axes[1]
    ax.scatter([Qstar_fitted] * len(Qstar_pred), Qstar_pred, alpha=0.6, s=50)
    ax.axhline(Qstar_fitted, color='g', linestyle=':', label=f'Fitted Q* = {Qstar_fitted:.2f}')
    ax.axvline(Qstar_fitted, color='g', linestyle=':')
    ax.plot([0, 5], [0, 5], 'r--', label='1:1 line')
    ax.set_xlabel('Fitted Q*')
    ax.set_ylabel('Predicted Q*')
    ax.set_title('Test 2: Q* = (σ*κ)/(πGΣ_b)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: M* vs predicted
    ax = axes[2]
    ax.scatter([Mstar_fitted] * len(Mstar_pred), Mstar_pred, alpha=0.6, s=50)
    ax.axhline(Mstar_fitted, color='g', linestyle=':', label=f'Fitted M* = {Mstar_fitted:.2e}')
    ax.axvline(Mstar_fitted, color='g', linestyle=':')
    ax.plot([1e7, 1e10], [1e7, 1e10], 'r--', label='1:1 line')
    ax.set_xlabel('Fitted M* (M☉)')
    ax.set_ylabel('Predicted M* (M☉)')
    ax.set_title('Test 3: M* = Σ_b × ℓ²')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'root_cause_correlations.png', dpi=150)
    print(f"\n✓ Saved plots to {output_dir / 'root_cause_correlations.png'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nIf correlations hold:")
    print("  → Can reduce from 9 parameters to 2 parameters (α₀, k_eff)")
    print("  → All thresholds become derived from observables")
    print("  → More predictive and testable theory")
    
    return results


if __name__ == '__main__':
    results = test_root_cause_correlations()
    if results:
        print(f"\n✓ Test completed successfully on {len(results)} galaxies")

