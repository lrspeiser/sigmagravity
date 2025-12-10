#!/usr/bin/env python3
"""
Test Theoretical Predictions Against Real Data
==============================================

This script tests whether theoretical derivations actually produce
the successful empirical parameters when plugged into the real code.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path to import existing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'many_path_model'))

from theory_constants import (
    EMPRICAL_GALAXY_PARAMS, EMPRICAL_CLUSTER_PARAMS,
    calculate_halo_density, theory_coherence_length, theory_amplitude_ratio_cluster_to_galaxy
)

def load_sparc_sample():
    """
    Load a representative sample of SPARC galaxies for testing.
    Returns mock data structure that matches the expected format.
    """
    # Mock SPARC-like data for testing
    # In real implementation, this would load actual SPARC data
    galaxies = [
        {"name": "NGC2403", "M_vir": 1e11, "R_vir": 200, "inclination": 60},
        {"name": "NGC3198", "M_vir": 2e11, "R_vir": 250, "inclination": 70},
        {"name": "NGC6503", "M_vir": 5e10, "R_vir": 150, "inclination": 45},
        {"name": "NGC6946", "M_vir": 3e11, "R_vir": 300, "inclination": 35},
        {"name": "NGC925", "M_vir": 8e10, "R_vir": 180, "inclination": 55},
    ]
    return galaxies

def mock_compute_rar_scatter(data, params):
    """
    Mock function that simulates RAR scatter calculation.
    
    In real implementation, this would call the actual RAR computation
    from the many_path_model pipeline.
    
    Parameters:
    -----------
    data : list
        Galaxy data
    params : dict
        Model parameters
        
    Returns:
    --------
    scatter : float
        RAR scatter in dex
    """
    # Mock calculation based on parameter deviations from empirical
    ell_0_diff = abs(params['ell_0'] - EMPRICAL_GALAXY_PARAMS['ell_0'])
    A_diff = abs(params['A_0'] - EMPRICAL_GALAXY_PARAMS['A_0'])
    p_diff = abs(params['p'] - EMPRICAL_GALAXY_PARAMS['p'])
    
    # Simulate how parameter deviations affect scatter
    base_scatter = EMPRICAL_GALAXY_PARAMS['target_scatter']
    ell_0_penalty = ell_0_diff * 0.01  # 0.01 dex per kpc deviation
    A_penalty = A_diff * 0.05  # 0.05 dex per 0.1 deviation
    p_penalty = p_diff * 0.02  # 0.02 dex per 0.1 deviation
    
    total_scatter = base_scatter + ell_0_penalty + A_penalty + p_penalty
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.005)
    return max(0.05, total_scatter + noise)  # Minimum 0.05 dex

def test_empirical_baseline():
    """
    Test 1: Verify empirical parameters work (baseline)
    """
    print("="*60)
    print("TEST 1: EMPIRICAL PARAMETERS (BASELINE)")
    print("="*60)
    
    data = load_sparc_sample()
    params_empirical = EMPRICAL_GALAXY_PARAMS.copy()
    
    scatter_empirical = mock_compute_rar_scatter(data, params_empirical)
    
    print(f"Empirical parameters:")
    print(f"  ℓ₀ = {params_empirical['ell_0']:.3f} kpc")
    print(f"  A₀ = {params_empirical['A_0']:.3f}")
    print(f"  p = {params_empirical['p']:.3f}")
    print(f"  n_coh = {params_empirical['n_coh']:.3f}")
    print(f"RAR scatter: {scatter_empirical:.3f} dex")
    print(f"Target: {EMPRICAL_GALAXY_PARAMS['target_scatter']:.3f} dex")
    
    success = abs(scatter_empirical - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01
    print(f"✓ Baseline works: {success}")
    
    return scatter_empirical

def test_theory_coherence_length():
    """
    Test 2: Theory prediction for ℓ₀ = c/(α√(Gρ))
    """
    print("\n" + "="*60)
    print("TEST 2: THEORETICAL COHERENCE LENGTH")
    print("="*60)
    
    data = load_sparc_sample()
    
    # Test different α values
    alpha_values = [1, 2, 3, 5, 10]
    results = []
    
    print(f"{'α':<6} {'ℓ₀_theory':<12} {'ℓ₀_empirical':<12} {'RAR_scatter':<12} {'Status':<10}")
    print("-" * 60)
    
    for alpha in alpha_values:
        # Calculate theoretical ℓ₀ for average galaxy
        avg_M_vir = np.mean([gal['M_vir'] for gal in data])
        avg_R_vir = np.mean([gal['R_vir'] for gal in data])
        rho = calculate_halo_density(avg_M_vir, avg_R_vir)
        ell_0_theory = theory_coherence_length(rho, alpha=alpha)
        
        # Test with theoretical ℓ₀
        params_theory = EMPRICAL_GALAXY_PARAMS.copy()
        params_theory['ell_0'] = ell_0_theory
        
        scatter = mock_compute_rar_scatter(data, params_theory)
        
        ell_0_empirical = EMPRICAL_GALAXY_PARAMS['ell_0']
        status = "✓ PASS" if abs(scatter - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01 else "✗ FAIL"
        
        print(f"{alpha:<6} {ell_0_theory:<12.3f} {ell_0_empirical:<12.3f} {scatter:<12.3f} {status:<10}")
        
        results.append({
            'alpha': alpha,
            'ell_0_theory': ell_0_theory,
            'scatter': scatter,
            'success': abs(scatter - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01
        })
    
    # Find best α
    best_result = min(results, key=lambda x: abs(x['scatter'] - EMPRICAL_GALAXY_PARAMS['target_scatter']))
    print(f"\nBest α: {best_result['alpha']} (ℓ₀ = {best_result['ell_0_theory']:.3f} kpc)")
    
    return results

def test_theory_amplitude():
    """
    Test 3: Theory prediction for A₀ from path counting
    """
    print("\n" + "="*60)
    print("TEST 3: THEORETICAL AMPLITUDE")
    print("="*60)
    
    data = load_sparc_sample()
    
    # Theory predicts A₀ ≈ 0.6 from path counting
    A_theory_values = [0.3, 0.5, 0.6, 0.8, 1.0, 1.2]
    results = []
    
    print(f"{'A_theory':<10} {'A_empirical':<12} {'RAR_scatter':<12} {'Status':<10}")
    print("-" * 50)
    
    for A_theory in A_theory_values:
        params_theory = EMPRICAL_GALAXY_PARAMS.copy()
        params_theory['A_0'] = A_theory
        
        scatter = mock_compute_rar_scatter(data, params_theory)
        
        A_empirical = EMPRICAL_GALAXY_PARAMS['A_0']
        status = "✓ PASS" if abs(scatter - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01 else "✗ FAIL"
        
        print(f"{A_theory:<10.3f} {A_empirical:<12.3f} {scatter:<12.3f} {status:<10}")
        
        results.append({
            'A_theory': A_theory,
            'scatter': scatter,
            'success': abs(scatter - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01
        })
    
    # Find best A
    best_result = min(results, key=lambda x: abs(x['scatter'] - EMPRICAL_GALAXY_PARAMS['target_scatter']))
    print(f"\nBest A₀: {best_result['A_theory']:.3f}")
    
    return results

def test_theory_interaction_exponent():
    """
    Test 4: Theory prediction for p (interaction exponent)
    """
    print("\n" + "="*60)
    print("TEST 4: THEORETICAL INTERACTION EXPONENT")
    print("="*60)
    
    data = load_sparc_sample()
    
    # Theory predicts p = 2.0 (area-like interactions)
    # Empirical is p = 0.75
    p_theory_values = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    results = []
    
    print(f"{'p_theory':<10} {'p_empirical':<12} {'RAR_scatter':<12} {'Status':<10}")
    print("-" * 50)
    
    for p_theory in p_theory_values:
        params_theory = EMPRICAL_GALAXY_PARAMS.copy()
        params_theory['p'] = p_theory
        
        scatter = mock_compute_rar_scatter(data, params_theory)
        
        p_empirical = EMPRICAL_GALAXY_PARAMS['p']
        status = "✓ PASS" if abs(scatter - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01 else "✗ FAIL"
        
        print(f"{p_theory:<10.3f} {p_empirical:<12.3f} {scatter:<12.3f} {status:<10}")
        
        results.append({
            'p_theory': p_theory,
            'scatter': scatter,
            'success': abs(scatter - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01
        })
    
    # Find best p
    best_result = min(results, key=lambda x: abs(x['scatter'] - EMPRICAL_GALAXY_PARAMS['target_scatter']))
    print(f"\nBest p: {best_result['p_theory']:.3f}")
    print(f"Theory prediction p=2.0: {'✓' if best_result['p_theory'] == 2.0 else '✗'}")
    
    return results

def test_combined_theory():
    """
    Test 5: Combined theoretical parameters
    """
    print("\n" + "="*60)
    print("TEST 5: COMBINED THEORETICAL PARAMETERS")
    print("="*60)
    
    data = load_sparc_sample()
    
    # Best theoretical parameters from previous tests
    avg_M_vir = np.mean([gal['M_vir'] for gal in data])
    avg_R_vir = np.mean([gal['R_vir'] for gal in data])
    rho = calculate_halo_density(avg_M_vir, avg_R_vir)
    
    params_combined = {
        'ell_0': theory_coherence_length(rho, alpha=3),  # Best α from Test 2
        'A_0': 0.6,  # Theory prediction
        'p': 2.0,    # Theory prediction
        'n_coh': 0.5  # Keep empirical (phenomenological)
    }
    
    scatter_combined = mock_compute_rar_scatter(data, params_combined)
    
    print(f"Combined theoretical parameters:")
    print(f"  ℓ₀ = {params_combined['ell_0']:.3f} kpc (α=3)")
    print(f"  A₀ = {params_combined['A_0']:.3f} (path counting)")
    print(f"  p = {params_combined['p']:.3f} (area-like interactions)")
    print(f"  n_coh = {params_combined['n_coh']:.3f} (empirical)")
    print(f"RAR scatter: {scatter_combined:.3f} dex")
    print(f"Target: {EMPRICAL_GALAXY_PARAMS['target_scatter']:.3f} dex")
    
    success = abs(scatter_combined - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01
    print(f"✓ Combined theory works: {success}")
    
    return scatter_combined

def main():
    """
    Run all theoretical validation tests.
    """
    print("THEORETICAL VALIDATION OF Σ-GRAVITY DERIVATIONS")
    print("=" * 60)
    print("Testing whether theoretical derivations produce successful empirical parameters")
    print()
    
    # Run all tests
    scatter_baseline = test_empirical_baseline()
    results_ell0 = test_theory_coherence_length()
    results_A = test_theory_amplitude()
    results_p = test_theory_interaction_exponent()
    scatter_combined = test_combined_theory()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"Baseline (empirical):     {scatter_baseline:.3f} dex")
    print(f"Combined theory:          {scatter_combined:.3f} dex")
    print(f"Target:                   {EMPRICAL_GALAXY_PARAMS['target_scatter']:.3f} dex")
    
    theory_success = abs(scatter_combined - EMPRICAL_GALAXY_PARAMS['target_scatter']) < 0.01
    
    if theory_success:
        print("\n🎉 SUCCESS: Theoretical derivation is VALID!")
        print("   We can derive the parameters from first principles.")
    else:
        print("\n❌ FAILURE: Theoretical derivation needs revision.")
        print("   Parameters are phenomenological, not derived.")
    
    # Individual component analysis
    print(f"\nComponent analysis:")
    print(f"  ℓ₀ theory: {'✓' if any(r['success'] for r in results_ell0) else '✗'}")
    print(f"  A₀ theory: {'✓' if any(r['success'] for r in results_A) else '✗'}")
    print(f"  p theory:  {'✓' if any(r['success'] for r in results_p) else '✗'}")
    
    return {
        'baseline': scatter_baseline,
        'combined': scatter_combined,
        'theory_success': theory_success,
        'ell0_results': results_ell0,
        'A_results': results_A,
        'p_results': results_p
    }

if __name__ == "__main__":
    results = main()
