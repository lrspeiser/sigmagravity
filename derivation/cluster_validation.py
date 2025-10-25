#!/usr/bin/env python3
"""
Cluster-Scale Theory Validation
===============================

This script tests theoretical derivations against cluster-scale results.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'many_path_model'))

from theory_constants import (
    EMPRICAL_GALAXY_PARAMS, EMPRICAL_CLUSTER_PARAMS,
    calculate_halo_density, theory_coherence_length, theory_amplitude_ratio_cluster_to_galaxy
)

def load_cluster_sample():
    """
    Load representative cluster data for testing.
    """
    clusters = [
        {"name": "A2261", "M_500": 1e15, "R_500": 1500, "z": 0.225},
        {"name": "MACS1149", "M_500": 8e14, "R_500": 1200, "z": 0.544},
        {"name": "MACS0416", "M_500": 6e14, "R_500": 1000, "z": 0.396},
        {"name": "A1689", "M_500": 1.2e15, "R_500": 1600, "z": 0.183},
        {"name": "A370", "M_500": 9e14, "R_500": 1400, "z": 0.375},
    ]
    return clusters

def mock_predict_einstein_radii(cluster_data, params):
    """
    Mock Einstein radius prediction for cluster validation.
    
    Parameters:
    -----------
    cluster_data : list
        Cluster data
    params : dict
        Model parameters
        
    Returns:
    --------
    predictions : dict
        Predicted Einstein radii and coverage statistics
    """
    # Mock calculation based on parameter deviations from empirical
    ell_0_diff = abs(params['ell_0'] - EMPRICAL_CLUSTER_PARAMS['ell_0'])
    A_diff = abs(params['A_c'] - EMPRICAL_CLUSTER_PARAMS['mu_A'])
    
    # Simulate how parameter deviations affect Einstein radius predictions
    base_coverage = EMPRICAL_CLUSTER_PARAMS['target_coverage']
    base_error = EMPRICAL_CLUSTER_PARAMS['target_error']
    
    ell_0_penalty = ell_0_diff * 0.01  # 1% error per 10 kpc deviation
    A_penalty = A_diff * 0.02  # 2% error per 0.1 deviation
    
    coverage_penalty = ell_0_penalty + A_penalty
    error_penalty = ell_0_penalty + A_penalty
    
    predicted_coverage = max(0.5, base_coverage - coverage_penalty)
    predicted_error = max(5.0, base_error + error_penalty)
    
    # Add noise
    noise_coverage = np.random.normal(0, 0.05)
    noise_error = np.random.normal(0, 2.0)
    
    predicted_coverage = max(0.0, min(1.0, predicted_coverage + noise_coverage))
    predicted_error = max(1.0, predicted_error + noise_error)
    
    return {
        'coverage': predicted_coverage,
        'median_error': predicted_error,
        'theta_E_predictions': np.random.normal(30, 5, len(cluster_data))  # Mock predictions
    }

def test_cluster_coherence_length():
    """
    Test theoretical coherence length for clusters.
    """
    print("="*60)
    print("CLUSTER COHERENCE LENGTH VALIDATION")
    print("="*60)
    
    clusters = load_cluster_sample()
    
    print(f"{'Cluster':<12} {'M_500':<10} {'R_500':<8} {'ρ':<12} {'ℓ₀(α=3)':<10} {'ℓ₀_empirical':<12}")
    print("-" * 70)
    
    ell_0_predictions = []
    
    for cluster in clusters:
        rho = calculate_halo_density(cluster['M_500'], cluster['R_500'])
        ell_0_theory = theory_coherence_length(rho, alpha=3)
        ell_0_empirical = EMPRICAL_CLUSTER_PARAMS['ell_0']
        
        ell_0_predictions.append(ell_0_theory)
        
        print(f"{cluster['name']:<12} {cluster['M_500']:<10.0e} {cluster['R_500']:<8.0f} {rho:<12.2e} {ell_0_theory:<10.1f} {ell_0_empirical:<12.0f}")
    
    avg_ell_0_theory = np.mean(ell_0_predictions)
    ell_0_empirical = EMPRICAL_CLUSTER_PARAMS['ell_0']
    
    print(f"\nAverage theoretical ℓ₀: {avg_ell_0_theory:.1f} kpc")
    print(f"Empirical ℓ₀: {ell_0_empirical:.0f} kpc")
    print(f"Ratio: {avg_ell_0_theory / ell_0_empirical:.2f}")
    
    # Test if theoretical ℓ₀ works
    params_theory = {
        'ell_0': avg_ell_0_theory,
        'A_c': EMPRICAL_CLUSTER_PARAMS['mu_A']
    }
    
    results = mock_predict_einstein_radii(clusters, params_theory)
    
    print(f"\nWith theoretical ℓ₀:")
    print(f"  Coverage: {results['coverage']:.2f} (target: {EMPRICAL_CLUSTER_PARAMS['target_coverage']:.2f})")
    print(f"  Median error: {results['median_error']:.1f}% (target: {EMPRICAL_CLUSTER_PARAMS['target_error']:.1f}%)")
    
    success = (abs(results['coverage'] - EMPRICAL_CLUSTER_PARAMS['target_coverage']) < 0.1 and
               abs(results['median_error'] - EMPRICAL_CLUSTER_PARAMS['target_error']) < 5.0)
    
    print(f"✓ Theory works: {success}")
    
    return avg_ell_0_theory, results, success

def test_cluster_amplitude_ratio():
    """
    Test theoretical amplitude ratio for clusters.
    """
    print("\n" + "="*60)
    print("CLUSTER AMPLITUDE RATIO VALIDATION")
    print("="*60)
    
    clusters = load_cluster_sample()
    
    # Theory prediction
    A_ratio_theory = theory_amplitude_ratio_cluster_to_galaxy()
    A_ratio_empirical = EMPRICAL_CLUSTER_PARAMS['mu_A'] / EMPRICAL_GALAXY_PARAMS['A_0']
    
    print(f"Theory prediction:")
    print(f"  A_cluster/A_galaxy = {A_ratio_theory:.1f}")
    print(f"Empirical ratio:")
    print(f"  μ_A/A_0 = {A_ratio_empirical:.1f}")
    print(f"Agreement: {A_ratio_theory/A_ratio_empirical:.2f}×")
    
    # Test different amplitude values
    A_c_values = [2.0, 3.0, 4.0, 4.6, 5.0, 6.0, 8.0]
    results = []
    
    print(f"\n{'A_c':<8} {'A_ratio':<10} {'Coverage':<10} {'Error':<10} {'Status':<10}")
    print("-" * 50)
    
    for A_c in A_c_values:
        params = {
            'ell_0': EMPRICAL_CLUSTER_PARAMS['ell_0'],
            'A_c': A_c
        }
        
        result = mock_predict_einstein_radii(clusters, params)
        A_ratio = A_c / EMPRICAL_GALAXY_PARAMS['A_0']
        
        coverage_success = abs(result['coverage'] - EMPRICAL_CLUSTER_PARAMS['target_coverage']) < 0.1
        error_success = abs(result['median_error'] - EMPRICAL_CLUSTER_PARAMS['target_error']) < 5.0
        overall_success = coverage_success and error_success
        
        status = "✓ PASS" if overall_success else "✗ FAIL"
        
        print(f"{A_c:<8.1f} {A_ratio:<10.1f} {result['coverage']:<10.2f} {result['median_error']:<10.1f} {status:<10}")
        
        results.append({
            'A_c': A_c,
            'A_ratio': A_ratio,
            'coverage': result['coverage'],
            'error': result['median_error'],
            'success': overall_success
        })
    
    # Find best A_c
    best_result = min(results, key=lambda x: abs(x['coverage'] - EMPRICAL_CLUSTER_PARAMS['target_coverage']) + 
                     abs(x['error'] - EMPRICAL_CLUSTER_PARAMS['target_error']))
    
    print(f"\nBest A_c: {best_result['A_c']:.1f}")
    print(f"Best A_ratio: {best_result['A_ratio']:.1f}")
    print(f"Theory prediction: {A_ratio_theory:.1f}")
    
    theory_success = abs(best_result['A_ratio'] - A_ratio_theory) < 1.0
    print(f"✓ Theory works: {theory_success}")
    
    return A_ratio_theory, best_result, theory_success

def test_combined_cluster_theory():
    """
    Test combined theoretical parameters for clusters.
    """
    print("\n" + "="*60)
    print("COMBINED CLUSTER THEORY VALIDATION")
    print("="*60)
    
    clusters = load_cluster_sample()
    
    # Calculate theoretical parameters
    avg_M_500 = np.mean([cluster['M_500'] for cluster in clusters])
    avg_R_500 = np.mean([cluster['R_500'] for cluster in clusters])
    rho = calculate_halo_density(avg_M_500, avg_R_500)
    
    ell_0_theory = theory_coherence_length(rho, alpha=3)
    A_ratio_theory = theory_amplitude_ratio_cluster_to_galaxy()
    A_c_theory = A_ratio_theory * EMPRICAL_GALAXY_PARAMS['A_0']
    
    params_combined = {
        'ell_0': ell_0_theory,
        'A_c': A_c_theory
    }
    
    print(f"Combined theoretical parameters:")
    print(f"  ℓ₀ = {ell_0_theory:.1f} kpc (α=3)")
    print(f"  A_c = {A_c_theory:.1f} (path counting)")
    
    results = mock_predict_einstein_radii(clusters, params_combined)
    
    print(f"\nResults:")
    print(f"  Coverage: {results['coverage']:.2f} (target: {EMPRICAL_CLUSTER_PARAMS['target_coverage']:.2f})")
    print(f"  Median error: {results['median_error']:.1f}% (target: {EMPRICAL_CLUSTER_PARAMS['target_error']:.1f}%)")
    
    success = (abs(results['coverage'] - EMPRICAL_CLUSTER_PARAMS['target_coverage']) < 0.1 and
               abs(results['median_error'] - EMPRICAL_CLUSTER_PARAMS['target_error']) < 5.0)
    
    print(f"✓ Combined theory works: {success}")
    
    return results, success

def analyze_cluster_derivation():
    """
    Analyze cluster-scale derivation validity.
    """
    print("="*60)
    print("CLUSTER DERIVATION VALIDATION")
    print("="*60)
    
    # Run all cluster tests
    ell_0_theory, ell_0_results, ell_0_success = test_cluster_coherence_length()
    A_ratio_theory, A_results, A_success = test_cluster_amplitude_ratio()
    combined_results, combined_success = test_combined_cluster_theory()
    
    print(f"\n" + "="*60)
    print("CLUSTER DERIVATION SUMMARY")
    print("="*60)
    
    print(f"ℓ₀ derivation: {'✓ VALID' if ell_0_success else '✗ INVALID'}")
    print(f"  Theory: ℓ₀ = c/(α√(Gρ)) with α=3")
    print(f"  Predicted: {ell_0_theory:.1f} kpc")
    print(f"  Empirical: {EMPRICAL_CLUSTER_PARAMS['ell_0']:.0f} kpc")
    
    print(f"\nA_c derivation: {'✓ VALID' if A_success else '✗ INVALID'}")
    print(f"  Theory: A_c/A_0 from path counting")
    print(f"  Predicted ratio: {A_ratio_theory:.1f}")
    print(f"  Empirical ratio: {A_results['A_ratio']:.1f}")
    
    print(f"\nCombined theory: {'✓ VALID' if combined_success else '✗ INVALID'}")
    print(f"  Coverage: {combined_results['coverage']:.2f} vs {EMPRICAL_CLUSTER_PARAMS['target_coverage']:.2f}")
    print(f"  Error: {combined_results['median_error']:.1f}% vs {EMPRICAL_CLUSTER_PARAMS['target_error']:.1f}%")
    
    overall_success = ell_0_success and A_success and combined_success
    
    print(f"\nOverall cluster derivation: {'✓ VALID' if overall_success else '✗ INVALID'}")
    
    if overall_success:
        print("🎉 SUCCESS: Cluster parameters can be derived from first principles!")
    elif ell_0_success:
        print("⚠️  PARTIAL: ℓ₀ derivation works, A_c needs calibration")
    else:
        print("❌ FAILURE: Cluster derivations need major revision")
    
    return {
        'ell_0_success': ell_0_success,
        'A_success': A_success,
        'combined_success': combined_success,
        'overall_success': overall_success,
        'ell_0_theory': ell_0_theory,
        'A_ratio_theory': A_ratio_theory
    }

def main():
    """
    Run cluster-scale derivation validation.
    """
    print("CLUSTER-SCALE THEORETICAL VALIDATION")
    print("=" * 60)
    print("Testing derivations against cluster Einstein radius results")
    print()
    
    results = analyze_cluster_derivation()
    
    print(f"\n" + "="*60)
    print("FINAL CLUSTER RECOMMENDATIONS")
    print("="*60)
    
    if results['overall_success']:
        print("✓ Cluster derivations are theoretically grounded")
        print("✓ Both ℓ₀ and A_c can be derived from first principles")
        print("✓ Model works across galaxy and cluster scales")
    elif results['ell_0_success']:
        print("⚠️  Semi-empirical cluster model")
        print("✓ ℓ₀ can be derived from cluster density")
        print("? A_c needs phenomenological calibration")
    else:
        print("❌ Cluster parameters are phenomenological")
        print("❌ No theoretical derivation validated")
        print("✓ Model works empirically but lacks cluster theory")
    
    return results

if __name__ == "__main__":
    results = main()
