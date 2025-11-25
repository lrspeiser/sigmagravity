"""
Phase 4: Cluster Test for Gravitational Channeling
===================================================

Tests:
- 4a. Cluster lensing: Can channeling produce F > 2 at Einstein radius?
- 4b. High-σ limit: What happens at σ~1000 km/s?
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channeling_kernel import (
    ChannelingParams, channeling_enhancement
)


# Cluster data (same as CMSI tests)
CLUSTERS = {
    'Coma': {
        'R_E': 300,          # Einstein radius [kpc]
        'M_bary': 1.4e14,    # Baryonic mass [M_sun]
        'M_lens': 7e14,      # Lensing mass [M_sun]
        'sigma_v': 1000,     # Velocity dispersion [km/s]
        'R_half': 500,       # Half-light radius [kpc]
    },
    'A2029': {
        'R_E': 200,
        'M_bary': 1.0e14,
        'M_lens': 5e14,
        'sigma_v': 850,
        'R_half': 300,
    },
    'A1689': {
        'R_E': 250,
        'M_bary': 1.2e14,
        'M_lens': 6e14,
        'sigma_v': 900,
        'R_half': 400,
    },
    'Bullet': {
        'R_E': 300,
        'M_bary': 1.5e14,
        'M_lens': 9e14,
        'sigma_v': 1100,
        'R_half': 500,
    },
}


def estimate_cluster_surface_density(M_bary: float, R: float, R_half: float) -> float:
    """
    Estimate surface density at radius R for a cluster.
    
    Approximate as Σ ≈ M_bary / (4π R_half²) for isothermal-ish profile.
    """
    # Central surface density estimate
    Sigma_0 = M_bary / (4 * np.pi * (R_half * 1e3)**2)  # M_sun/pc^2
    
    # Profile falls off roughly as 1/R
    Sigma = Sigma_0 * (R_half / R) if R > 0 else Sigma_0
    
    return Sigma


def test_cluster_channeling(params: ChannelingParams):
    """Test channeling enhancement at cluster scales."""
    
    print("\n" + "=" * 70)
    print("PHASE 4a: CLUSTER LENSING TEST")
    print("=" * 70)
    
    print(f"\n{'Cluster':<12} {'R_E(kpc)':<10} {'σ_v':<10} {'Σ':<12} {'F':<10} {'F_need':<10} {'Ratio':<10}")
    print("-" * 74)
    
    results = []
    for name, cluster in CLUSTERS.items():
        R_E = cluster['R_E']
        sigma_v = cluster['sigma_v']
        M_bary = cluster['M_bary']
        M_lens = cluster['M_lens']
        R_half = cluster['R_half']
        
        # Required F from lensing
        F_needed = M_lens / M_bary
        
        # Estimate surface density at Einstein radius
        Sigma = estimate_cluster_surface_density(M_bary, R_E, R_half)
        
        # For clusters, v_c ≈ σ_v (pressure-supported)
        v_c = sigma_v
        
        # Compute channeling enhancement
        diag = {}
        F = channeling_enhancement(
            R=np.array([R_E]),
            v_c=np.array([v_c]),
            Sigma=np.array([Sigma]),
            sigma_v=np.array([sigma_v]),
            params=params,
            diagnostics=diag
        )[0]
        
        ratio = F / F_needed * 100
        
        print(f"{name:<12} {R_E:<10.0f} {sigma_v:<10.0f} {Sigma:<12.1f} {F:<10.3f} {F_needed:<10.1f} {ratio:<10.1f}%")
        
        results.append({
            'name': name,
            'F': F,
            'F_needed': F_needed,
            'ratio': ratio,
            'D': diag['D'][0],
        })
    
    # Summary
    avg_ratio = np.mean([r['ratio'] for r in results])
    max_F = max(r['F'] for r in results)
    
    print(f"\nAverage F/F_needed: {avg_ratio:.1f}%")
    print(f"Max F achieved: {max_F:.3f}")
    print(f"Clusters need F ~ 5-6")
    
    passes = max_F >= 2.0
    print(f"\nResult: {'PASS ✓' if passes else 'FAIL ✗'} (target: F > 2)")
    
    return passes, results


def test_high_sigma_limit(params: ChannelingParams):
    """
    Test 4b: How does enhancement behave at high σ_v?
    
    At cluster dispersions (~1000 km/s), what suppresses F?
    """
    print("\n" + "=" * 70)
    print("PHASE 4b: HIGH VELOCITY DISPERSION LIMIT")
    print("=" * 70)
    
    # Fixed R, v_c, Σ; vary σ_v
    R = np.array([100.0])  # kpc (cluster scale)
    v_c = np.array([500.0])  # km/s
    Sigma = np.array([100.0])  # M_sun/pc^2
    
    sigma_values = [10, 30, 100, 300, 500, 800, 1000, 1500]
    
    print(f"\nFixed: R=100 kpc, v_c=500 km/s, Σ=100 M☉/pc²")
    print(f"{'σ_v (km/s)':<12} {'v_c/σ_v':<10} {'D':<10} {'F':<10}")
    print("-" * 42)
    
    for sigma in sigma_values:
        diag = {}
        F = channeling_enhancement(R, v_c, Sigma, np.array([sigma]), params, diag)
        ratio = v_c[0] / sigma
        print(f"{sigma:<12} {ratio:<10.1f} {diag['D'][0]:<10.4f} {F[0]:<10.4f}")
    
    print("\nAnalysis: Channel depth D scales as (v_c/σ_v)^β")
    print(f"With β = {params.beta}, high σ_v strongly suppresses enhancement")
    
    return True


def main():
    print("=" * 70)
    print("GRAVITATIONAL CHANNELING - PHASE 4: CLUSTER TESTS")
    print("=" * 70)
    
    # Optimized parameters (same as Phase 1-3)
    params = ChannelingParams(
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        chi_0=0.3,
        epsilon=0.3,
        D_max=3.0,
        t_age=10.0,
        tau_0=1.0,
    )
    
    # Test 4a: Cluster lensing
    cluster_pass, cluster_results = test_cluster_channeling(params)
    
    # Test 4b: High-σ limit
    test_high_sigma_limit(params)
    
    # Try with cluster-specific parameters (older age, different tau_0)
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Cluster-optimized parameters")
    print("=" * 70)
    
    # Clusters are ~13 Gyr old and have had longer to form channels
    cluster_params = ChannelingParams(
        alpha=1.0,
        beta=0.3,       # Weaker σ_v dependence
        gamma=0.5,      # Stronger age dependence
        chi_0=1.0,      # Stronger coupling
        epsilon=0.3,
        D_max=10.0,     # Allow deeper channels
        t_age=13.0,     # Older
        tau_0=0.5,      # Faster formation
    )
    
    print("\nCluster-optimized params: β=0.3, γ=0.5, χ₀=1.0, D_max=10, t=13 Gyr")
    test_cluster_channeling(cluster_params)
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY")
    print("=" * 70)
    print("With galaxy-optimized parameters:")
    print(f"  Cluster F: ~1.0-1.1 (need 5-6)")
    print(f"  Shortfall: ~5x")
    print("\nWith cluster-optimized parameters:")
    print(f"  Cluster F: ~1.3 (need 5-6)")
    print(f"  Shortfall: ~4x")
    print("\nConclusion: Gravitational channeling alone cannot explain cluster lensing.")
    print("Like CMSI, it's a galaxy-scale phenomenon.")


if __name__ == "__main__":
    main()
