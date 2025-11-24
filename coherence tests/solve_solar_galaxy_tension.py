#!/usr/bin/env python3
"""
Solve the Solar-Galaxy Tension with Cosmically Locked Σ-Gravity

This script finds the stiffened Burr-XII parameters (p, n) that satisfy:
1. Solar System: Boost < 10⁻¹⁰
2. Galaxies: Enhancement ~ 1.3-1.8×
3. Clusters: Enhancement ~ 6.25×

The physics is LOCKED:
- α = 5.25 (from Ω_b/Ω_m)
- a_H = 3700 (from H₀)

Only the shape (p, n) is fitted, with constraint p×n ≥ 2.2 for Solar System safety.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- LOCKED COSMIC PHYSICS ---
ALPHA = 5.25          # Derived from Omega_b/Omega_m = 0.16
A_HUBBLE = 3700.0     # Derived from H0 ~ 70 km/s/Mpc

# --- SOLAR SYSTEM SAFETY CONSTRAINT ---
# At 1 AU: g ~ 9e5, a_H ~ 3700, x ~ 243
# Need: Boost = α × I_solar × C(x) < 1e-10
# With I_solar ~ 1e-4 (from eccentricity):
# Need: C(x) < 2e-7
# With Burr-XII: C = 1/(1 + x^p)^n
# At x = 243: Need (243^p)^n >> 1e7
# This requires p×n ≥ 2.2 approximately
SAFETY_THRESHOLD = 2.2

def cosmic_burr_kernel(r, v_bar, p, n, sigma_star=20.0, velocity_type='disk'):
    """
    The Cosmically Locked Kernel with Stiffened Burr-XII.
    
    Parameters:
    -----------
    r : array
        Radius in kpc
    v_bar : array
        Baryonic circular velocity in km/s
    p, n : float
        Burr-XII shape parameters (to be fitted)
    sigma_star : float
        Stellar velocity dispersion in km/s
    velocity_type : str
        'disk' or 'cluster'
        
    Returns:
    --------
    v_pred : array
        Predicted total circular velocity
    diagnostics : dict
        Internal state variables
    """
    r_safe = np.maximum(r, 1e-3)
    g_bar = (v_bar**2) / r_safe
    
    # The Acceleration Ratio
    x = g_bar / A_HUBBLE
    
    # Isotropy Gate (simplified - should use actual σ profile from data)
    if velocity_type == 'cluster':
        I_geo = 1.0
    else:
        # For rotation-dominated disks
        I_geo = (3.0 * sigma_star**2) / (v_bar**2 + 3.0 * sigma_star**2)
        I_geo = np.maximum(I_geo, 1e-4)  # Floor to avoid zeros
    
    # Stiffened Coherence (Burr-XII in acceleration space)
    # Note: This form goes to 0 as x → ∞ (high acceleration)
    # and goes to 1 as x → 0 (low acceleration)
    coherence = 1.0 / ((1.0 + x**p)**n)
    
    # Enhancement
    K = ALPHA * I_geo * coherence
    g_eff = g_bar * (1.0 + K)
    
    v_pred = np.sqrt(g_eff * r_safe)
    
    return v_pred, {
        'g_bar': g_bar,
        'x': x,
        'I_geo': I_geo,
        'coherence': coherence,
        'enhancement': 1.0 + K,
        'K': K
    }

def test_parameters(p, n, verbose=True):
    """
    Test a given (p, n) pair across all three scales.
    
    Returns:
    --------
    results : dict
        Test results for Solar System, galaxy, and cluster
    """
    results = {
        'p': p,
        'n': n,
        'stiffness': p * n
    }
    
    # ========================================================================
    # TEST 1: SOLAR SYSTEM (Critical Constraint)
    # ========================================================================
    g_solar = 9e5  # (km/s)²/kpc at 1 AU
    x_solar = g_solar / A_HUBBLE
    I_solar = 1e-4  # From eccentricity (e ~ 0.017, I ~ e²)
    
    C_solar = 1.0 / ((1.0 + x_solar**p)**n)
    boost_solar = ALPHA * I_solar * C_solar
    
    results['solar'] = {
        'x': x_solar,
        'C': C_solar,
        'I_geo': I_solar,
        'boost': boost_solar,
        'safe': boost_solar < 1e-10
    }
    
    # ========================================================================
    # TEST 2: GALAXY (Typical outer disk)
    # ========================================================================
    # At R ~ 10 kpc in typical SPARC galaxy
    g_galaxy = 1000.0  # (km/s)²/kpc
    x_galaxy = g_galaxy / A_HUBBLE
    I_galaxy = 0.15  # Typical rotation-dominated disk
    
    C_galaxy = 1.0 / ((1.0 + x_galaxy**p)**n)
    enh_galaxy = 1.0 + ALPHA * I_galaxy * C_galaxy
    
    results['galaxy'] = {
        'x': x_galaxy,
        'C': C_galaxy,
        'I_geo': I_galaxy,
        'enhancement': enh_galaxy,
        'valid': 1.2 < enh_galaxy < 2.0
    }
    
    # ========================================================================
    # TEST 3: CLUSTER (Outer regions)
    # ========================================================================
    # At R ~ 1 Mpc in cluster outskirts
    g_cluster = 50.0  # (km/s)²/kpc
    x_cluster = g_cluster / A_HUBBLE
    I_cluster = 1.0  # Pressure-supported
    
    C_cluster = 1.0 / ((1.0 + x_cluster**p)**n)
    enh_cluster = 1.0 + ALPHA * I_cluster * C_cluster
    
    results['cluster'] = {
        'x': x_cluster,
        'C': C_cluster,
        'I_geo': I_cluster,
        'enhancement': enh_cluster,
        'valid': enh_cluster > 5.0
    }
    
    # Overall status
    results['all_pass'] = (
        results['solar']['safe'] and 
        results['galaxy']['valid'] and 
        results['cluster']['valid']
    )
    
    if verbose:
        print("\n" + "="*70)
        print(f"TESTING: p = {p:.3f}, n = {n:.3f}, p×n = {p*n:.3f}")
        print("="*70)
        
        print("\n1. SOLAR SYSTEM:")
        print(f"   x = {x_solar:.2e}")
        print(f"   C = {C_solar:.2e}")
        print(f"   Boost = {boost_solar:.2e}")
        print(f"   Status: {'✅ SAFE' if results['solar']['safe'] else '❌ UNSAFE'}")
        
        print("\n2. GALAXY (R ~ 10 kpc):")
        print(f"   x = {x_galaxy:.3f}")
        print(f"   C = {C_galaxy:.3f}")
        print(f"   Enhancement = {enh_galaxy:.2f}×")
        print(f"   Status: {'✅ VALID' if results['galaxy']['valid'] else '⚠️  OUT OF RANGE'}")
        
        print("\n3. CLUSTER (R ~ 1 Mpc):")
        print(f"   x = {x_cluster:.3f}")
        print(f"   C = {C_cluster:.3f}")
        print(f"   Enhancement = {enh_cluster:.2f}×")
        print(f"   Status: {'✅ VALID' if results['cluster']['valid'] else '⚠️  TOO LOW'}")
        
        print("\n" + "="*70)
        print(f"OVERALL: {'✅ ALL TESTS PASS' if results['all_pass'] else '❌ SOME TESTS FAIL'}")
        print("="*70)
    
    return results

def grid_search(p_range=(0.5, 3.0), n_range=(0.5, 5.0), n_points=20):
    """
    Grid search to find viable (p, n) region.
    """
    print("\n" + "="*70)
    print("GRID SEARCH FOR VIABLE PARAMETERS")
    print("="*70)
    print(f"p range: {p_range}")
    print(f"n range: {n_range}")
    print(f"Grid points: {n_points} × {n_points} = {n_points**2}")
    print(f"Constraint: p×n ≥ {SAFETY_THRESHOLD}")
    
    p_vals = np.linspace(p_range[0], p_range[1], n_points)
    n_vals = np.linspace(n_range[0], n_range[1], n_points)
    
    viable = []
    
    for p in p_vals:
        for n in n_vals:
            # Skip if doesn't meet stiffness constraint
            if p * n < SAFETY_THRESHOLD:
                continue
            
            results = test_parameters(p, n, verbose=False)
            
            if results['all_pass']:
                viable.append(results)
    
    print(f"\nFound {len(viable)} viable parameter sets!")
    
    if len(viable) > 0:
        print("\nTop 5 viable sets:")
        print(f"{'p':<8} {'n':<8} {'p×n':<8} {'Solar Boost':<12} {'Galaxy Enh':<12} {'Cluster Enh':<12}")
        print("-" * 70)
        
        # Sort by how close galaxy enhancement is to 1.5
        viable_sorted = sorted(viable, key=lambda x: abs(x['galaxy']['enhancement'] - 1.5))
        
        for r in viable_sorted[:5]:
            print(f"{r['p']:<8.3f} {r['n']:<8.3f} {r['stiffness']:<8.3f} "
                  f"{r['solar']['boost']:<12.2e} {r['galaxy']['enhancement']:<12.2f} "
                  f"{r['cluster']['enhancement']:<12.2f}")
        
        # Return best candidate
        return viable_sorted[0]
    else:
        print("\n⚠️  No viable parameters found in grid!")
        print("This suggests a phase transition may be needed.")
        return None

def analyze_tradeoffs():
    """
    Analyze the fundamental tradeoffs in parameter space.
    """
    print("\n" + "="*70)
    print("ANALYZING PARAMETER SPACE TRADEOFFS")
    print("="*70)
    
    print("\nEffect of increasing p (steepness):")
    print("  - Higher p → steeper transition")
    print("  - Solar System: Better (faster decay)")
    print("  - Galaxy: Worse (transition becomes too sharp)")
    print("  - Cluster: Depends (may reduce low-x coherence)")
    
    print("\nEffect of increasing n (strength):")
    print("  - Higher n → stronger suppression")
    print("  - Solar System: Better (faster decay)")
    print("  - Galaxy: Mixed (reduces enhancement)")
    print("  - Cluster: Worse (reduces enhancement)")
    
    print("\nThe fundamental tension:")
    print("  - Need p×n ≥ 2.2 for Solar System")
    print("  - Need C(x~0.27) ~ 0.5-0.8 for galaxies (x ~ g_gal/a_H)")
    print("  - Need C(x~0.01) ~ 0.8-1.0 for clusters (x ~ g_clus/a_H)")
    
    print("\nTesting representative cases:")
    
    # Case 1: Moderate stiffness
    print("\n" + "-"*70)
    print("CASE 1: p = 1.0, n = 2.5 (p×n = 2.5)")
    test_parameters(1.0, 2.5)
    
    # Case 2: High stiffness
    print("\n" + "-"*70)
    print("CASE 2: p = 1.5, n = 2.0 (p×n = 3.0)")
    test_parameters(1.5, 2.0)
    
    # Case 3: Very high stiffness
    print("\n" + "-"*70)
    print("CASE 3: p = 2.0, n = 1.5 (p×n = 3.0)")
    test_parameters(2.0, 1.5)

def main():
    print("\n" + "="*70)
    print("COSMICALLY LOCKED Σ-GRAVITY: SOLVING THE TRIPLE CONSTRAINT")
    print("="*70)
    print("\n🌌 LOCKED PHYSICS:")
    print(f"   α = {ALPHA:.2f} (from Ω_b/Ω_m = 0.16)")
    print(f"   a_H = {A_HUBBLE:.0f} (km/s)²/kpc (from H₀)")
    
    print("\n🎯 GOAL:")
    print("   Find p, n such that:")
    print("   1. Solar System: Boost < 10⁻¹⁰ ✓")
    print("   2. Galaxy: Enhancement ~ 1.3-1.8× ✓")
    print("   3. Cluster: Enhancement > 5× ✓")
    
    print("\n⚙️  CONSTRAINT:")
    print(f"   p × n ≥ {SAFETY_THRESHOLD} (Solar System safety)")
    
    # Step 1: Analyze tradeoffs
    analyze_tradeoffs()
    
    # Step 2: Grid search
    print("\n" + "="*70)
    best = grid_search(p_range=(0.8, 2.5), n_range=(1.0, 4.0), n_points=25)
    
    if best:
        print("\n" + "="*70)
        print("🎉 SOLUTION FOUND!")
        print("="*70)
        print(f"\nBest parameters:")
        print(f"  p = {best['p']:.4f}")
        print(f"  n = {best['n']:.4f}")
        print(f"  p×n = {best['stiffness']:.4f}")
        
        print(f"\nPerformance:")
        print(f"  Solar System: {best['solar']['boost']:.2e} ✅")
        print(f"  Galaxy: {best['galaxy']['enhancement']:.2f}× ✅")
        print(f"  Cluster: {best['cluster']['enhancement']:.2f}× ✅")
        
        print("\n📋 NEXT STEPS:")
        print("  1. Test these parameters on full SPARC sample")
        print("  2. Measure RAR scatter (target < 0.10 dex)")
        print("  3. Validate on cluster lensing data")
        print("  4. Write Nature Physics paper!")
        
    else:
        print("\n" + "="*70)
        print("⚠️  NO SIMPLE SOLUTION FOUND")
        print("="*70)
        print("\nThis suggests the vacuum may have a phase transition:")
        print("  - Superfluid phase (x < 1): C ≈ 1")
        print("  - Normal phase (x > 1): C → 0 rapidly")
        print("\nThis is physically motivated by:")
        print("  - Hubble scale as thermal boundary")
        print("  - Decoherence temperature T_H ~ ℏa_H/k_B")
        print("  - Similar to superfluidity in liquid helium")
        
        print("\nRecommended approach:")
        print("  1. Model as piecewise function with smooth transition")
        print("  2. Or accept that simple Burr-XII can't satisfy all three")
        print("  3. Use different (p,n) for galaxies vs clusters")

if __name__ == "__main__":
    main()
