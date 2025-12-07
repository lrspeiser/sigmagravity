#!/usr/bin/env python3
"""
Can we optimize Option A to beat MOND on galaxies while keeping cluster performance?

The issue: Option A with current parameters loses to MOND on galaxies.
But maybe we can adjust A(G) or h(g) to fix this.

Author: Leonard Speiser
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from pathlib import Path
import json

# Physical constants
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
km_to_m = 1000

g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
a0_mond = 1.2e-10

def load_galaxies():
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG")
    galaxies = []
    for gf in sparc_dir.glob("*.dat"):
        try:
            data = np.loadtxt(gf, comments='#')
            if len(data) < 5:
                continue
            gal = {
                'name': gf.stem,
                'R': data[:, 0],
                'V_obs': data[:, 1],
                'V_bar': np.sqrt(data[:, 3]**2 + 0.5*data[:, 4]**2 + 
                                (0.7*data[:, 5]**2 if data.shape[1] > 5 else 0))
            }
            if np.max(gal['V_obs']) > 10:
                galaxies.append(gal)
        except:
            continue
    return galaxies

def h_function(g, g_dag):
    """Σ-Gravity h function with adjustable g†."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)

def nu_mond(g):
    """MOND simple interpolation."""
    x = g / a0_mond
    return 0.5 + np.sqrt(0.25 + 1/x)

def compute_rms(galaxies, A, g_dag):
    """Compute RMS for Option A with given parameters."""
    total_sq = 0
    total_n = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * km_to_m
        g_N = V_bar_ms**2 / R_m
        
        h = h_function(g_N, g_dag)
        Sigma = 1 + A * h
        V_pred = V_bar * np.sqrt(Sigma)
        
        total_sq += np.sum((V_obs - V_pred)**2)
        total_n += len(R)
    
    return np.sqrt(total_sq / total_n)

def compute_mond_rms(galaxies):
    """Compute RMS for MOND."""
    total_sq = 0
    total_n = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * km_to_m
        g_N = V_bar_ms**2 / R_m
        
        nu = nu_mond(g_N)
        V_pred = V_bar * np.sqrt(nu)
        
        total_sq += np.sum((V_obs - V_pred)**2)
        total_n += len(R)
    
    return np.sqrt(total_sq / total_n)

def main():
    print("=" * 70)
    print("OPTIMIZING OPTION A TO BEAT MOND")
    print("=" * 70)
    
    galaxies = load_galaxies()
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Current parameters
    A_current = np.sqrt(1.6 + 109 * 0.038**2)  # 1.326
    g_dag_current = g_dagger
    
    rms_current = compute_rms(galaxies, A_current, g_dag_current)
    rms_mond = compute_mond_rms(galaxies)
    
    print(f"\nCurrent Option A: RMS = {rms_current:.2f} km/s")
    print(f"MOND: RMS = {rms_mond:.2f} km/s")
    print(f"Gap: {rms_current - rms_mond:.2f} km/s")
    
    # =========================================================================
    # OPTION 1: Optimize A only (keep g†)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 1: OPTIMIZE A (KEEP g†)")
    print(f"{'='*70}")
    
    def obj_A(A):
        return compute_rms(galaxies, A, g_dag_current)
    
    result = minimize_scalar(obj_A, bounds=(0.5, 5.0), method='bounded')
    A_opt = result.x
    rms_opt_A = result.fun
    
    print(f"Optimal A = {A_opt:.3f} (was {A_current:.3f})")
    print(f"RMS = {rms_opt_A:.2f} km/s")
    print(f"vs MOND: {rms_opt_A - rms_mond:+.2f} km/s")
    
    # =========================================================================
    # OPTION 2: Optimize g† only (keep A)
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 2: OPTIMIZE g† (KEEP A)")
    print(f"{'='*70}")
    
    def obj_g(g_dag):
        return compute_rms(galaxies, A_current, g_dag)
    
    result = minimize_scalar(obj_g, bounds=(5e-11, 2e-10), method='bounded')
    g_opt = result.x
    rms_opt_g = result.fun
    
    print(f"Optimal g† = {g_opt:.3e} m/s² (was {g_dag_current:.3e})")
    print(f"RMS = {rms_opt_g:.2f} km/s")
    print(f"vs MOND: {rms_opt_g - rms_mond:+.2f} km/s")
    
    # =========================================================================
    # OPTION 3: Optimize both A and g†
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 3: OPTIMIZE BOTH A AND g†")
    print(f"{'='*70}")
    
    def obj_both(params):
        A, g_dag = params
        if A < 0.1 or g_dag < 1e-11:
            return 1e10
        return compute_rms(galaxies, A, g_dag)
    
    result = minimize(obj_both, [A_current, g_dag_current], method='Nelder-Mead')
    A_both, g_both = result.x
    rms_both = result.fun
    
    print(f"Optimal A = {A_both:.3f}, g† = {g_both:.3e} m/s²")
    print(f"RMS = {rms_both:.2f} km/s")
    print(f"vs MOND: {rms_both - rms_mond:+.2f} km/s")
    
    # =========================================================================
    # OPTION 4: Use MOND's a0 instead of our g†
    # =========================================================================
    print(f"\n{'='*70}")
    print("OPTION 4: USE MOND's a₀ INSTEAD OF g†")
    print(f"{'='*70}")
    
    def obj_A_with_a0(A):
        return compute_rms(galaxies, A, a0_mond)
    
    result = minimize_scalar(obj_A_with_a0, bounds=(0.5, 5.0), method='bounded')
    A_a0 = result.x
    rms_a0 = result.fun
    
    print(f"With a₀ = 1.2×10⁻¹⁰ m/s²:")
    print(f"Optimal A = {A_a0:.3f}")
    print(f"RMS = {rms_a0:.2f} km/s")
    print(f"vs MOND: {rms_a0 - rms_mond:+.2f} km/s")
    
    # =========================================================================
    # CHECK CLUSTER IMPLICATIONS
    # =========================================================================
    print(f"\n{'='*70}")
    print("CLUSTER IMPLICATIONS")
    print(f"{'='*70}")
    
    # For clusters, we need A(G=1) to be much larger than A(G=0.038)
    # Current: A(G=1) = √(1.6 + 109) = 10.5
    # If we change A for galaxies, what happens to the formula?
    
    # The formula is A(G) = √(a + b×G²)
    # For galaxies (G=0.038): A ≈ √a (since b×G² is small)
    # For clusters (G=1): A ≈ √(a + b)
    
    print(f"""
Current formula: A(G) = √(1.6 + 109×G²)
  A(G=0.038) = {A_current:.3f}
  A(G=1.0) = {np.sqrt(1.6 + 109):.2f}
  Ratio = {np.sqrt(1.6 + 109)/A_current:.1f}×

If we optimize A for galaxies to {A_opt:.3f}:
  Need a = {A_opt**2:.2f}
  Keep b = 109 → A(G=1) = {np.sqrt(A_opt**2 + 109):.2f}
  Ratio = {np.sqrt(A_opt**2 + 109)/A_opt:.1f}×
  
This REDUCES the cluster/galaxy ratio, potentially breaking clusters!
""")
    
    # What if we keep the ratio fixed?
    ratio_current = np.sqrt(1.6 + 109) / A_current
    
    # If A_galaxy = A_opt, and we want same ratio:
    # A_cluster = ratio_current × A_opt
    # √(a + b) = ratio × √a
    # a + b = ratio² × a
    # b = a × (ratio² - 1)
    
    a_new = A_opt**2
    b_new = a_new * (ratio_current**2 - 1)
    A_cluster_new = np.sqrt(a_new + b_new)
    
    print(f"To maintain cluster/galaxy ratio = {ratio_current:.1f}×:")
    print(f"  a = {a_new:.2f}, b = {b_new:.1f}")
    print(f"  A(G=0.038) = {A_opt:.3f}")
    print(f"  A(G=1.0) = {A_cluster_new:.2f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"""
| Configuration | Galaxy RMS | vs MOND | Cluster OK? |
|---------------|------------|---------|-------------|
| Current (A=1.33, g†) | {rms_current:.1f} km/s | +{rms_current-rms_mond:.1f} | Yes |
| Optimize A only | {rms_opt_A:.1f} km/s | {rms_opt_A-rms_mond:+.1f} | Maybe |
| Optimize g† only | {rms_opt_g:.1f} km/s | {rms_opt_g-rms_mond:+.1f} | Yes |
| Optimize both | {rms_both:.1f} km/s | {rms_both-rms_mond:+.1f} | Check |
| Use MOND's a₀ | {rms_a0:.1f} km/s | {rms_a0-rms_mond:+.1f} | Yes |
| MOND | {rms_mond:.1f} km/s | — | No |

KEY INSIGHT:
The issue is that our h(g) function with g† = 9.6×10⁻¹¹ gives LESS
enhancement than MOND's ν with a₀ = 1.2×10⁻¹⁰ at typical galaxy accelerations.

Options to fix:
1. Use a₀ instead of g† (but loses "derived" status)
2. Increase A slightly (but may affect cluster ratio)
3. Accept that Option A needs W(r) to compete with MOND

The geometry factor A(G) is what distinguishes us from MOND for CLUSTERS,
but for GALAXIES, our h(g) underperforms MOND's ν.
""")
    
    # Save results
    results = {
        'current': {'A': A_current, 'g_dagger': g_dag_current, 'rms': rms_current},
        'mond': {'rms': rms_mond},
        'opt_A': {'A': A_opt, 'rms': rms_opt_A},
        'opt_g': {'g_dagger': g_opt, 'rms': rms_opt_g},
        'opt_both': {'A': A_both, 'g_dagger': g_both, 'rms': rms_both},
        'with_a0': {'A': A_a0, 'rms': rms_a0}
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/option_a_optimized_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

