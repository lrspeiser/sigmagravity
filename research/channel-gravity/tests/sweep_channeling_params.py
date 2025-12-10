"""
Parameter Sweep for Gravitational Channeling
=============================================

Find optimal parameters that:
1. Pass Solar System constraint
2. Maximize SPARC improvement rate
3. Minimize over-boosting of massive spirals
"""

import sys
import os
import glob
import numpy as np
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channeling_kernel import (
    ChannelingParams, load_sparc_galaxy, fit_galaxy, 
    test_solar_system
)


def quick_sparc_test(data_dir: str, params: ChannelingParams, max_galaxies: int = 50):
    """Quick test on subset of galaxies."""
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))[:max_galaxies]
    
    improved = 0
    total = 0
    delta_rms_list = []
    
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, params)
            total += 1
            if result['improved']:
                improved += 1
            delta_rms_list.append(result['delta_rms'])
        except:
            continue
    
    return {
        'pct_improved': 100 * improved / total if total > 0 else 0,
        'median_delta_rms': np.median(delta_rms_list) if delta_rms_list else 0,
        'max_delta_rms': np.max(delta_rms_list) if delta_rms_list else 0,
    }


def main():
    print("=" * 70)
    print("GRAVITATIONAL CHANNELING - PARAMETER SWEEP")
    print("=" * 70)
    
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    
    # Parameter ranges to sweep
    chi_0_values = [0.3, 0.5, 0.8, 1.2]
    beta_values = [0.5, 0.8, 1.0, 1.5]
    alpha_values = [0.5, 0.7, 1.0]
    epsilon_values = [0.3, 0.5, 0.7]
    D_max_values = [3.0, 5.0, 10.0]
    
    results = []
    
    total_combos = len(chi_0_values) * len(beta_values) * len(alpha_values) * len(epsilon_values) * len(D_max_values)
    print(f"\nTesting {total_combos} parameter combinations...")
    
    combo_num = 0
    for chi_0, beta, alpha, epsilon, D_max in product(
        chi_0_values, beta_values, alpha_values, epsilon_values, D_max_values
    ):
        combo_num += 1
        
        params = ChannelingParams(
            alpha=alpha,
            beta=beta,
            gamma=0.3,  # Keep fixed
            chi_0=chi_0,
            epsilon=epsilon,
            D_max=D_max,
            t_age=10.0,
            tau_0=1.0,
        )
        
        # Solar System test
        delta_g, ss_passes = test_solar_system(params)
        
        if not ss_passes:
            continue  # Skip if fails SS
        
        # Quick SPARC test
        sparc = quick_sparc_test(data_dir, params, max_galaxies=171)
        
        results.append({
            'chi_0': chi_0,
            'beta': beta,
            'alpha': alpha,
            'epsilon': epsilon,
            'D_max': D_max,
            'delta_g': delta_g,
            'pct_improved': sparc['pct_improved'],
            'median_delta_rms': sparc['median_delta_rms'],
            'max_delta_rms': sparc['max_delta_rms'],
        })
        
        if combo_num % 20 == 0:
            print(f"  Tested {combo_num}/{total_combos}...")
    
    print(f"\n{len(results)} combinations pass Solar System constraint")
    
    if not results:
        print("ERROR: No parameter combinations pass Solar System!")
        return
    
    # Sort by improvement rate
    results.sort(key=lambda x: -x['pct_improved'])
    
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS (by % improved)")
    print("=" * 70)
    print(f"{'χ₀':<6} {'β':<6} {'α':<6} {'ε':<6} {'D_max':<6} {'%imp':<8} {'med_Δ':<10} {'max_Δ':<10} {'δg/g':<12}")
    print("-" * 80)
    
    for r in results[:10]:
        print(f"{r['chi_0']:<6.1f} {r['beta']:<6.1f} {r['alpha']:<6.1f} {r['epsilon']:<6.1f} "
              f"{r['D_max']:<6.1f} {r['pct_improved']:<8.1f} {r['median_delta_rms']:<10.2f} "
              f"{r['max_delta_rms']:<10.1f} {r['delta_g']:<12.2e}")
    
    # Also show top by lowest max_delta_rms (least over-boosting)
    results_low_overshoot = sorted(results, key=lambda x: x['max_delta_rms'])
    
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS (least over-boosting)")
    print("=" * 70)
    print(f"{'χ₀':<6} {'β':<6} {'α':<6} {'ε':<6} {'D_max':<6} {'%imp':<8} {'med_Δ':<10} {'max_Δ':<10} {'δg/g':<12}")
    print("-" * 80)
    
    for r in results_low_overshoot[:10]:
        print(f"{r['chi_0']:<6.1f} {r['beta']:<6.1f} {r['alpha']:<6.1f} {r['epsilon']:<6.1f} "
              f"{r['D_max']:<6.1f} {r['pct_improved']:<8.1f} {r['median_delta_rms']:<10.2f} "
              f"{r['max_delta_rms']:<10.1f} {r['delta_g']:<12.2e}")
    
    # Best balanced (high improvement, low overshoot)
    # Score = pct_improved - max_delta_rms/10 (penalize overshoot)
    for r in results:
        r['score'] = r['pct_improved'] - r['max_delta_rms'] / 5
    
    results_balanced = sorted(results, key=lambda x: -x['score'])
    
    print("\n" + "=" * 70)
    print("TOP 10 BALANCED (improvement vs overshoot)")
    print("=" * 70)
    print(f"{'χ₀':<6} {'β':<6} {'α':<6} {'ε':<6} {'D_max':<6} {'%imp':<8} {'med_Δ':<10} {'max_Δ':<10} {'score':<8}")
    print("-" * 80)
    
    for r in results_balanced[:10]:
        print(f"{r['chi_0']:<6.1f} {r['beta']:<6.1f} {r['alpha']:<6.1f} {r['epsilon']:<6.1f} "
              f"{r['D_max']:<6.1f} {r['pct_improved']:<8.1f} {r['median_delta_rms']:<10.2f} "
              f"{r['max_delta_rms']:<10.1f} {r['score']:<8.1f}")
    
    # Best overall
    best = results_balanced[0]
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"χ₀ = {best['chi_0']}")
    print(f"β = {best['beta']}")
    print(f"α = {best['alpha']}")
    print(f"ε = {best['epsilon']}")
    print(f"D_max = {best['D_max']}")
    print(f"\nExpected: {best['pct_improved']:.1f}% improved, median ΔRMS = {best['median_delta_rms']:.2f} km/s")


if __name__ == "__main__":
    main()
