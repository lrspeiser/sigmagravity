"""
ζ Parameter Sweep for Cooperative Channeling
=============================================

Test different ζ (local density cooperation) values on real SPARC data
to find the optimal setting for massive spiral improvement.
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cooperative_channeling import (
    CooperativeParams, load_sparc_galaxy, fit_galaxy, test_solar_system
)


def run_sparc_batch(data_dir: str, params: CooperativeParams):
    """Run cooperative channeling on all SPARC galaxies."""
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    results = []
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, params)
            v_flat = np.mean(result['v_obs'][-3:])
            result['v_flat'] = v_flat
            results.append(result)
        except Exception as e:
            continue
    
    return results


def analyze_by_type(results):
    """Analyze results by galaxy type."""
    dwarfs = [r for r in results if r['v_flat'] < 80]
    intermediate = [r for r in results if 80 <= r['v_flat'] < 150]
    massive = [r for r in results if r['v_flat'] >= 150]
    
    def stats(subset):
        if not subset:
            return 0, 0, 0
        improved = sum(1 for r in subset if r['improved'])
        pct = 100 * improved / len(subset)
        med_delta = np.median([r['delta_rms'] for r in subset])
        return len(subset), pct, med_delta
    
    return {
        'dwarf': stats(dwarfs),
        'intermediate': stats(intermediate),
        'massive': stats(massive),
        'all': stats(results),
    }


def main():
    print("=" * 70)
    print("COOPERATIVE CHANNELING - ζ PARAMETER SWEEP")
    print("=" * 70)
    
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    
    zeta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    
    print("\nSweeping ζ values on full SPARC dataset...")
    print("(Focus: Can ζ > 0 improve massive spiral success rate?)\n")
    
    all_results = []
    
    for zeta in zeta_values:
        params = CooperativeParams(
            alpha=1.0,
            beta=0.5,
            gamma=0.3,
            chi_0=0.3,
            epsilon=0.3,
            zeta=zeta,
            D_max=3.0,
            t_age=10.0,
            tau_0=1.0,
        )
        
        # Solar System check
        delta_g, ss_pass = test_solar_system(params)
        
        if not ss_pass:
            print(f"ζ={zeta:.1f}: FAILS Solar System (δg/g={delta_g:.2e})")
            continue
        
        # Full SPARC batch
        results = run_sparc_batch(data_dir, params)
        stats = analyze_by_type(results)
        
        all_results.append({
            'zeta': zeta,
            'delta_g': delta_g,
            'stats': stats,
        })
        
        print(f"ζ={zeta:.1f}: All={stats['all'][1]:.1f}%, "
              f"Dwarf={stats['dwarf'][1]:.1f}%, "
              f"Inter={stats['intermediate'][1]:.1f}%, "
              f"Massive={stats['massive'][1]:.1f}%")
    
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    print(f"\n{'ζ':<6} {'δg/g':<12} {'All %':<8} {'Dwarf %':<10} {'Inter %':<10} {'Massive %':<10} {'Mass Δmed':<10}")
    print("-" * 76)
    
    for r in all_results:
        s = r['stats']
        print(f"{r['zeta']:<6.1f} {r['delta_g']:<12.2e} "
              f"{s['all'][1]:<8.1f} {s['dwarf'][1]:<10.1f} "
              f"{s['intermediate'][1]:<10.1f} {s['massive'][1]:<10.1f} "
              f"{s['massive'][2]:<10.2f}")
    
    # Find best for massive spirals
    best_massive = max(all_results, key=lambda x: x['stats']['massive'][1])
    best_all = max(all_results, key=lambda x: x['stats']['all'][1])
    
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"\nBest for MASSIVE spirals: ζ = {best_massive['zeta']}")
    print(f"  Massive spiral success: {best_massive['stats']['massive'][1]:.1f}%")
    print(f"  Overall success: {best_massive['stats']['all'][1]:.1f}%")
    
    print(f"\nBest for OVERALL: ζ = {best_all['zeta']}")
    print(f"  Overall success: {best_all['stats']['all'][1]:.1f}%")
    print(f"  Massive spiral success: {best_all['stats']['massive'][1]:.1f}%")
    
    # Key comparison: ζ=0 (original) vs best ζ
    if all_results:
        orig = next((r for r in all_results if r['zeta'] == 0.0), None)
        if orig and best_massive:
            print("\n" + "=" * 70)
            print("IMPROVEMENT FROM COOPERATIVE CHANNELING")
            print("=" * 70)
            print(f"\nOriginal (ζ=0.0):")
            print(f"  Massive spiral success: {orig['stats']['massive'][1]:.1f}%")
            print(f"  Overall success: {orig['stats']['all'][1]:.1f}%")
            
            print(f"\nCooperative (ζ={best_massive['zeta']}):")
            print(f"  Massive spiral success: {best_massive['stats']['massive'][1]:.1f}%")
            print(f"  Overall success: {best_massive['stats']['all'][1]:.1f}%")
            
            delta_massive = best_massive['stats']['massive'][1] - orig['stats']['massive'][1]
            delta_all = best_massive['stats']['all'][1] - orig['stats']['all'][1]
            
            print(f"\nChange:")
            print(f"  Massive spirals: {delta_massive:+.1f}% points")
            print(f"  Overall: {delta_all:+.1f}% points")
            
            if delta_massive > 5:
                print("\n🎉 COOPERATIVE CHANNELING HELPS MASSIVE SPIRALS!")
            elif delta_massive < -5:
                print("\n⚠️ Cooperative channeling HURTS massive spirals")
            else:
                print("\n→ Minimal effect on massive spirals")


if __name__ == "__main__":
    main()
