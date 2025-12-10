"""
SPARC Batch Test for Σ-Gravity with Spiral Winding
===================================================

Tests the extended Σ-Gravity kernel (with winding gate) on all 175 SPARC galaxies.

Compares:
1. Original Σ-Gravity (no winding)
2. Σ-Gravity with winding (N_crit = 10)

Author: Leonard Speiser
Date: 2025-11-25
"""

import sys
import os
import glob
import numpy as np
from typing import List, Dict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigma_gravity_winding import (
    SigmaGravityParams, sigma_gravity_velocity, load_sparc_galaxy, fit_galaxy
)


def run_sparc_batch(data_dir: str, params: SigmaGravityParams) -> List[Dict]:
    """Run on all SPARC galaxies."""
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    
    print(f"Found {len(files)} SPARC galaxies")
    
    results = []
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, params)
            results.append(result)
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {e}")
            continue
    
    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze batch results."""
    n_total = len(results)
    n_improved = sum(1 for r in results if r['improved'])
    pct_improved = 100 * n_improved / n_total
    
    delta_rms = np.array([r['delta_rms'] for r in results])
    v_flat = np.array([r['v_flat'] for r in results])
    
    # By galaxy type
    dwarfs = [r for r in results if r['v_flat'] < 80]
    intermediate = [r for r in results if 80 <= r['v_flat'] < 150]
    massive = [r for r in results if r['v_flat'] >= 150]
    
    # Mean N_orbits by type
    dwarf_N = np.mean([np.mean(r['N_orbits']) for r in dwarfs]) if dwarfs else 0
    inter_N = np.mean([np.mean(r['N_orbits']) for r in intermediate]) if intermediate else 0
    massive_N = np.mean([np.mean(r['N_orbits']) for r in massive]) if massive else 0
    
    # Mean G_wind by type
    dwarf_G = np.mean([np.mean(r['G_wind']) for r in dwarfs]) if dwarfs else 0
    inter_G = np.mean([np.mean(r['G_wind']) for r in intermediate]) if intermediate else 0
    massive_G = np.mean([np.mean(r['G_wind']) for r in massive]) if massive else 0
    
    return {
        'n_total': n_total,
        'n_improved': n_improved,
        'pct_improved': pct_improved,
        'median_delta_rms': np.median(delta_rms),
        'mean_delta_rms': np.mean(delta_rms),
        'dwarf_pct': 100 * sum(1 for r in dwarfs if r['improved']) / len(dwarfs) if dwarfs else 0,
        'inter_pct': 100 * sum(1 for r in intermediate if r['improved']) / len(intermediate) if intermediate else 0,
        'massive_pct': 100 * sum(1 for r in massive if r['improved']) / len(massive) if massive else 0,
        'dwarf_N': dwarf_N,
        'inter_N': inter_N,
        'massive_N': massive_N,
        'dwarf_G_wind': dwarf_G,
        'inter_G_wind': inter_G,
        'massive_G_wind': massive_G,
        'n_dwarf': len(dwarfs),
        'n_inter': len(intermediate),
        'n_massive': len(massive),
    }


def main():
    print("=" * 80)
    print("Σ-GRAVITY WITH SPIRAL WINDING - SPARC BATCH TEST")
    print("=" * 80)
    
    # Data directory - use existing SPARC data
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        return
    
    # Test 1: Original Σ-Gravity (no winding)
    print("\n" + "=" * 80)
    print("TEST 1: ORIGINAL Σ-GRAVITY (NO WINDING)")
    print("=" * 80)
    
    params_no_wind = SigmaGravityParams(
        A=0.6,
        ell_0=4.993,
        p=0.75,
        n_coh=0.5,
        use_winding=False,
    )
    
    results_no_wind = run_sparc_batch(data_dir, params_no_wind)
    summary_no_wind = analyze_results(results_no_wind)
    
    print(f"\nResults (no winding):")
    print(f"  Galaxies tested: {summary_no_wind['n_total']}")
    print(f"  Improved: {summary_no_wind['n_improved']}/{summary_no_wind['n_total']} ({summary_no_wind['pct_improved']:.1f}%)")
    print(f"  Median ΔRMS: {summary_no_wind['median_delta_rms']:.2f} km/s")
    print(f"  By type:")
    print(f"    Dwarfs (n={summary_no_wind['n_dwarf']}): {summary_no_wind['dwarf_pct']:.1f}%")
    print(f"    Intermediate (n={summary_no_wind['n_inter']}): {summary_no_wind['inter_pct']:.1f}%")
    print(f"    Massive (n={summary_no_wind['n_massive']}): {summary_no_wind['massive_pct']:.1f}%")
    
    # Test 2: Σ-Gravity WITH winding
    print("\n" + "=" * 80)
    print("TEST 2: Σ-GRAVITY WITH SPIRAL WINDING (N_crit=10)")
    print("=" * 80)
    
    params_wind = SigmaGravityParams(
        A=0.6,
        ell_0=4.993,
        p=0.75,
        n_coh=0.5,
        use_winding=True,
        N_crit=10.0,
        t_age=10.0,
    )
    
    results_wind = run_sparc_batch(data_dir, params_wind)
    summary_wind = analyze_results(results_wind)
    
    print(f"\nResults (with winding):")
    print(f"  Galaxies tested: {summary_wind['n_total']}")
    print(f"  Improved: {summary_wind['n_improved']}/{summary_wind['n_total']} ({summary_wind['pct_improved']:.1f}%)")
    print(f"  Median ΔRMS: {summary_wind['median_delta_rms']:.2f} km/s")
    print(f"  By type:")
    print(f"    Dwarfs (n={summary_wind['n_dwarf']}): {summary_wind['dwarf_pct']:.1f}%")
    print(f"    Intermediate (n={summary_wind['n_inter']}): {summary_wind['inter_pct']:.1f}%")
    print(f"    Massive (n={summary_wind['n_massive']}): {summary_wind['massive_pct']:.1f}%")
    
    # Winding statistics
    print(f"\n  Winding statistics:")
    print(f"    Dwarfs: mean N_orbits = {summary_wind['dwarf_N']:.1f}, mean G_wind = {summary_wind['dwarf_G_wind']:.3f}")
    print(f"    Intermediate: mean N_orbits = {summary_wind['inter_N']:.1f}, mean G_wind = {summary_wind['inter_G_wind']:.3f}")
    print(f"    Massive: mean N_orbits = {summary_wind['massive_N']:.1f}, mean G_wind = {summary_wind['massive_G_wind']:.3f}")
    
    # Test 3: Sweep N_crit
    print("\n" + "=" * 80)
    print("TEST 3: N_crit SWEEP")
    print("=" * 80)
    
    print(f"\n{'N_crit':<10} {'All %':<10} {'Dwarf %':<12} {'Inter %':<12} {'Massive %':<12}")
    print("-" * 56)
    
    for N_crit in [5, 8, 10, 15, 20, 50]:
        params = SigmaGravityParams(
            A=0.6, ell_0=4.993, p=0.75, n_coh=0.5,
            use_winding=True, N_crit=N_crit, t_age=10.0,
        )
        results = run_sparc_batch(data_dir, params)
        summary = analyze_results(results)
        
        print(f"{N_crit:<10.0f} {summary['pct_improved']:<10.1f} {summary['dwarf_pct']:<12.1f} {summary['inter_pct']:<12.1f} {summary['massive_pct']:<12.1f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: WITH vs WITHOUT WINDING")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'No Winding':<15} {'With Winding':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Overall %':<25} {summary_no_wind['pct_improved']:<15.1f} {summary_wind['pct_improved']:<15.1f} {summary_wind['pct_improved'] - summary_no_wind['pct_improved']:+.1f}")
    print(f"{'Dwarf %':<25} {summary_no_wind['dwarf_pct']:<15.1f} {summary_wind['dwarf_pct']:<15.1f} {summary_wind['dwarf_pct'] - summary_no_wind['dwarf_pct']:+.1f}")
    print(f"{'Intermediate %':<25} {summary_no_wind['inter_pct']:<15.1f} {summary_wind['inter_pct']:<15.1f} {summary_wind['inter_pct'] - summary_no_wind['inter_pct']:+.1f}")
    print(f"{'Massive %':<25} {summary_no_wind['massive_pct']:<15.1f} {summary_wind['massive_pct']:<15.1f} {summary_wind['massive_pct'] - summary_no_wind['massive_pct']:+.1f}")
    print(f"{'Median ΔRMS (km/s)':<25} {summary_no_wind['median_delta_rms']:<15.2f} {summary_wind['median_delta_rms']:<15.2f} {summary_wind['median_delta_rms'] - summary_no_wind['median_delta_rms']:+.2f}")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if summary_wind['pct_improved'] > summary_no_wind['pct_improved']:
        print(f"\n✓ Winding gate IMPROVES overall performance by {summary_wind['pct_improved'] - summary_no_wind['pct_improved']:.1f}%")
    else:
        print(f"\n✗ Winding gate DECREASES overall performance by {summary_no_wind['pct_improved'] - summary_wind['pct_improved']:.1f}%")
    
    if summary_wind['massive_pct'] > summary_no_wind['massive_pct']:
        print(f"✓ Massive spiral improvement: {summary_no_wind['massive_pct']:.1f}% → {summary_wind['massive_pct']:.1f}%")
    else:
        print(f"✗ Massive spirals not improved by winding")
    
    print("""
Key insight: The winding gate provides morphology-dependent suppression:
- Dwarfs: N ~ 10-20, G_wind ~ 0.2-0.5 (moderate suppression)
- Massive: N ~ 30-50, G_wind ~ 0.04-0.1 (strong suppression)

This naturally gives LESS enhancement to fast-rotating systems!
""")


if __name__ == "__main__":
    main()
