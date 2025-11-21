"""
Analyze compact dwarfs: Are stars going faster than GR/baryons predict?

This checks whether the catastrophic failures have stars that are:
1. Going FASTER than baryons predict (need dark matter / modified gravity)
2. Going SLOWER than baryons predict (shouldn't happen in standard cosmology)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_integration.load_real_data import RealDataLoader

# The worst compact dwarf failures
problem_galaxies = ['UGC05750', 'UGC04305', 'NGC2976']

print("="*80)
print("COMPACT DWARF ANALYSIS: Do stars go FASTER than baryons predict?")
print("="*80)

loader = RealDataLoader()

for name in problem_galaxies:
    print(f"\n{'='*80}")
    print(f"Galaxy: {name}")
    print('='*80)
    
    try:
        gal = loader.load_rotmod_galaxy(name)
        
        r = gal['r']
        v_obs = gal['v_obs']
        v_err = gal['v_err']
        v_disk = gal['v_disk']
        v_gas = gal['v_gas']
        v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
        v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
        
        # Check if stars are going faster or slower than baryons
        residuals = v_obs - v_bar
        
        # Statistics
        mean_residual = np.mean(residuals)
        median_residual = np.median(residuals)
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        # Fractional difference
        frac_diff = residuals / v_bar * 100  # percent
        
        print(f"\nRotation curve data:")
        print(f"  N_points: {len(r)}")
        print(f"  v_obs range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
        print(f"  v_bar range: {v_bar.min():.1f} - {v_bar.max():.1f} km/s")
        
        print(f"\nResiduals (v_obs - v_bar):")
        print(f"  Mean: {mean_residual:+.1f} km/s")
        print(f"  Median: {median_residual:+.1f} km/s")
        print(f"  RMS: {rms_residual:.1f} km/s")
        print(f"  Fractional: {np.median(frac_diff):+.1f}% (median)")
        
        # Count how many points are faster vs slower
        n_faster = np.sum(residuals > 0)
        n_slower = np.sum(residuals < 0)
        
        print(f"\nPoint-by-point:")
        print(f"  {n_faster}/{len(r)} points have v_obs > v_bar (stars going FASTER)")
        print(f"  {n_slower}/{len(r)} points have v_obs < v_bar (stars going SLOWER)")
        
        # Detailed breakdown
        print(f"\n  Radius [kpc]  |  v_obs [km/s]  |  v_bar [km/s]  |  Residual [km/s]  |  Frac [%]")
        print(f"  {'-'*80}")
        for i in range(len(r)):
            print(f"  {r[i]:6.2f}       |  {v_obs[i]:7.1f}       |  {v_bar[i]:7.1f}       |  {residuals[i]:+7.1f}         |  {frac_diff[i]:+6.1f}")
        
        # Verdict
        if median_residual > 0:
            print(f"\n✓ YES: Stars ARE going faster than baryons predict (median +{median_residual:.1f} km/s)")
            print(f"  → This galaxy NEEDS dark matter or modified gravity")
            print(f"  → GPM is adding coherence, but adding TOO MUCH (overshooting)")
        else:
            print(f"\n✗ NO: Stars are NOT going faster (median {median_residual:.1f} km/s)")
            print(f"  → Baryons already over-predict → GPM makes it worse")
        
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("If compact dwarfs have stars going FASTER than baryons:")
print("  → They DO need extra mass/modified gravity")
print("  → But GPM is adding TOO MUCH coherence (overshooting)")
print("  → Solution: Stricter mass/compactness gate to turn OFF GPM for these")
print("\nIf compact dwarfs have stars going at BARYON speeds:")
print("  → They DON'T need dark matter")
print("  → GPM shouldn't be adding anything")
print("  → Hard floor (α_eff=0) should already handle this")
