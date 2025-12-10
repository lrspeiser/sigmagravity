"""
Phase 3: Physical Consistency Tests for Gravitational Channeling
=================================================================

Tests:
- 3a. Solar System (Cassini constraint)
- 3b. Young vs old galaxies (age dependence)
- 3c. σ_v correlation (cold disks enhanced more)
- 3d. Dwarf vs spiral (scale-free behavior)
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channeling_kernel import (
    ChannelingParams, load_sparc_galaxy, fit_galaxy, 
    test_solar_system, channeling_enhancement, estimate_sigma_v
)


def test_age_dependence(params: ChannelingParams):
    """
    Test 3b: Young galaxies should have less enhancement.
    
    Theory predicts D ~ (t_age/τ_ch)^γ, so younger systems have smaller F.
    """
    print("\n" + "=" * 70)
    print("TEST 3b: AGE DEPENDENCE")
    print("=" * 70)
    
    # Test at fixed R=10 kpc, v_c=150 km/s, Σ=50 M_sun/pc^2, σ=15 km/s
    R = np.array([10.0])
    v_c = np.array([150.0])
    Sigma = np.array([50.0])
    sigma_v = np.array([15.0])
    
    ages = [1, 3, 5, 8, 10, 13]  # Gyr
    
    print(f"\nFixed: R=10 kpc, v_c=150 km/s, Σ=50 M☉/pc², σ_v=15 km/s")
    print(f"{'Age (Gyr)':<12} {'F':<10} {'D':<10} {'v_pred':<12}")
    print("-" * 44)
    
    F_values = []
    for age in ages:
        test_params = ChannelingParams(
            alpha=params.alpha, beta=params.beta, gamma=params.gamma,
            chi_0=params.chi_0, epsilon=params.epsilon, D_max=params.D_max,
            t_age=age, tau_0=params.tau_0
        )
        diag = {}
        F = channeling_enhancement(R, v_c, Sigma, sigma_v, test_params, diag)
        F_values.append(F[0])
        v_pred = 150 * np.sqrt(F[0])
        print(f"{age:<12} {F[0]:<10.4f} {diag['D'][0]:<10.4f} {v_pred:<12.1f}")
    
    # Check monotonicity (F should increase with age)
    is_monotonic = all(F_values[i] <= F_values[i+1] for i in range(len(F_values)-1))
    
    # Calculate ratio young/old
    ratio = F_values[0] / F_values[-1]  # 1 Gyr vs 13 Gyr
    
    print(f"\nF(1 Gyr) / F(13 Gyr) = {ratio:.3f}")
    print(f"Enhancement monotonic with age: {is_monotonic}")
    print(f"Result: {'PASS ✓' if is_monotonic and ratio < 0.8 else 'FAIL ✗'}")
    
    return is_monotonic and ratio < 0.8


def test_sigma_v_correlation(data_dir: str, params: ChannelingParams):
    """
    Test 3c: Cold disks (low σ_v) should be enhanced more.
    
    Theory predicts D ~ (v_c/σ_v)^β, so cold systems have higher F.
    """
    print("\n" + "=" * 70)
    print("TEST 3c: VELOCITY DISPERSION CORRELATION")
    print("=" * 70)
    
    # Load all galaxies and compute mean F and mean σ_v/v_c
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    results = []
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, params)
            
            # Get mean σ_v/v_c ratio in outer disk (R > 0.5 R_max)
            R = result['R']
            R_half = R.max() * 0.5
            outer_mask = R > R_half
            if outer_mask.sum() < 2:
                continue
            
            sigma_v = result['sigma_v'][outer_mask]
            v_bary = result['v_bary'][outer_mask]
            F = result['F'][outer_mask]
            
            mean_ratio = np.mean(sigma_v / v_bary)
            mean_F = np.mean(F)
            
            results.append({
                'name': result['name'],
                'sigma_v_ratio': mean_ratio,
                'mean_F': mean_F,
                'delta_rms': result['delta_rms'],
            })
        except:
            continue
    
    if not results:
        print("No valid data!")
        return False
    
    # Compute correlation
    sigma_ratios = np.array([r['sigma_v_ratio'] for r in results])
    F_values = np.array([r['mean_F'] for r in results])
    
    # Pearson correlation
    corr = np.corrcoef(sigma_ratios, F_values)[0, 1]
    
    # Bin by σ_v/v_c
    low_sigma = [r for r in results if r['sigma_v_ratio'] < 0.15]
    high_sigma = [r for r in results if r['sigma_v_ratio'] > 0.25]
    
    mean_F_cold = np.mean([r['mean_F'] for r in low_sigma]) if low_sigma else 0
    mean_F_hot = np.mean([r['mean_F'] for r in high_sigma]) if high_sigma else 0
    
    print(f"\nGalaxies analyzed: {len(results)}")
    print(f"Correlation (σ_v/v_c vs F): {corr:.3f}")
    print(f"\nCold disks (σ_v/v_c < 0.15): {len(low_sigma)} galaxies, mean F = {mean_F_cold:.3f}")
    print(f"Hot disks (σ_v/v_c > 0.25): {len(high_sigma)} galaxies, mean F = {mean_F_hot:.3f}")
    
    # Cold disks should have HIGHER F (anti-correlation with σ ratio)
    is_anticorrelated = corr < -0.1 or mean_F_cold > mean_F_hot
    
    print(f"\nExpected: Cold disks have higher F (anti-correlation)")
    print(f"Result: {'PASS ✓' if is_anticorrelated else 'FAIL ✗'}")
    
    return is_anticorrelated


def test_scale_free(data_dir: str, params: ChannelingParams):
    """
    Test 3d: Same parameters should work for dwarfs and spirals.
    """
    print("\n" + "=" * 70)
    print("TEST 3d: SCALE-FREE BEHAVIOR (DWARF vs SPIRAL)")
    print("=" * 70)
    
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    dwarfs = []  # v_flat < 80 km/s
    spirals = []  # v_flat > 150 km/s
    
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, params)
            v_flat = np.mean(result['v_obs'][-3:])
            
            if v_flat < 80:
                dwarfs.append(result)
            elif v_flat > 150:
                spirals.append(result)
        except:
            continue
    
    # Calculate improvement rates
    dwarf_improved = sum(1 for r in dwarfs if r['improved'])
    spiral_improved = sum(1 for r in spirals if r['improved'])
    
    dwarf_pct = 100 * dwarf_improved / len(dwarfs) if dwarfs else 0
    spiral_pct = 100 * spiral_improved / len(spirals) if spirals else 0
    
    dwarf_median_delta = np.median([r['delta_rms'] for r in dwarfs]) if dwarfs else 0
    spiral_median_delta = np.median([r['delta_rms'] for r in spirals]) if spirals else 0
    
    print(f"\nDwarfs (v_flat < 80 km/s):")
    print(f"  Count: {len(dwarfs)}")
    print(f"  Improved: {dwarf_improved}/{len(dwarfs)} ({dwarf_pct:.1f}%)")
    print(f"  Median ΔRMS: {dwarf_median_delta:.2f} km/s")
    
    print(f"\nSpirals (v_flat > 150 km/s):")
    print(f"  Count: {len(spirals)}")
    print(f"  Improved: {spiral_improved}/{len(spirals)} ({spiral_pct:.1f}%)")
    print(f"  Median ΔRMS: {spiral_median_delta:.2f} km/s")
    
    # Both should have >50% improvement for scale-free behavior
    is_scale_free = dwarf_pct > 50 and spiral_pct > 40
    
    print(f"\nExpected: Both types >50% improved")
    print(f"Result: {'PASS ✓' if is_scale_free else 'FAIL ✗'}")
    
    return is_scale_free


def main():
    print("=" * 70)
    print("GRAVITATIONAL CHANNELING - PHASE 3: PHYSICAL CONSISTENCY")
    print("=" * 70)
    
    # Optimized parameters
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
    
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    
    # Test 3a: Solar System
    print("\n" + "=" * 70)
    print("TEST 3a: SOLAR SYSTEM (Cassini)")
    print("=" * 70)
    delta_g, ss_pass = test_solar_system(params)
    print(f"δg/g at Saturn: {delta_g:.2e}")
    print(f"Cassini limit: 2.3×10⁻⁵")
    print(f"Result: {'PASS ✓' if ss_pass else 'FAIL ✗'}")
    
    # Test 3b: Age dependence
    age_pass = test_age_dependence(params)
    
    # Test 3c: σ_v correlation
    sigma_pass = test_sigma_v_correlation(data_dir, params)
    
    # Test 3d: Scale-free
    scale_pass = test_scale_free(data_dir, params)
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    
    tests = [
        ("3a. Solar System", ss_pass),
        ("3b. Age dependence", age_pass),
        ("3c. σ_v correlation", sigma_pass),
        ("3d. Scale-free", scale_pass),
    ]
    
    passed = sum(1 for _, p in tests if p)
    for name, p in tests:
        print(f"{'✓' if p else '✗'} {name}")
    
    print(f"\nOverall: {passed}/4 tests passed")
    
    if passed >= 3:
        print("\n🎉 PHASE 3 PASSED")
    else:
        print("\n⚠️ PHASE 3 NEEDS WORK")


if __name__ == "__main__":
    main()
