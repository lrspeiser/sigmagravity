#!/usr/bin/env python3
"""
DEMONSTRATION: Σ-GRAVITY HYPOTHESIS TESTS
==========================================

This script demonstrates the hypothesis tests using SIMULATED data
that mimics what real SPARC/Gaia data would look like.

Use this to understand the methodology before running on real data.

The TRUE underlying model in the simulation is:
  K = tanh((x - 0.75) / 0.3) + 1

We test if various hypotheses can be distinguished.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, ks_2samp
import json
from pathlib import Path

np.random.seed(42)

print("="*70)
print("  Σ-GRAVITY HYPOTHESIS TESTS - DEMONSTRATION")
print("  (Using simulated data to show methodology)")
print("="*70)

# =============================================================================
# GENERATE SIMULATED SPARC-LIKE DATA
# =============================================================================

def generate_sparc_like_data(n_galaxies=150):
    """Generate data that mimics SPARC galaxy sample."""
    
    galaxies = []
    
    for i in range(n_galaxies):
        # Random galaxy properties
        R_disk = np.random.uniform(1.5, 5.0)  # kpc
        M_disk = 10**np.random.uniform(9.5, 11)  # M_sun
        SB_disk = np.random.uniform(19, 24)  # mag/arcsec² (surface brightness)
        v_flat = np.random.uniform(80, 280)  # km/s
        
        # LSB galaxies have larger R_disk on average
        if SB_disk > 22:  # LSB
            R_disk *= 1.3
            M_disk *= 0.5
        
        # Generate rotation curve
        n_points = np.random.randint(15, 40)
        R = np.linspace(0.3, np.random.uniform(3, 6) * R_disk, n_points)
        
        # Baryonic velocity (exponential disk)
        x = R / R_disk
        G = 4.3e-6  # kpc (km/s)² / M_sun
        M_enc = M_disk * (1 - (1 + x) * np.exp(-x))
        v_bar = np.sqrt(G * M_enc / np.maximum(R, 0.1))
        
        # TRUE enhancement model: K = tanh((x - 0.75)/0.3) + 1
        # But add scatter that correlates slightly with SB
        x_c_true = 0.75 + 0.05 * (SB_disk - 21.5) / 2.5  # Slight SB dependence
        K_true = np.tanh((x - x_c_true) / 0.3) + 1.0
        K_true += np.random.normal(0, 0.1, len(R))
        K_true = np.clip(K_true, 0.2, 5)
        
        # Observed velocity
        v_obs = v_bar * np.sqrt(K_true)
        v_obs += np.random.normal(0, 5, len(R))  # Measurement error
        
        # Derived quantities
        Sigma = M_disk / (2 * np.pi * R_disk**2) * np.exp(-x)  # Surface density
        g_bar = v_bar**2 / np.maximum(R, 0.1)  # Baryonic acceleration
        sigma_v = 50 * np.exp(-R / (2 * R_disk)) + 10  # Velocity dispersion
        
        galaxies.append({
            'name': f'SimGal_{i:03d}',
            'R_disk': R_disk,
            'M_disk': M_disk,
            'SB_disk': SB_disk,
            'v_flat': v_flat,
            'R': R,
            'v_obs': v_obs,
            'v_bar': v_bar,
            'K': K_true,
            'x': x,
            'Sigma': Sigma,
            'g_bar': g_bar,
            'sigma_v': sigma_v,
        })
    
    return galaxies


# =============================================================================
# TEST 1: LSB vs HSB AT SAME x
# =============================================================================

def test1_lsb_hsb(galaxies):
    """Test if K differs between LSB and HSB at same x."""
    
    print("\n" + "="*70)
    print("  TEST 1: LSB vs HSB at same x = R/R_disk")
    print("="*70)
    
    # Classify
    lsb = [g for g in galaxies if g['SB_disk'] > 22]
    hsb = [g for g in galaxies if g['SB_disk'] < 21]
    
    print(f"\n  LSB galaxies (SB > 22): {len(lsb)}")
    print(f"  HSB galaxies (SB < 21): {len(hsb)}")
    
    # Bin by x
    x_bins = np.linspace(0.5, 4.0, 15)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    
    lsb_K = {i: [] for i in range(len(x_centers))}
    hsb_K = {i: [] for i in range(len(x_centers))}
    
    for g in lsb:
        for k, xi in zip(g['K'], g['x']):
            idx = np.digitize(xi, x_bins) - 1
            if 0 <= idx < len(x_centers):
                lsb_K[idx].append(k)
    
    for g in hsb:
        for k, xi in zip(g['K'], g['x']):
            idx = np.digitize(xi, x_bins) - 1
            if 0 <= idx < len(x_centers):
                hsb_K[idx].append(k)
    
    # Compare
    print(f"\n  {'x bin':<12s} {'LSB K':<15s} {'HSB K':<15s} {'Diff':<12s} {'p-value':<10s}")
    print("  " + "-"*65)
    
    results = []
    for i, xc in enumerate(x_centers):
        if len(lsb_K[i]) > 5 and len(hsb_K[i]) > 5:
            lsb_med = np.median(lsb_K[i])
            hsb_med = np.median(hsb_K[i])
            lsb_err = np.std(lsb_K[i]) / np.sqrt(len(lsb_K[i]))
            hsb_err = np.std(hsb_K[i]) / np.sqrt(len(hsb_K[i]))
            diff = lsb_med - hsb_med
            
            _, p = ks_2samp(lsb_K[i], hsb_K[i])
            
            print(f"  {xc:<12.2f} {lsb_med:.3f}±{lsb_err:.3f}    "
                  f"{hsb_med:.3f}±{hsb_err:.3f}    {diff:+.3f}      {p:.3f}")
            
            results.append({'x': xc, 'diff': diff, 'p': p})
    
    # Conclusion
    avg_diff = np.mean([r['diff'] for r in results])
    sig_diff = np.mean([r['p'] < 0.05 for r in results])
    
    print(f"\n  RESULTS:")
    print(f"  Average K difference (LSB - HSB): {avg_diff:+.4f}")
    print(f"  Fraction with p < 0.05: {sig_diff:.1%}")
    
    if abs(avg_diff) < 0.15:
        print(f"\n  → K is SIMILAR for LSB and HSB at same x")
        print(f"  → SUPPORTS: Geometry hypothesis (x = R/R_disk matters)")
        print(f"  → DISFAVORS: Surface brightness Σ hypothesis")
    else:
        print(f"\n  → K DIFFERS between LSB and HSB at same x")
        print(f"  → SUPPORTS: Σ-dependent hypothesis")
    
    return {'avg_diff': avg_diff, 'results': results}


# =============================================================================
# TEST 2: RAR vs GEOMETRY
# =============================================================================

def test2_rar_geometry(galaxies):
    """Test if K depends more on g_bar (RAR) or x (geometry)."""
    
    print("\n" + "="*70)
    print("  TEST 2: RAR vs Geometry (Partial Correlation Analysis)")
    print("="*70)
    
    # Collect all points
    all_K = []
    all_x = []
    all_g = []
    
    for g in galaxies:
        for i in range(len(g['R'])):
            if g['g_bar'][i] > 10:  # Avoid very low g_bar
                all_K.append(g['K'][i])
                all_x.append(g['x'][i])
                all_g.append(g['g_bar'][i])
    
    K = np.array(all_K)
    x = np.array(all_x)
    log_g = np.log10(all_g)
    
    print(f"\n  Total data points: {len(K)}")
    
    # Simple correlations
    r_Kx, p_Kx = pearsonr(K, x)
    r_Kg, p_Kg = pearsonr(K, log_g)
    
    print(f"\n  Simple correlations:")
    print(f"    K vs x:      r = {r_Kx:+.4f}  (p = {p_Kx:.2e})")
    print(f"    K vs log(g): r = {r_Kg:+.4f}  (p = {p_Kg:.2e})")
    
    # Partial correlations (residual method)
    from numpy.linalg import lstsq
    
    # K vs x, controlling for g
    A1 = np.column_stack([np.ones(len(K)), log_g])
    coef_K1, _, _, _ = lstsq(A1, K, rcond=None)
    K_resid1 = K - A1 @ coef_K1
    coef_x1, _, _, _ = lstsq(A1, x, rcond=None)
    x_resid1 = x - A1 @ coef_x1
    r_partial_Kx, _ = pearsonr(K_resid1, x_resid1)
    
    # K vs g, controlling for x
    A2 = np.column_stack([np.ones(len(K)), x])
    coef_K2, _, _, _ = lstsq(A2, K, rcond=None)
    K_resid2 = K - A2 @ coef_K2
    coef_g2, _, _, _ = lstsq(A2, log_g, rcond=None)
    g_resid2 = log_g - A2 @ coef_g2
    r_partial_Kg, _ = pearsonr(K_resid2, g_resid2)
    
    print(f"\n  Partial correlations:")
    print(f"    K vs x (controlling g):      r_partial = {r_partial_Kx:+.4f}")
    print(f"    K vs log(g) (controlling x): r_partial = {r_partial_Kg:+.4f}")
    
    # Conclusion
    print(f"\n  RESULTS:")
    if abs(r_partial_Kx) > abs(r_partial_Kg) + 0.1:
        print(f"  → x has STRONGER partial correlation with K")
        print(f"  → SUPPORTS: Geometry hypothesis (x = R/R_disk)")
        print(f"  → DISFAVORS: RAR (g_bar dependence)")
        winner = "geometry"
    elif abs(r_partial_Kg) > abs(r_partial_Kx) + 0.1:
        print(f"  → g_bar has STRONGER partial correlation with K")
        print(f"  → SUPPORTS: RAR hypothesis")
        print(f"  → DISFAVORS: Geometry hypothesis")
        winner = "RAR"
    else:
        print(f"  → Both have similar partial correlations")
        print(f"  → INCONCLUSIVE: Need more data or different test")
        winner = "inconclusive"
    
    return {
        'r_Kx': r_Kx, 'r_Kg': r_Kg,
        'r_partial_Kx': r_partial_Kx, 'r_partial_Kg': r_partial_Kg,
        'winner': winner
    }


# =============================================================================
# TEST 3: SIMULATED MW THICK/THIN DISK
# =============================================================================

def test3_mw_thick_thin():
    """Simulate MW thick/thin disk test."""
    
    print("\n" + "="*70)
    print("  TEST 3: MW Thick vs Thin Disk (Simulated)")
    print("="*70)
    
    # Simulate MW-like data
    R_sun = 8.2  # kpc
    R_disk_thin = 2.5  # kpc
    R_disk_thick = 3.5  # kpc
    
    # Generate stars
    n_stars = 5000
    
    # Thin disk: low dispersion, high metallicity
    R_thin = np.abs(np.random.normal(R_sun, 2, n_stars))
    sigma_thin = 25 * np.exp(-R_thin / 8) + 10  # km/s
    FeH_thin = np.random.normal(-0.1, 0.2, n_stars)
    
    # Thick disk: high dispersion, low metallicity
    R_thick = np.abs(np.random.normal(R_sun, 3, n_stars))
    sigma_thick = 50 * np.exp(-R_thick / 8) + 20  # km/s
    FeH_thick = np.random.normal(-0.5, 0.3, n_stars)
    
    # TRUE K model: depends on x = R/R_disk (different R_disk for thick/thin)
    # Add small sigma_v dependence to test
    x_thin = R_thin / R_disk_thin
    x_thick = R_thick / R_disk_thick
    
    K_thin = np.tanh((x_thin - 0.75) / 0.3) + 1.0
    K_thick = np.tanh((x_thick - 0.75) / 0.3) + 1.0
    
    # Add noise
    K_thin += np.random.normal(0, 0.05, n_stars)
    K_thick += np.random.normal(0, 0.08, n_stars)  # More scatter in thick
    
    print(f"\n  Simulated stars:")
    print(f"    Thin disk: {n_stars} stars, ⟨σ_v⟩ = {np.mean(sigma_thin):.1f} km/s")
    print(f"    Thick disk: {n_stars} stars, ⟨σ_v⟩ = {np.mean(sigma_thick):.1f} km/s")
    
    # Compare K at same R
    R_bins = np.linspace(4, 12, 9)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    
    print(f"\n  {'R (kpc)':<10s} {'Thin K':<15s} {'Thick K':<15s} {'Diff':<10s}")
    print("  " + "-"*50)
    
    results = []
    for i in range(len(R_centers)):
        mask_thin = (R_thin >= R_bins[i]) & (R_thin < R_bins[i+1])
        mask_thick = (R_thick >= R_bins[i]) & (R_thick < R_bins[i+1])
        
        if mask_thin.sum() > 20 and mask_thick.sum() > 20:
            K_thin_bin = K_thin[mask_thin]
            K_thick_bin = K_thick[mask_thick]
            
            med_thin = np.median(K_thin_bin)
            med_thick = np.median(K_thick_bin)
            err_thin = np.std(K_thin_bin) / np.sqrt(len(K_thin_bin))
            err_thick = np.std(K_thick_bin) / np.sqrt(len(K_thick_bin))
            
            diff = med_thick - med_thin
            
            print(f"  {R_centers[i]:<10.1f} {med_thin:.3f}±{err_thin:.3f}    "
                  f"{med_thick:.3f}±{err_thick:.3f}    {diff:+.3f}")
            
            results.append({'R': R_centers[i], 'K_thin': med_thin, 'K_thick': med_thick, 'diff': diff})
    
    # Conclusion
    avg_diff = np.mean([r['diff'] for r in results])
    
    print(f"\n  RESULTS:")
    print(f"  Average K difference (thick - thin): {avg_diff:+.4f}")
    
    if abs(avg_diff) < 0.1:
        print(f"\n  → K is SIMILAR for thick and thin disk at same R")
        print(f"  → But thick disk has larger R_disk, so x differs!")
        print(f"  → SUPPORTS: Geometry hypothesis if K same at same x")
    else:
        print(f"\n  → K DIFFERS between thick and thin disk")
        print(f"  → Could support σ_v hypothesis")
    
    return {'avg_diff': avg_diff, 'results': results}


# =============================================================================
# TEST 4: BARRED vs UNBARRED (SIMULATED)
# =============================================================================

def test4_barred(galaxies):
    """Test barred vs unbarred (simulated classification)."""
    
    print("\n" + "="*70)
    print("  TEST 4: Barred vs Unbarred Galaxies (Simulated)")
    print("="*70)
    
    # Randomly classify 30% as barred
    np.random.shuffle(galaxies)
    barred = galaxies[:int(0.3 * len(galaxies))]
    unbarred = galaxies[int(0.3 * len(galaxies)):]
    
    print(f"\n  Barred galaxies: {len(barred)}")
    print(f"  Unbarred galaxies: {len(unbarred)}")
    
    # Compare K(x) profiles
    x_bins = np.linspace(0.5, 3.0, 11)
    
    barred_K = {i: [] for i in range(len(x_bins)-1)}
    unbarred_K = {i: [] for i in range(len(x_bins)-1)}
    
    for g in barred:
        for k, xi in zip(g['K'], g['x']):
            idx = np.digitize(xi, x_bins) - 1
            if 0 <= idx < len(x_bins)-1:
                barred_K[idx].append(k)
    
    for g in unbarred:
        for k, xi in zip(g['K'], g['x']):
            idx = np.digitize(xi, x_bins) - 1
            if 0 <= idx < len(x_bins)-1:
                unbarred_K[idx].append(k)
    
    # KS test at each x
    print(f"\n  {'x bin':<15s} {'Barred K':<12s} {'Unbarred K':<12s} {'KS p-value':<10s}")
    print("  " + "-"*55)
    
    p_values = []
    for i in range(len(x_bins)-1):
        if len(barred_K[i]) > 10 and len(unbarred_K[i]) > 10:
            bar_med = np.median(barred_K[i])
            unbar_med = np.median(unbarred_K[i])
            _, p = ks_2samp(barred_K[i], unbarred_K[i])
            
            x_lo, x_hi = x_bins[i], x_bins[i+1]
            print(f"  [{x_lo:.1f}, {x_hi:.1f}]      {bar_med:.3f}        {unbar_med:.3f}        {p:.3f}")
            p_values.append(p)
    
    # Conclusion
    sig_frac = np.mean([p < 0.05 for p in p_values])
    
    print(f"\n  RESULTS:")
    print(f"  Fraction of bins with p < 0.05: {sig_frac:.1%}")
    
    if sig_frac < 0.2:
        print(f"\n  → No significant difference between barred and unbarred")
        print(f"  → SUPPORTS: Global geometry matters, not local structure")
    else:
        print(f"\n  → Barred galaxies differ from unbarred")
        print(f"  → Local structure may matter")
    
    return {'sig_fraction': sig_frac}


# =============================================================================
# TEST 5: ELLIPTICALS (SIMULATED)
# =============================================================================

def test5_ellipticals():
    """Simulate test with elliptical galaxies."""
    
    print("\n" + "="*70)
    print("  TEST 5: Elliptical Galaxies (Simulated)")
    print("="*70)
    
    # Ellipticals: spheroidal, no disk
    # Model: K might follow different pattern
    
    n_ellipticals = 30
    
    print(f"\n  Simulating {n_ellipticals} elliptical galaxies...")
    print(f"  (Spheroidal geometry, no disk R_disk)")
    
    results = []
    for i in range(n_ellipticals):
        R_eff = np.random.uniform(2, 10)  # kpc, effective radius
        M_star = 10**np.random.uniform(10, 12)  # M_sun
        
        # Generate profile
        R = np.linspace(0.3, 5 * R_eff, 20)
        x = R / R_eff
        
        # Baryonic from de Vaucouleurs profile (simplified)
        G = 4.3e-6
        M_enc = M_star * (1 - np.exp(-x))  # Simplified
        v_bar = np.sqrt(G * M_enc / R)
        
        # HYPOTHESIS TEST:
        # If disk geometry required: ellipticals should NOT follow tanh(x-0.75)
        # Simulate: ellipticals follow RAR instead
        
        g_bar = v_bar**2 / R
        g_dagger = 1.2e-10 * 3.086e16  # m/s² in (km/s)²/kpc units
        K_rar = 1.0 / (1 - np.exp(-np.sqrt(g_bar / g_dagger)))
        K_rar = np.clip(K_rar, 1, 10)
        K_rar += np.random.normal(0, 0.1, len(R))
        
        # Compare to disk prediction
        K_disk_pred = np.tanh((x - 0.75) / 0.3) + 1.0
        
        results.append({
            'R_eff': R_eff,
            'R': R,
            'x': x,
            'K_observed': K_rar,
            'K_disk_pred': K_disk_pred,
        })
    
    # Compare observed K to disk prediction
    all_K_obs = np.concatenate([r['K_observed'] for r in results])
    all_K_pred = np.concatenate([r['K_disk_pred'] for r in results])
    all_x = np.concatenate([r['x'] for r in results])
    
    # Residuals
    resid = all_K_obs - all_K_pred
    rms_resid = np.sqrt(np.mean(resid**2))
    
    # Does disk model fit?
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((all_K_obs - np.mean(all_K_obs))**2)
    r2_disk = 1 - ss_res / ss_tot
    
    print(f"\n  Disk model (tanh) fit to ellipticals:")
    print(f"    R² = {r2_disk:.3f}")
    print(f"    RMS residual = {rms_resid:.3f}")
    
    print(f"\n  RESULTS:")
    if r2_disk > 0.7:
        print(f"  → Ellipticals FOLLOW disk-like K(x) pattern")
        print(f"  → Surprising! Geometry hypothesis may be more general")
    else:
        print(f"  → Ellipticals do NOT follow disk K(x) pattern")
        print(f"  → SUPPORTS: Disk geometry specifically required")
        print(f"  → Ellipticals may follow RAR instead")
    
    return {'r2_disk': r2_disk, 'rms_resid': rms_resid}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all demonstration tests."""
    
    print("\n" + "="*70)
    print("  GENERATING SIMULATED DATA")
    print("="*70)
    
    galaxies = generate_sparc_like_data(n_galaxies=150)
    print(f"\n  Generated {len(galaxies)} simulated galaxies")
    
    # Run all tests
    results = {}
    
    results['test1'] = test1_lsb_hsb(galaxies)
    results['test2'] = test2_rar_geometry(galaxies)
    results['test3'] = test3_mw_thick_thin()
    results['test4'] = test4_barred(galaxies)
    results['test5'] = test5_ellipticals()
    
    # Summary
    print("\n" + "="*70)
    print("  OVERALL SUMMARY")
    print("="*70)
    
    print("""
  Test 1 (LSB vs HSB):
    → Tests if surface brightness Σ matters independently of x
    → Result: Small difference → geometry dominates
    
  Test 2 (RAR vs Geometry):
    → Tests if g_bar or x = R/R_disk is primary variable
    → Result: Partial correlations reveal which matters more
    
  Test 3 (MW thick/thin):
    → Tests if velocity dispersion σ_v matters
    → Result: Compare same R, different σ_v populations
    
  Test 4 (Barred vs Unbarred):
    → Tests if local structure matters
    → Result: Similar K → global geometry matters
    
  Test 5 (Ellipticals):
    → Tests if disk geometry specifically required
    → Result: Different pattern → disk geometry special
""")
    
    print("\n" + "="*70)
    print("  TO RUN ON REAL DATA:")
    print("="*70)
    print("""
  1. Download SPARC from: http://astroweb.cwru.edu/SPARC/
  2. Download Gaia DR3 from: https://gea.esac.esa.int/archive/
  3. Place data in ./data/ directories
  4. Run: python run_all_tests.py
  
  See DATA_ACQUISITION_GUIDE.md for detailed instructions.
""")
    
    # Save results
    Path("./results").mkdir(exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open('./results/demonstration_results.json', 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print("  Results saved to ./results/demonstration_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
