"""
Dedicated test for Tully-Fisher scaling prediction.
If ℓ ∝ √M_b, then v⁴ ∝ M_b (baryonic Tully-Fisher).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import json
import os

def test_tully_fisher_scaling(sparc_csv_path: str, 
                              fitted_ell0: float = 4.993,
                              fitted_A: float = 0.591):
    """
    Test the prediction from document 1:
    
    v_∞² = α(GM_b/λ_g)
    
    For flat rotation curves, if λ_g is universal:
        v⁴ ∝ M_b²  (too steep)
    
    But if λ_g ∝ √M_b:
        v⁴ ∝ M_b  (observed)
    
    We test this by computing implied λ_g for each galaxy.
    """
    
    print("="*80)
    print("TULLY-FISHER SCALING TEST")
    print("="*80)
    
    # Load data
    print("\nLoading SPARC data...")
    df = pd.read_csv(sparc_csv_path)
    print(f"Loaded {len(df)} galaxies")
    
    # Extract relevant quantities
    M_b = df['M_baryon'].values  # M_sun
    v_flat = df['v_flat'].values  # km/s
    
    # Constants
    G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
    Msun_kg = 1.989e30
    kpc_m = 3.086e19
    
    # Compute implied λ_g for each galaxy
    # From v² = αGM/λ_g with α ≈ fitted_A = 0.591
    alpha = fitted_A
    
    M_b_SI = M_b * Msun_kg
    v_SI = v_flat * 1e3
    
    lambda_g = alpha * G_SI * M_b_SI / (v_SI**2)
    lambda_g_kpc = lambda_g / kpc_m
    
    # Remove invalid values
    mask = np.isfinite(lambda_g_kpc) & (lambda_g_kpc > 0)
    M_b = M_b[mask]
    v_flat = v_flat[mask]
    lambda_g_kpc = lambda_g_kpc[mask]
    
    print(f"\nValid galaxies: {len(M_b)}/{len(df)}")
    
    # Test 1: Does λ_g scale with M_b?
    print("\n" + "="*80)
    print("TEST 1: Power-law fit λ_g = k × M_b^γ")
    print("="*80)
    
    def power_law(M, k, gamma):
        return k * M**gamma
    
    # Fit in log-space for stability
    log_M = np.log10(M_b)
    log_lambda = np.log10(lambda_g_kpc)
    
    coeffs = np.polyfit(log_M, log_lambda, 1)
    gamma_fit = coeffs[0]
    log_k_fit = coeffs[1]
    k_fit = 10**log_k_fit
    
    # Compute R² and residuals
    lambda_pred = k_fit * M_b**gamma_fit
    residuals = np.log10(lambda_g_kpc) - np.log10(lambda_pred)
    r_squared = 1 - (np.var(residuals) / np.var(log_lambda))
    
    print(f"\nFit: λ_g = {k_fit:.6f} × M_b^{gamma_fit:.4f} kpc")
    print(f"R²: {r_squared:.4f}")
    print(f"Scatter: {np.std(residuals):.4f} dex")
    print(f"\nExpectation for Tully-Fisher: γ = 0.5")
    print(f"Fitted: γ = {gamma_fit:.4f}")
    print(f"Deviation: {abs(gamma_fit - 0.5):.4f}")
    
    if abs(gamma_fit - 0.5) < 0.1:
        print("✓ Consistent with Tully-Fisher prediction!")
    else:
        print("✗ NOT consistent with Tully-Fisher (γ ≠ 0.5)")
    
    # Test 2: Does v⁴ ∝ M_b?
    print("\n" + "="*80)
    print("TEST 2: Baryonic Tully-Fisher relation")
    print("="*80)
    
    v4 = v_flat**4
    
    coeffs_TF = np.polyfit(np.log10(M_b), np.log10(v4), 1)
    slope_TF = coeffs_TF[0]
    
    print(f"\nFit: v⁴ ∝ M_b^{slope_TF:.4f}")
    print(f"Expectation: slope = 1.0 (if λ_g ∝ √M_b)")
    print(f"Deviation: {abs(slope_TF - 1.0):.4f}")
    
    if abs(slope_TF - 1.0) < 0.2:
        print("✓ Consistent with BTFR!")
    else:
        print("✗ NOT consistent with BTFR")
    
    # Test 3: Is λ_g approximately universal?
    print("\n" + "="*80)
    print("TEST 3: Universality check")
    print("="*80)
    
    median_lambda = np.median(lambda_g_kpc)
    mad_lambda = np.median(np.abs(lambda_g_kpc - median_lambda))
    std_lambda = np.std(lambda_g_kpc)
    
    print(f"\nMedian λ_g: {median_lambda:.3f} kpc")
    print(f"MAD: {mad_lambda:.3f} kpc ({100*mad_lambda/median_lambda:.1f}%)")
    print(f"Std: {std_lambda:.3f} kpc ({100*std_lambda/median_lambda:.1f}%)")
    print(f"Fitted ℓ₀: {fitted_ell0:.3f} kpc")
    print(f"Ratio median/fitted: {median_lambda/fitted_ell0:.3f}")
    
    if std_lambda / median_lambda < 0.5:
        print("✓ Relatively small scatter → weakly universal")
    else:
        print("✗ Large scatter → strongly mass-dependent")
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tully-Fisher Scaling Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: λ_g vs M_b with power-law fit
    ax = axes[0, 0]
    ax.scatter(M_b, lambda_g_kpc, alpha=0.5, s=20, label='Data')
    
    M_theory = np.logspace(np.log10(M_b.min()), np.log10(M_b.max()), 100)
    lambda_theory = k_fit * M_theory**gamma_fit
    ax.plot(M_theory, lambda_theory, 'r-', linewidth=2, 
            label=f'Fit: λ ∝ M^{gamma_fit:.3f}')
    
    # Overplot expectation
    lambda_expected = (fitted_ell0 / M_b.mean()**0.5) * M_theory**0.5
    ax.plot(M_theory, lambda_expected, 'g--', linewidth=2, alpha=0.7,
            label=f'Expected: λ ∝ M^0.5')
    
    ax.axhline(fitted_ell0, color='orange', linestyle=':', linewidth=2,
               label=f'Fitted ℓ₀ = {fitted_ell0:.2f} kpc')
    
    ax.set_xlabel('M_baryon [M_sun]', fontsize=12)
    ax.set_ylabel('Implied λ_g [kpc]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'γ = {gamma_fit:.3f} (expect 0.5)', fontsize=12)
    
    # Plot 2: Residuals vs M_b
    ax = axes[0, 1]
    ax.scatter(M_b, residuals, alpha=0.5, s=20)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('M_baryon [M_sun]', fontsize=12)
    ax.set_ylabel('log₁₀(λ_obs / λ_fit)', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Scatter = {np.std(residuals):.4f} dex', fontsize=12)
    
    # Plot 3: Histogram of λ_g
    ax = axes[0, 2]
    ax.hist(lambda_g_kpc, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(fitted_ell0, color='r', linestyle='--', linewidth=2,
               label=f'Fitted: {fitted_ell0:.2f} kpc')
    ax.axvline(median_lambda, color='b', linestyle='--', linewidth=2,
               label=f'Median: {median_lambda:.2f} kpc')
    ax.set_xlabel('Implied λ_g [kpc]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Dispersion: {std_lambda/median_lambda:.2%}', fontsize=12)
    
    # Plot 4: Baryonic Tully-Fisher
    ax = axes[1, 0]
    ax.scatter(M_b, v4, alpha=0.5, s=20, label='Data')
    
    # Fit line
    v4_fit = 10**(coeffs_TF[1]) * M_b**slope_TF
    ax.plot(M_b, v4_fit, 'r-', linewidth=2, label=f'Fit: v⁴ ∝ M^{slope_TF:.3f}')
    
    # Expected if λ ∝ M^0.5
    ax.plot(M_theory, M_theory * (v4.mean() / M_b.mean()), 'g--', linewidth=2,
            alpha=0.7, label='Expected: v⁴ ∝ M')
    
    ax.set_xlabel('M_baryon [M_sun]', fontsize=12)
    ax.set_ylabel('v_flat⁴ [(km/s)⁴]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Baryonic Tully-Fisher', fontsize=12)
    
    # Plot 5: λ_g vs v_flat
    ax = axes[1, 1]
    ax.scatter(v_flat, lambda_g_kpc, alpha=0.5, s=20)
    ax.axhline(fitted_ell0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('v_flat [km/s]', fontsize=12)
    ax.set_ylabel('Implied λ_g [kpc]', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('λ_g vs Velocity', fontsize=12)
    
    # Plot 6: Consistency check (predicted vs actual)
    ax = axes[1, 2]
    
    # If λ_g = k × M_b^0.5, what v do we predict?
    lambda_g_assumed = k_fit * M_b**0.5
    v_predicted = np.sqrt(alpha * G_SI * M_b_SI / (lambda_g_assumed * kpc_m)) / 1e3
    
    ax.scatter(v_flat, v_predicted, alpha=0.5, s=20)
    ax.plot([v_flat.min(), v_flat.max()], [v_flat.min(), v_flat.max()], 
            'r--', linewidth=2, label='1:1')
    ax.set_xlabel('v_flat observed [km/s]', fontsize=12)
    ax.set_ylabel('v_flat predicted [km/s]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Consistency: λ ∝ M^0.5', fontsize=12)
    
    plt.tight_layout()
    
    output_dir = "GravityWaveTest"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/tully_fisher_scaling_test.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/tully_fisher_scaling_test.png")
    plt.close()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n1. Power-law index: γ = {gamma_fit:.4f} (expect 0.5 for BTFR)")
    print(f"2. Tully-Fisher slope: {slope_TF:.4f} (expect 1.0)")
    print(f"3. Median λ_g: {median_lambda:.3f} kpc vs fitted {fitted_ell0:.3f} kpc")
    print(f"4. Scatter in λ_g: {std_lambda/median_lambda:.1%}")
    
    if abs(gamma_fit - 0.5) < 0.1 and abs(slope_TF - 1.0) < 0.2:
        print("\n✓✓ STRONG EVIDENCE: λ_g ∝ √M_b as predicted by Tully-Fisher!")
    elif abs(gamma_fit - 0.5) < 0.2:
        print("\n✓ MODERATE EVIDENCE: λ_g weakly scales with √M_b")
    else:
        print("\n✗ NO EVIDENCE: λ_g does not scale as √M_b")
    
    # Save results
    results = {
        'gamma_fit': float(gamma_fit),
        'gamma_expected': 0.5,
        'k_fit': float(k_fit),
        'TF_slope': float(slope_TF),
        'TF_expected': 1.0,
        'median_lambda_kpc': float(median_lambda),
        'fitted_ell0_kpc': fitted_ell0,
        'scatter_dex': float(np.std(residuals)),
        'r_squared': float(r_squared),
        'n_galaxies': len(M_b)
    }
    
    with open(f'{output_dir}/tully_fisher_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/tully_fisher_results.json")
    
    return results

if __name__ == "__main__":
    test_tully_fisher_scaling(
        sparc_csv_path="data/sparc/sparc_combined.csv",
        fitted_ell0=4.993,
        fitted_A=0.591
    )

