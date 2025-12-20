#!/usr/bin/env python3
"""
Test different bulge-specific modifications to Sigma-Gravity.

This script tests various approaches to improve bulge predictions:
1. Direct sigma enhancement (not through V_circ)
2. Different coherence formulas for bulges
3. Bulge-specific parameters
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# Physical constants
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19  # m
H0_SI = 2.27e-18  # 1/s
c = 2.998e8  # m/s

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60×10⁻¹¹ m/s²


def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = sqrt(g_dagger/g) * g_dagger / (g_dagger + g)"""
    g_safe = np.maximum(g, 1e-30)
    return np.sqrt(g_dagger / g_safe) * g_dagger / (g_dagger + g_safe)


def compute_baryonic_acceleration_mw(R_kpc: np.ndarray) -> np.ndarray:
    """Compute baryonic acceleration from simplified MW model."""
    M_bulge = 1.0e10 * M_sun  # kg
    M_disk = 4.6e10 * M_sun  # kg
    R_d = 2.5 * kpc_to_m  # m
    
    R_m = R_kpc * kpc_to_m
    M_enc = M_bulge + M_disk * (1.0 - np.exp(-R_m / R_d))
    g_bar = G * M_enc / np.maximum(R_m**2, 1e-9)
    return g_bar  # m/s²


def C_covariant_coherence(omega2: np.ndarray, rho_kg_m3: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Covariant coherence: C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)"""
    om2 = np.asarray(omega2, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    th2 = np.asarray(theta2, dtype=float)
    
    denom = om2 + 4 * np.pi * G * rho + th2 + H0_SI**2
    C = om2 / np.maximum(denom, 1e-30)
    return np.clip(C, 0.0, 1.0)


def main():
    """Test different bulge modifications."""
    binned_path = Path("data/gaia/6d_brava_galcen.parquet")
    
    if not binned_path.exists():
        print(f"ERROR: Binned data not found: {binned_path}")
        return
    
    print("=" * 80)
    print("TESTING BULGE-SPECIFIC MODIFICATIONS")
    print("=" * 80)
    
    df = pd.read_parquet(binned_path)
    print(f"\nLoaded {len(df)} bins")
    
    # Get data
    R_kpc = df['R_kpc'].values
    omega2 = df['omega2'].values
    theta2 = df['theta2'].values
    rho_kg_m3 = df['rho_kg_m3'].values
    C_cov = df['C_cov'].values
    
    # Observed dispersions
    sigma_R_obs = df['vR_std'].values
    sigma_phi_obs = df['vphi_std'].values
    sigma_z_obs = df['vz_std'].values
    sigma_tot_obs = np.sqrt(sigma_R_obs**2 + sigma_phi_obs**2 + sigma_z_obs**2)
    
    # Compute baseline
    g_bar = compute_baryonic_acceleration_mw(R_kpc)
    V_bar = np.sqrt(g_bar * R_kpc * kpc_to_m) / 1000.0  # km/s
    calibration_factor = 0.51
    sigma_tot_pred_base = V_bar * calibration_factor
    rms_base = np.sqrt(((sigma_tot_obs - sigma_tot_pred_base)**2).mean())
    
    print(f"\nBaseline RMS: {rms_base:.2f} km/s")
    
    # Test 1: Direct sigma enhancement (enhance sigma_base directly)
    # sigma_pred = sigma_base * sqrt(Sigma)
    h = h_function(g_bar)
    Sigma = 1.0 + A_0 * C_cov * h
    sigma_pred_direct = sigma_tot_pred_base * np.sqrt(np.maximum(Sigma, 0.0))
    rms_direct = np.sqrt(((sigma_tot_obs - sigma_pred_direct)**2).mean())
    improvement_direct = rms_base - rms_direct
    
    print(f"\n1. Direct sigma enhancement:")
    print(f"   sigma_pred = sigma_base * sqrt(Sigma)")
    print(f"   RMS: {rms_direct:.2f} km/s, Improvement: {improvement_direct:+.2f} km/s")
    
    # Test 2: Enhanced sigma with different power
    # sigma_pred = sigma_base * Sigma^gamma, fit gamma
    gamma_candidates = np.linspace(0.0, 1.5, 30)
    best_rms_gamma = np.inf
    best_gamma = 0.0
    for gamma_test in gamma_candidates:
        sigma_pred_test = sigma_tot_pred_base * (Sigma ** gamma_test)
        rms_test = np.sqrt(((sigma_tot_obs - sigma_pred_test)**2).mean())
        if rms_test < best_rms_gamma:
            best_rms_gamma = rms_test
            best_gamma = gamma_test
    
    print(f"\n2. Sigma power law (fit gamma):")
    print(f"   sigma_pred = sigma_base * Sigma^gamma")
    print(f"   Best gamma: {best_gamma:.3f}")
    print(f"   RMS: {best_rms_gamma:.2f} km/s, Improvement: {rms_base - best_rms_gamma:+.2f} km/s")
    
    # Test 3: Bulge-specific A_0 with direct sigma
    # Fit A_0_bulge for direct sigma enhancement
    A_0_candidates = np.logspace(np.log10(0.1 * A_0), np.log10(10 * A_0), 30)
    best_rms_A0 = np.inf
    best_A_0_bulge = A_0
    for A_test in A_0_candidates:
        Sigma_test = 1.0 + A_test * C_cov * h
        sigma_pred_test = sigma_tot_pred_base * np.sqrt(np.maximum(Sigma_test, 0.0))
        rms_test = np.sqrt(((sigma_tot_obs - sigma_pred_test)**2).mean())
        if rms_test < best_rms_A0:
            best_rms_A0 = rms_test
            best_A_0_bulge = A_test
    
    print(f"\n3. Bulge-specific A_0 (with direct sigma):")
    print(f"   A_0_bulge = {best_A_0_bulge:.4f} (vs standard {A_0:.4f})")
    print(f"   RMS: {best_rms_A0:.2f} km/s, Improvement: {rms_base - best_rms_A0:+.2f} km/s")
    
    # Test 4: Modified C_cov for bulge (reduce density term)
    # C_cov_bulge = omega^2 / (omega^2 + alpha * 4*pi*G*rho + theta^2 + beta * H0^2)
    alpha_candidates = np.logspace(-2, 0, 20)  # 0.01 to 1.0
    best_rms_alpha = np.inf
    best_alpha = 1.0
    for alpha_test in alpha_candidates:
        denom_mod = omega2 + alpha_test * 4 * np.pi * G * rho_kg_m3 + theta2 + H0_SI**2
        C_cov_mod = omega2 / np.maximum(denom_mod, 1e-30)
        C_cov_mod = np.clip(C_cov_mod, 0.0, 1.0)
        Sigma_mod = 1.0 + A_0 * C_cov_mod * h
        sigma_pred_mod = sigma_tot_pred_base * np.sqrt(np.maximum(Sigma_mod, 0.0))
        rms_test = np.sqrt(((sigma_tot_obs - sigma_pred_mod)**2).mean())
        if rms_test < best_rms_alpha:
            best_rms_alpha = rms_test
            best_alpha = alpha_test
    
    print(f"\n4. Modified C_cov (reduce density term):")
    print(f"   alpha = {best_alpha:.4f} (density term weight)")
    print(f"   RMS: {best_rms_alpha:.2f} km/s, Improvement: {rms_base - best_rms_alpha:+.2f} km/s")
    
    # Test 5: Combined approach (best A_0 + best alpha + best gamma)
    denom_best = omega2 + best_alpha * 4 * np.pi * G * rho_kg_m3 + theta2 + H0_SI**2
    C_cov_best = omega2 / np.maximum(denom_best, 1e-30)
    C_cov_best = np.clip(C_cov_best, 0.0, 1.0)
    Sigma_best = 1.0 + best_A_0_bulge * C_cov_best * h
    sigma_pred_best = sigma_tot_pred_base * (Sigma_best ** best_gamma)
    rms_best = np.sqrt(((sigma_tot_obs - sigma_pred_best)**2).mean())
    
    print(f"\n5. Combined (best A_0 + best alpha + best gamma):")
    print(f"   A_0_bulge = {best_A_0_bulge:.4f}, alpha = {best_alpha:.4f}, gamma = {best_gamma:.3f}")
    print(f"   RMS: {rms_best:.2f} km/s, Improvement: {rms_base - rms_best:+.2f} km/s")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_tests = {
        'Baseline': rms_base,
        'Direct sigma (sqrt)': rms_direct,
        'Sigma power law': best_rms_gamma,
        'Bulge A_0': best_rms_A0,
        'Modified C_cov': best_rms_alpha,
        'Combined': rms_best,
    }
    
    best_test = min(all_tests.items(), key=lambda x: x[1])
    print(f"\nBest approach: {best_test[0]} (RMS = {best_test[1]:.2f} km/s)")
    print(f"Improvement over baseline: {rms_base - best_test[1]:+.2f} km/s")
    
    if best_test[1] < rms_base - 0.5:
        print(f"\nRECOMMENDATION: Use {best_test[0]} approach for bulge predictions")
        if best_test[0] == 'Combined':
            print(f"  - A_0_bulge = {best_A_0_bulge:.4f}")
            print(f"  - C_cov alpha (density weight) = {best_alpha:.4f}")
            print(f"  - Sigma power (gamma) = {best_gamma:.3f}")
            print(f"  - Formula: sigma_pred = sigma_base * Sigma^gamma")
            print(f"  - Where: Sigma = 1 + A_0_bulge * C_cov_bulge * h")
            print(f"  - And: C_cov_bulge = omega^2 / (omega^2 + alpha*4*pi*G*rho + theta^2 + H0^2)")
    else:
        print(f"\nWARNING: No significant improvement found.")
        print(f"  All modifications perform similarly to baseline.")
        print(f"  This suggests the problem may require a different approach entirely.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

