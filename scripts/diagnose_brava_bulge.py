#!/usr/bin/env python3
"""
Diagnose BRAVA Bulge Predictions to Find Sigma-Gravity Modifications

This script analyzes why C_cov isn't improving bulge predictions and
identifies what modifications to Sigma-Gravity might help.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical constants
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19  # m
H0_SI = 2.27e-18  # 1/s
c = 2.998e8  # m/s

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
L_0 = 0.40  # kpc
N_EXP = 0.27

# g_dagger for h_function
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


def main():
    """Diagnose bulge predictions and identify needed modifications."""
    binned_path = Path("data/gaia/6d_brava_galcen.parquet")
    
    if not binned_path.exists():
        print(f"ERROR: Binned data not found: {binned_path}")
        return
    
    print("=" * 80)
    print("BRAVA BULGE DIAGNOSTIC: Finding Sigma-Gravity Modifications")
    print("=" * 80)
    
    df = pd.read_parquet(binned_path)
    print(f"\nLoaded {len(df)} bins")
    
    # Get observables
    R_kpc = df['R_kpc'].values
    omega2 = df['omega2'].values
    theta2 = df['theta2'].values
    rho_kg_m3 = df['rho_kg_m3'].values
    C_cov = df['C_cov'].values
    Sigma = df['Sigma'].values
    
    # Observed dispersions
    sigma_R_obs = df['vR_std'].values
    sigma_phi_obs = df['vphi_std'].values
    sigma_z_obs = df['vz_std'].values
    sigma_tot_obs = np.sqrt(sigma_R_obs**2 + sigma_phi_obs**2 + sigma_z_obs**2)
    
    # Compute predictions
    g_bar = compute_baryonic_acceleration_mw(R_kpc)
    h = h_function(g_bar)
    V_bar = np.sqrt(g_bar * R_kpc * kpc_to_m) / 1000.0  # km/s
    V_circ = V_bar * np.sqrt(np.maximum(Sigma, 0.0))
    calibration_factor = 0.51
    sigma_tot_pred = V_circ * calibration_factor
    
    # Baseline (no enhancement)
    sigma_tot_pred_base = V_bar * calibration_factor
    
    # Residuals
    resid = sigma_tot_obs - sigma_tot_pred
    resid_base = sigma_tot_obs - sigma_tot_pred_base
    
    rms = np.sqrt((resid**2).mean())
    rms_base = np.sqrt((resid_base**2).mean())
    improvement = rms_base - rms
    
    print("\n" + "=" * 80)
    print("CURRENT PERFORMANCE")
    print("=" * 80)
    print(f"RMS (with C_cov):     {rms:.2f} km/s")
    print(f"RMS (baseline):       {rms_base:.2f} km/s")
    print(f"Improvement:          {improvement:.2f} km/s")
    print(f"  {'IMPROVEMENT' if improvement > 0 else 'NO IMPROVEMENT'}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Why isn't C_cov helping?")
    print("=" * 80)
    
    print(f"\n1. Enhancement factors:")
    print(f"   mean(C_cov):       {C_cov.mean():.4f}  (range: {C_cov.min():.4f} - {C_cov.max():.4f})")
    print(f"   mean(h):           {h.mean():.6e}  (range: {h.min():.6e} - {h.max():.6e})")
    print(f"   mean(Sigma):       {Sigma.mean():.4f}  (range: {Sigma.min():.4f} - {Sigma.max():.4f})")
    print(f"   mean(A_0 * C_cov * h): {A_0 * C_cov.mean() * h.mean():.6e}")
    print(f"   Expected Sigma:    1.0 + {A_0 * C_cov.mean() * h.mean():.6e} = {1.0 + A_0 * C_cov.mean() * h.mean():.4f}")
    
    print(f"\n2. Physical quantities:")
    print(f"   mean(omega^2):     {omega2.mean():.2f}  (1/s^2)")
    print(f"   mean(theta^2):     {theta2.mean():.6e}  (1/s^2)")
    print(f"   mean(rho):         {rho_kg_m3.mean():.3e}  (kg/m^3)")
    print(f"   mean(g_bar):       {g_bar.mean():.6e}  (m/s^2)")
    print(f"   mean(R):           {R_kpc.mean():.2f}  (kpc)")
    
    print(f"\n3. Predictions vs Observations:")
    print(f"   mean(sigma_obs):   {sigma_tot_obs.mean():.2f} km/s")
    print(f"   mean(sigma_pred):  {sigma_tot_pred.mean():.2f} km/s")
    print(f"   mean(sigma_base):  {sigma_tot_pred_base.mean():.2f} km/s")
    print(f"   Bias (pred):       {resid.mean():.2f} km/s")
    print(f"   Bias (base):       {resid_base.mean():.2f} km/s")
    
    # Check if Sigma is too small
    sigma_enhancement = Sigma - 1.0
    print(f"\n4. Enhancement analysis:")
    print(f"   mean(Sigma - 1):    {sigma_enhancement.mean():.6f}")
    print(f"   max(Sigma - 1):    {sigma_enhancement.max():.6f}")
    print(f"   Enhancement %:     {100 * sigma_enhancement.mean():.2f}%")
    
    if sigma_enhancement.mean() < 0.01:
        print(f"\n   WARNING: Sigma enhancement is too small (< 1%)!")
        print(f"      This means C_cov * h is too small to matter.")
        print(f"      Possible fixes:")
        print(f"        a) Increase A_0 (currently {A_0:.4f})")
        print(f"        b) Use bulge-specific A_0_bulge > A_0")
        print(f"        c) Modify h_function for bulge (different g_dagger?)")
        print(f"        d) Use different coherence formula for bulge")
    
    # Check correlation between residuals and physical quantities
    print(f"\n5. Residual correlations (what predicts errors?):")
    corr_R = np.corrcoef(resid, R_kpc)[0, 1]
    corr_C_cov = np.corrcoef(resid, C_cov)[0, 1]
    corr_Sigma = np.corrcoef(resid, Sigma)[0, 1]
    corr_omega2 = np.corrcoef(resid, omega2)[0, 1]
    corr_rho = np.corrcoef(resid, rho_kg_m3)[0, 1]
    
    print(f"   resid vs R:        {corr_R:+.3f}")
    print(f"   resid vs C_cov:    {corr_C_cov:+.3f}")
    print(f"   resid vs Sigma:    {corr_Sigma:+.3f}")
    print(f"   resid vs omega^2:  {corr_omega2:+.3f}")
    print(f"   resid vs rho:      {corr_rho:+.3f}")
    
    # Test different modifications
    print("\n" + "=" * 80)
    print("TESTING MODIFICATIONS")
    print("=" * 80)
    
    modifications = {}
    
    # Modification 1: Increase A_0 for bulge
    A_0_bulge_2x = 2.0 * A_0
    Sigma_2x = 1.0 + A_0_bulge_2x * C_cov * h
    V_circ_2x = V_bar * np.sqrt(np.maximum(Sigma_2x, 0.0))
    sigma_pred_2x = V_circ_2x * calibration_factor
    rms_2x = np.sqrt(((sigma_tot_obs - sigma_pred_2x)**2).mean())
    modifications['2x A_0'] = {
        'rms': rms_2x,
        'improvement': rms_base - rms_2x,
        'mean_sigma': Sigma_2x.mean()
    }
    
    # Modification 2: Bulge-specific A_0 (fit to data)
    # Try A_0_bulge = 5.0 (much larger)
    A_0_bulge_5x = 5.0 * A_0
    Sigma_5x = 1.0 + A_0_bulge_5x * C_cov * h
    V_circ_5x = V_bar * np.sqrt(np.maximum(Sigma_5x, 0.0))
    sigma_pred_5x = V_circ_5x * calibration_factor
    rms_5x = np.sqrt(((sigma_tot_obs - sigma_pred_5x)**2).mean())
    modifications['5x A_0'] = {
        'rms': rms_5x,
        'improvement': rms_base - rms_5x,
        'mean_sigma': Sigma_5x.mean()
    }
    
    # Modification 3: Different h_function for bulge (larger g_dagger)
    g_dagger_bulge = 10.0 * g_dagger
    def h_bulge(g):
        g_safe = np.maximum(g, 1e-30)
        return np.sqrt(g_dagger_bulge / g_safe) * g_dagger_bulge / (g_dagger_bulge + g_safe)
    h_bulge_vals = h_bulge(g_bar)
    Sigma_h_bulge = 1.0 + A_0 * C_cov * h_bulge_vals
    V_circ_h_bulge = V_bar * np.sqrt(np.maximum(Sigma_h_bulge, 0.0))
    sigma_pred_h_bulge = V_circ_h_bulge * calibration_factor
    rms_h_bulge = np.sqrt(((sigma_tot_obs - sigma_pred_h_bulge)**2).mean())
    modifications['10x g_dagger'] = {
        'rms': rms_h_bulge,
        'improvement': rms_base - rms_h_bulge,
        'mean_sigma': Sigma_h_bulge.mean()
    }
    
    # Modification 4: Direct sigma prediction (skip V_circ)
    # Try: sigma_pred = sigma_base * sqrt(Sigma)
    sigma_pred_direct = sigma_tot_pred_base * np.sqrt(np.maximum(Sigma, 0.0))
    rms_direct = np.sqrt(((sigma_tot_obs - sigma_pred_direct)**2).mean())
    modifications['Direct sigma'] = {
        'rms': rms_direct,
        'improvement': rms_base - rms_direct,
        'mean_sigma': Sigma.mean()
    }
    
    # Modification 5: Fit optimal A_0_bulge
    # Grid search for best A_0_bulge
    A_0_candidates = np.logspace(np.log10(A_0), np.log10(10 * A_0), 20)
    best_rms = np.inf
    best_A_0_bulge = A_0
    for A_test in A_0_candidates:
        Sigma_test = 1.0 + A_test * C_cov * h
        V_circ_test = V_bar * np.sqrt(np.maximum(Sigma_test, 0.0))
        sigma_pred_test = V_circ_test * calibration_factor
        rms_test = np.sqrt(((sigma_tot_obs - sigma_pred_test)**2).mean())
        if rms_test < best_rms:
            best_rms = rms_test
            best_A_0_bulge = A_test
    
    Sigma_opt = 1.0 + best_A_0_bulge * C_cov * h
    V_circ_opt = V_bar * np.sqrt(np.maximum(Sigma_opt, 0.0))
    sigma_pred_opt = V_circ_opt * calibration_factor
    modifications['Optimal A_0'] = {
        'rms': best_rms,
        'improvement': rms_base - best_rms,
        'mean_sigma': Sigma_opt.mean(),
        'A_0_bulge': best_A_0_bulge
    }
    
    # Modification 6: Modified C_cov formula (weight omega^2 more)
    # C_cov_mod = omega^2 / (omega^2 + alpha * 4*pi*G*rho + theta^2 + beta * H0^2)
    # Try reducing density term weight
    alpha = 0.1  # Reduce density term
    beta = 0.1   # Reduce H0 term
    denom_mod = omega2 + alpha * 4 * np.pi * G * rho_kg_m3 + theta2 + beta * H0_SI**2
    C_cov_mod = omega2 / np.maximum(denom_mod, 1e-30)
    C_cov_mod = np.clip(C_cov_mod, 0.0, 1.0)
    Sigma_mod = 1.0 + A_0 * C_cov_mod * h
    V_circ_mod = V_bar * np.sqrt(np.maximum(Sigma_mod, 0.0))
    sigma_pred_mod = V_circ_mod * calibration_factor
    rms_mod = np.sqrt(((sigma_tot_obs - sigma_pred_mod)**2).mean())
    modifications['Modified C_cov (alpha=0.1)'] = {
        'rms': rms_mod,
        'improvement': rms_base - rms_mod,
        'mean_sigma': Sigma_mod.mean()
    }
    
    # Modification 7: Direct sigma from flow invariants (Jeans-like)
    # sigma^2 = alpha * (omega^2 * R^2) / (1 + C_cov)
    # This directly relates dispersion to vorticity
    R_m = R_kpc * kpc_to_m
    omega = np.sqrt(np.maximum(omega2, 0.0))  # 1/s
    # Try: sigma^2 = k * omega^2 * R^2, fit k
    k_candidates = np.logspace(-2, 2, 50)
    best_rms_direct = np.inf
    best_k = 1.0
    for k_test in k_candidates:
        sigma_sq_pred = k_test * omega2 * R_m**2  # m^2/s^2
        sigma_pred_direct2 = np.sqrt(np.maximum(sigma_sq_pred, 0.0)) / 1000.0  # km/s
        rms_test = np.sqrt(((sigma_tot_obs - sigma_pred_direct2)**2).mean())
        if rms_test < best_rms_direct:
            best_rms_direct = rms_test
            best_k = k_test
    
    sigma_sq_best = best_k * omega2 * R_m**2
    sigma_pred_direct2 = np.sqrt(np.maximum(sigma_sq_best, 0.0)) / 1000.0
    modifications['Direct sigma from omega'] = {
        'rms': best_rms_direct,
        'improvement': rms_base - best_rms_direct,
        'mean_sigma': 1.0,  # Not applicable
        'k': best_k
    }
    
    # Modification 8: Sigma-dependent calibration factor
    # Instead of fixed calibration, use: sigma = V_circ * f(Sigma)
    # Try: f(Sigma) = calibration_factor * Sigma^gamma
    gamma_candidates = np.linspace(0.0, 1.0, 20)
    best_rms_cal = np.inf
    best_gamma = 0.0
    for gamma_test in gamma_candidates:
        cal_factor_mod = calibration_factor * (Sigma ** gamma_test)
        sigma_pred_cal = V_bar * np.sqrt(np.maximum(Sigma, 0.0)) * cal_factor_mod
        rms_test = np.sqrt(((sigma_tot_obs - sigma_pred_cal)**2).mean())
        if rms_test < best_rms_cal:
            best_rms_cal = rms_test
            best_gamma = gamma_test
    
    cal_factor_best = calibration_factor * (Sigma ** best_gamma)
    sigma_pred_cal = V_bar * np.sqrt(np.maximum(Sigma, 0.0)) * cal_factor_best
    modifications['Sigma-dependent cal'] = {
        'rms': best_rms_cal,
        'improvement': rms_base - best_rms_cal,
        'mean_sigma': Sigma.mean(),
        'gamma': best_gamma
    }
    
    print(f"\nModification          RMS (km/s)    Improvement    Mean Sigma")
    print("-" * 80)
    print(f"Current (C_cov)      {rms:8.2f}    {improvement:8.2f}    {Sigma.mean():.4f}")
    print(f"Baseline (no enh)    {rms_base:8.2f}    {'0.00':>8}    {1.0:.4f}")
    for name, mod in modifications.items():
        extra = ""
        if 'A_0_bulge' in mod:
            extra = f", A_0={mod['A_0_bulge']:.3f}"
        elif 'k' in mod:
            extra = f", k={mod['k']:.4e}"
        elif 'gamma' in mod:
            extra = f", gamma={mod['gamma']:.3f}"
        print(f"{name:30s} {mod['rms']:8.2f}    {mod['improvement']:8.2f}    {mod['mean_sigma']:.4f}{extra}")
    
    # Find best modification
    all_rms = {'Current': rms, 'Baseline': rms_base}
    all_rms.update({k: v['rms'] for k, v in modifications.items()})
    best_mod = min(all_rms.items(), key=lambda x: x[1])
    
    print(f"\n{'='*80}")
    print(f"BEST MODIFICATION: {best_mod[0]} (RMS = {best_mod[1]:.2f} km/s)")
    print(f"{'='*80}")
    
    if 'A_0_bulge' in modifications.get('Optimal A_0', {}):
        print(f"\nRecommended: Use A_0_bulge = {modifications['Optimal A_0']['A_0_bulge']:.4f}")
        print(f"  (vs standard A_0 = {A_0:.4f})")
        print(f"  This gives {modifications['Optimal A_0']['improvement']:.2f} km/s improvement")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best modification (excluding baseline)
    mod_scores = {k: v for k, v in modifications.items() if k != 'Baseline'}
    best_mod_name = max(mod_scores.items(), key=lambda x: x[1]['improvement'])[0]
    best_mod = modifications[best_mod_name]
    
    if best_mod['improvement'] > 0.5:
        print(f"\n1. BEST MODIFICATION: {best_mod_name}")
        print(f"   RMS improvement: {best_mod['improvement']:.2f} km/s")
        if 'A_0_bulge' in best_mod:
            print(f"   Use A_0_bulge = {best_mod['A_0_bulge']:.4f} (vs standard {A_0:.4f})")
        elif 'k' in best_mod:
            print(f"   Use direct sigma: sigma^2 = {best_mod['k']:.4e} * omega^2 * R^2")
        elif 'gamma' in best_mod:
            print(f"   Use calibration: f(Sigma) = {calibration_factor} * Sigma^{best_mod['gamma']:.3f}")
        elif 'alpha' in best_mod_name.lower():
            print(f"   Use modified C_cov with reduced density term weight")
    else:
        print(f"\n1. NO CLEAR WINNER:")
        print(f"   Best modification ({best_mod_name}) only improves by {best_mod['improvement']:.2f} km/s")
        print(f"   This suggests the problem may be more fundamental.")
    
    if modifications.get('Direct sigma from omega', {}).get('improvement', -999) > 1.0:
        print(f"\n2. CONSIDER DIRECT SIGMA FROM FLOW INVARIANTS:")
        print(f"   sigma^2 = k * omega^2 * R^2")
        print(f"   k = {modifications['Direct sigma from omega']['k']:.4e}")
        print(f"   This bypasses V_circ entirely and directly uses vorticity.")
    
    if modifications.get('Sigma-dependent cal', {}).get('improvement', -999) > 1.0:
        print(f"\n3. CONSIDER SIGMA-DEPENDENT CALIBRATION:")
        print(f"   calibration_factor(Sigma) = {calibration_factor} * Sigma^{modifications['Sigma-dependent cal']['gamma']:.3f}")
        print(f"   This accounts for how enhancement affects dispersion differently than rotation.")
    
    print(f"\n4. CHECK IF C_cov FORMULA NEEDS ADJUSTMENT:")
    print(f"   Current: C_cov = omega^2 / (omega^2 + 4*pi*G*rho + theta^2 + H0^2)")
    print(f"   For bulge, maybe need different weighting of terms?")
    if modifications.get('Modified C_cov (alpha=0.1)', {}).get('improvement', -999) > 0:
        print(f"   Tested alpha=0.1 (reduce density term): improvement = {modifications['Modified C_cov (alpha=0.1)']['improvement']:.2f} km/s")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

