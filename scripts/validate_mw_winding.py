#!/usr/bin/env python3
"""
MW Star-Level Validation: With vs Without Winding
==================================================

Applies the Σ-Gravity kernel with winding gate to Milky Way stars
to confirm the +0.062 dex bias is unchanged with winding.

This uses the same Burr-XII coherence window as SPARC, not the 
saturated-well model used in the original MW pipeline.

Author: Leonard Speiser
Date: 2025-11-25
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "spiral"))

# Constants
G_KPC = 4.302e-6  # (km/s)^2 kpc / Msun
KPC_TO_M = 3.086e19


def C_burr_XII(R, ell_0, p, n_coh):
    """Burr-XII coherence window from paper §2.4."""
    R_safe = np.maximum(R, 1e-10)
    return 1.0 - (1.0 + (R_safe / ell_0)**p)**(-n_coh)


def compute_winding_gate(R, v_c, t_age, N_crit, wind_power):
    """Compute winding gate G_wind from paper §2.9."""
    # N_orbits = t_age * v_c / (2πR * conversion)
    # conversion: kpc * km/s -> Gyr requires factor 0.978
    R_safe = np.maximum(R, 1e-10)
    N_orbits = t_age * v_c / (2.0 * np.pi * R_safe * 0.978)
    G_wind = 1.0 / (1.0 + (N_orbits / N_crit)**wind_power)
    return G_wind, N_orbits


def sigma_gravity_kernel(R, g_bar, params, use_winding=True):
    """
    Compute Σ-Gravity kernel K(R) from paper §2.8.
    
    K(R) = A₀ × (g†/g_bar)^p × C(R) × G_wind
    """
    # Unpack parameters
    A_0 = params.get('A_0', 0.591)
    ell_0 = params.get('ell_0', 4.993)
    p = params.get('p', 0.757)
    n_coh = params.get('n_coh', 0.5)
    g_dagger = params.get('g_dagger', 1.2e-10)  # m/s²
    
    # Winding parameters
    N_crit = params.get('N_crit', 150.0)
    t_age = params.get('t_age', 10.0)
    wind_power = params.get('wind_power', 1.0)
    
    # Convert g_bar from (km/s)²/kpc to m/s²
    g_bar_si = g_bar * 1e6 / KPC_TO_M
    g_bar_si = np.maximum(g_bar_si, 1e-15)
    
    # Coherence window
    C = C_burr_XII(R, ell_0, p, n_coh)
    
    # Acceleration weighting (g†/g_bar)^p
    accel_weight = (g_dagger / g_bar_si)**p
    
    # Estimate v_c from g_bar for winding
    R_safe = np.maximum(R, 1e-10)
    v_c = np.sqrt(np.maximum(g_bar * R_safe, 0))  # km/s
    
    # Winding gate
    if use_winding:
        G_wind, N_orbits = compute_winding_gate(R, v_c, t_age, N_crit, wind_power)
    else:
        G_wind = np.ones_like(R)
        N_orbits = np.zeros_like(R)
    
    # Full kernel
    K = A_0 * accel_weight * C * G_wind
    
    return K, {'C': C, 'G_wind': G_wind, 'N_orbits': N_orbits, 'accel_weight': accel_weight}


def load_mw_stars(npz_path):
    """Load MW star data from NPZ."""
    d = np.load(npz_path)
    return {
        'R': d['R_kpc'],
        'z': d['z_kpc'],
        'v_obs': d['v_obs_kms'],
        'v_err': d['v_err_kms'],
        'gN': d['gN_kms2_per_kpc'],  # Newtonian g in (km/s)²/kpc
    }


def compute_rar_metrics(g_obs, g_pred, label=""):
    """Compute RAR bias and scatter in dex."""
    log_obs = np.log10(np.maximum(g_obs, 1e-15))
    log_pred = np.log10(np.maximum(g_pred, 1e-15))
    delta = log_obs - log_pred
    
    valid = np.isfinite(delta)
    delta = delta[valid]
    
    bias = np.mean(delta)
    scatter = np.std(delta)
    
    return bias, scatter, len(delta)


def main():
    print("=" * 80)
    print("MW STAR-LEVEL RAR VALIDATION: WITH vs WITHOUT WINDING")
    print("=" * 80)
    
    # Paths
    npz_path = REPO_ROOT / "data" / "gaia" / "mw" / "mw_gaia_full_coverage.npz"
    
    if not npz_path.exists():
        print(f"ERROR: Data file not found: {npz_path}")
        print("Run: python scripts/merge_gaia_datasets.py first")
        sys.exit(1)
    
    # Load MW star data
    print(f"\nLoading MW stars from: {npz_path}")
    data = load_mw_stars(npz_path)
    n_stars = len(data['R'])
    print(f"Loaded {n_stars:,} stars, R: {data['R'].min():.2f} - {data['R'].max():.2f} kpc")
    
    # Paper parameters (from hyperparams_track2.json)
    params = {
        'A_0': 0.591,
        'ell_0': 4.993,  # kpc
        'p': 0.757,
        'n_coh': 0.5,
        'g_dagger': 1.2e-10,  # m/s²
        # Winding (effective/tuned for RAR)
        'N_crit': 150.0,
        't_age': 10.0,  # Gyr
        'wind_power': 1.0,
    }
    
    print(f"\nΣ-Gravity parameters (paper §2.8):")
    print(f"  A₀ = {params['A_0']}, ℓ₀ = {params['ell_0']} kpc, p = {params['p']}, n_coh = {params['n_coh']}")
    print(f"  Winding: N_crit = {params['N_crit']}, t_age = {params['t_age']} Gyr, α = {params['wind_power']}")
    
    # Extract data
    R = data['R']
    g_bar = data['gN']  # (km/s)²/kpc
    v_obs = data['v_obs']
    
    # Observed acceleration: g_obs = v²/R
    R_safe = np.maximum(R, 1e-10)
    g_obs = v_obs**2 / R_safe  # (km/s)²/kpc
    
    # ========================================
    # Test 1: WITHOUT winding
    # ========================================
    print("\n" + "-" * 40)
    print("Test 1: Σ-Gravity WITHOUT winding")
    print("-" * 40)
    
    K_no_wind, diag_no = sigma_gravity_kernel(R, g_bar, params, use_winding=False)
    g_pred_no_wind = g_bar * (1 + K_no_wind)
    
    bias_no, scatter_no, n_no = compute_rar_metrics(g_obs, g_pred_no_wind)
    print(f"  Bias:    {bias_no:+.4f} dex")
    print(f"  Scatter: {scatter_no:.4f} dex")
    print(f"  Stars:   {n_no:,}")
    
    # ========================================
    # Test 2: WITH winding (N_crit=150)
    # ========================================
    print("\n" + "-" * 40)
    print("Test 2: Σ-Gravity WITH winding (N_crit=150)")
    print("-" * 40)
    
    K_wind, diag_wind = sigma_gravity_kernel(R, g_bar, params, use_winding=True)
    g_pred_wind = g_bar * (1 + K_wind)
    
    bias_wind, scatter_wind, n_wind = compute_rar_metrics(g_obs, g_pred_wind)
    print(f"  Bias:    {bias_wind:+.4f} dex")
    print(f"  Scatter: {scatter_wind:.4f} dex")
    print(f"  Stars:   {n_wind:,}")
    
    # Winding statistics
    print(f"\n  Winding gate statistics:")
    print(f"    Mean G_wind: {np.mean(diag_wind['G_wind']):.4f}")
    print(f"    Min  G_wind: {np.min(diag_wind['G_wind']):.4f}")
    print(f"    Max  G_wind: {np.max(diag_wind['G_wind']):.4f}")
    print(f"    Mean N_orbits: {np.mean(diag_wind['N_orbits']):.1f}")
    
    # ========================================
    # Test 3: WITH winding (N_crit=10, physical)
    # ========================================
    print("\n" + "-" * 40)
    print("Test 3: Σ-Gravity WITH winding (N_crit=10, physical)")
    print("-" * 40)
    
    params_phys = params.copy()
    params_phys['N_crit'] = 10.0
    params_phys['wind_power'] = 2.0
    
    K_wind10, diag_wind10 = sigma_gravity_kernel(R, g_bar, params_phys, use_winding=True)
    g_pred_wind10 = g_bar * (1 + K_wind10)
    
    bias_wind10, scatter_wind10, n_wind10 = compute_rar_metrics(g_obs, g_pred_wind10)
    print(f"  Bias:    {bias_wind10:+.4f} dex")
    print(f"  Scatter: {scatter_wind10:.4f} dex")
    print(f"  Stars:   {n_wind10:,}")
    print(f"\n  Winding gate statistics:")
    print(f"    Mean G_wind: {np.mean(diag_wind10['G_wind']):.4f}")
    
    # ========================================
    # GR (baryons only) comparison
    # ========================================
    print("\n" + "-" * 40)
    print("Reference: GR (baryons only)")
    print("-" * 40)
    
    bias_gr, scatter_gr, n_gr = compute_rar_metrics(g_obs, g_bar)
    print(f"  Bias:    {bias_gr:+.4f} dex")
    print(f"  Scatter: {scatter_gr:.4f} dex")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"""
┌────────────────────────────────┬────────────┬────────────┐
│ Model                          │ Bias [dex] │ Scatter    │
├────────────────────────────────┼────────────┼────────────┤
│ GR (baryons only)              │ {bias_gr:+.4f}     │ {scatter_gr:.4f}     │
│ Σ-Gravity (no winding)         │ {bias_no:+.4f}     │ {scatter_no:.4f}     │
│ Σ-Gravity + winding (N=150)    │ {bias_wind:+.4f}     │ {scatter_wind:.4f}     │
│ Σ-Gravity + winding (N=10)     │ {bias_wind10:+.4f}     │ {scatter_wind10:.4f}     │
├────────────────────────────────┼────────────┼────────────┤
│ Paper target (MW)              │ +0.062     │ 0.142      │
└────────────────────────────────┴────────────┴────────────┘
""")
    
    # Check if bias is preserved
    delta_bias = abs(bias_wind - bias_no)
    if delta_bias < 0.01:
        print(f"✓ CONFIRMED: Winding has minimal effect on MW bias (Δ = {delta_bias:.4f} dex)")
        print(f"  This validates the paper's claim that MW results are unchanged by winding.")
    else:
        print(f"⚠ Winding changes MW bias by {delta_bias:.4f} dex")
    
    # Save results
    results = {
        'n_stars': int(n_stars),
        'no_winding': {'bias': float(bias_no), 'scatter': float(scatter_no)},
        'winding_N150': {'bias': float(bias_wind), 'scatter': float(scatter_wind)},
        'winding_N10': {'bias': float(bias_wind10), 'scatter': float(scatter_wind10)},
        'gr_baryons': {'bias': float(bias_gr), 'scatter': float(scatter_gr)},
        'delta_bias_winding': float(delta_bias),
    }
    
    out_path = REPO_ROOT / "data" / "gaia" / "outputs" / "mw_winding_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
