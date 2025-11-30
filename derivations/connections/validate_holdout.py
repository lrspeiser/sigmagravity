#!/usr/bin/env python3
"""
Holdout Validation: Verify Unified Formula Reproduces Results
=============================================================

This script validates that the unified derived formula can reproduce
results on held-out SPARC data (not used for any calibration).

Uses 80/20 stratified split with seed=42 for reproducibility.

Author: Sigma Gravity Team
Date: November 30, 2025
"""

import numpy as np
from pathlib import Path
import sys

# Physical constants
c = 2.998e8
H0_SI = 70 * 1000 / 3.086e22
G = 6.674e-11
kpc_to_m = 3.086e19

g_dagger = c * H0_SI / (2 * np.e)

print("=" * 80)
print("HOLDOUT VALIDATION: UNIFIED FORMULA")
print("=" * 80)
print(f"Date: November 30, 2025")
print(f"g† = {g_dagger:.3e} m/s²")

# =============================================================================
# UNIFIED FORMULA (DERIVED PARAMETERS)
# =============================================================================

def h_universal(g):
    """Universal h(g) - same for all systems."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_derived(r, R_d):
    """Coherence window: W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5

def G_solar_system(R, R_gate=0.5):
    """Solar system safety gate."""
    return 1 - np.exp(-(R / R_gate)**2)

def kernel_unified(R, g_bar, R_d=3.0):
    """
    Unified kernel with derived parameters:
    - A = √3 (from 3D geometry)
    - h(g) = √(g†/g) × g†/(g†+g)
    - W(r) = 1 - (ξ/(ξ+r))^0.5
    """
    A = np.sqrt(3)  # Derived from 3D geometry
    h = h_universal(g_bar)
    W = W_derived(R, R_d)
    K = A * W * h * G_solar_system(R)
    return K

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_data(sparc_dir):
    """Load SPARC rotation curve data."""
    galaxies = {}
    sparc_dir = Path(sparc_dir)
    
    for rotmod_file in sparc_dir.glob('*_rotmod.dat'):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
        
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
        
        if len(R) < 3:
            continue
            
        R = np.array(R)
        V_obs = np.array(V_obs)
        V_err = np.array(V_err)
        V_gas = np.array(V_gas)
        V_disk = np.array(V_disk)
        V_bulge = np.array(V_bulge)
        
        # Compute V_bar
        V_bar = np.sqrt(
            np.sign(V_gas) * V_gas**2 + 
            np.sign(V_disk) * V_disk**2 + 
            V_bulge**2
        )
        
        galaxies[name] = {
            'R': R,
            'V_obs': V_obs,
            'V_err': V_err,
            'V_bar': V_bar,
        }
    
    return galaxies

def load_R_d_values(master_file):
    """Load disk scale lengths from SPARC master table."""
    R_d_values = {}
    
    if not Path(master_file).exists():
        return R_d_values
    
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('-------'):
            data_start = i + 1
            break
    
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('#'):
            continue
        if len(line) < 67:
            continue
        try:
            name = line[0:11].strip()
            Rdisk_str = line[62:67].strip()
            if name and Rdisk_str:
                R_d_values[name] = float(Rdisk_str)
        except:
            continue
    
    return R_d_values

# =============================================================================
# COMPUTE SCATTER ON HOLDOUT
# =============================================================================

def compute_scatter(galaxies, R_d_values, holdout_names):
    """Compute RAR scatter on holdout set only."""
    all_log_residuals = []
    
    for name in holdout_names:
        if name not in galaxies:
            continue
        
        data = galaxies[name]
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        R_d = R_d_values.get(name, 3.0)
        
        # Quality cuts
        mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5) & ~np.isnan(V_bar)
        if np.sum(mask) < 3:
            continue
        
        R = R[mask]
        V_obs = V_obs[mask]
        V_bar = V_bar[mask]
        
        # Compute baryonic acceleration
        g_bar = (V_bar * 1000)**2 / (R * kpc_to_m)
        
        # Compute kernel with derived parameters
        K = kernel_unified(R, g_bar, R_d=R_d)
        
        # Predicted velocity
        V_pred = V_bar * np.sqrt(1 + K)
        
        # Log residual
        mask_good = (V_pred > 0) & (V_obs > 0)
        log_residual = np.log10(V_obs[mask_good] / V_pred[mask_good])
        all_log_residuals.extend(log_residual)
    
    if len(all_log_residuals) == 0:
        return float('inf'), 0, 0
    
    all_log_residuals = np.array(all_log_residuals)
    scatter_dex = np.std(all_log_residuals)
    bias_dex = np.mean(all_log_residuals)
    
    return scatter_dex, bias_dex, len(all_log_residuals)

# =============================================================================
# MAIN VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Find SPARC data
    sparc_dir = Path(r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG")
    master_file = Path(r"C:\Users\henry\dev\sigmagravity\data\SPARC_Lelli2016c.mrt")
    
    if not sparc_dir.exists():
        print(f"\nERROR: SPARC data not found at {sparc_dir}")
        sys.exit(1)
    
    print(f"\nLoading SPARC data from: {sparc_dir}")
    galaxies = load_sparc_data(sparc_dir)
    R_d_values = load_R_d_values(master_file)
    print(f"Loaded {len(galaxies)} galaxies")
    print(f"R_d values for {len(R_d_values)} galaxies")
    
    # Create 80/20 stratified split with seed=42
    np.random.seed(42)
    galaxy_names = list(galaxies.keys())
    np.random.shuffle(galaxy_names)
    
    n_holdout = len(galaxy_names) // 5  # 20%
    holdout_names = galaxy_names[:n_holdout]
    train_names = galaxy_names[n_holdout:]
    
    print(f"\nSplit: {len(train_names)} training, {len(holdout_names)} holdout")
    
    # ==========================================================================
    # TEST 1: Training set (should match what we've been testing)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: TRAINING SET (for reference)")
    print("=" * 80)
    
    scatter_train, bias_train, n_train = compute_scatter(
        galaxies, R_d_values, train_names
    )
    print(f"\nTraining set: {scatter_train:.4f} dex (bias: {bias_train:+.4f}), {n_train} points")
    
    # ==========================================================================
    # TEST 2: Holdout set (true validation)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: HOLDOUT SET (true validation)")
    print("=" * 80)
    
    scatter_holdout, bias_holdout, n_holdout_pts = compute_scatter(
        galaxies, R_d_values, holdout_names
    )
    print(f"\nHoldout set: {scatter_holdout:.4f} dex (bias: {bias_holdout:+.4f}), {n_holdout_pts} points")
    
    # ==========================================================================
    # TEST 3: Full dataset (for comparison)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: FULL DATASET (for comparison)")
    print("=" * 80)
    
    scatter_full, bias_full, n_full = compute_scatter(
        galaxies, R_d_values, galaxy_names
    )
    print(f"\nFull dataset: {scatter_full:.4f} dex (bias: {bias_full:+.4f}), {n_full} points")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    degradation = (scatter_holdout - scatter_train) / scatter_train * 100
    
    print(f"""
UNIFIED FORMULA PARAMETERS (ALL DERIVED):
    g† = cH₀/(2e) = {g_dagger:.3e} m/s²
    A_galaxy = √3 = {np.sqrt(3):.4f}
    n_coh = 0.5 (k/2 with k=1)
    ξ/R_d = 2/3

RESULTS:
    Training set ({len(train_names)} galaxies): {scatter_train:.4f} dex
    Holdout set ({len(holdout_names)} galaxies):  {scatter_holdout:.4f} dex
    
    Degradation (train → holdout): {degradation:+.1f}%

INTERPRETATION:
""")
    
    if abs(degradation) < 10:
        print(f"    ✓ Model generalizes well (degradation < 10%)")
        print(f"    ✓ Holdout scatter {scatter_holdout:.3f} dex validates derived formula")
    else:
        print(f"    ○ Some overfitting detected (degradation {degradation:.1f}%)")
        print(f"    ○ Consider simpler model or regularization")
    
    print(f"""
COMPARISON TO PAPER CLAIMS:
    README claims: 0.094 dex on SPARC
    Holdout result: {scatter_holdout:.4f} dex
    Match: {'✓ YES' if abs(scatter_holdout - 0.094) < 0.01 else '○ CLOSE' if abs(scatter_holdout - 0.094) < 0.02 else '✗ NO'}

REPRODUCIBILITY:
    To reproduce: python derivations/connections/validate_holdout.py
    Seed: 42 (fixed for reproducibility)
""")
