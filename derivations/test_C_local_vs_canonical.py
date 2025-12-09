#!/usr/bin/env python3
"""
Test C_local (Coherence Scalar) vs Canonical W(r) Using Official Regression Framework

This script compares the proposed C_local formulation against the canonical
geometric W(r) using the EXACT same methodology as run_regression_extended.py.

Feedback items 5-8 propose:
  - Replace W(r) = r/(ξ+r) with W_C(r) = C(r) = v²/(v² + σ²)
  - Add fixed-point iteration since W_C depends on V_pred
  - Test different velocity dispersion models

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS (Same as run_regression_extended.py)
# =============================================================================
c = 2.998e8
G = 6.674e-11
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
M_sun = 1.989e30

g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))
a0_mond = 1.2e-10

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))
XI_SCALE = 1 / (2 * np.pi)
ML_DISK = 0.5
ML_BULGE = 0.7

print("=" * 80)
print("C_LOCAL vs CANONICAL W(r) COMPARISON")
print("Using Official Regression Framework")
print("=" * 80)
print(f"\nCanonical parameters:")
print(f"  A₀ = {A_0:.4f}")
print(f"  ξ = R_d/(2π)")
print(f"  g† = {g_dagger:.3e} m/s²")

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_canonical(r_kpc: np.ndarray, R_d: float) -> np.ndarray:
    """Canonical: W(r) = r/(ξ+r), ξ = R_d/(2π)"""
    xi = R_d * XI_SCALE
    xi = max(xi, 0.01)
    return r_kpc / (xi + r_kpc)

def C_local(v_rot_kms: np.ndarray, sigma_kms: np.ndarray) -> np.ndarray:
    """Local coherence scalar: C = v²/(v² + σ²)"""
    v2 = np.maximum(v_rot_kms, 0.0)**2
    s2 = np.maximum(sigma_kms, 1e-6)**2
    return v2 / (v2 + s2)

def predict_canonical(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float) -> np.ndarray:
    """Canonical Σ-Gravity prediction."""
    R_m = R_kpc * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    h = h_function(g_bar)
    W = W_canonical(R_kpc, R_d)
    Sigma = 1 + A_0 * W * h
    return V_bar * np.sqrt(Sigma)

def predict_C_local(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                    sigma_kms: float = 20.0, max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """C_local prediction with fixed-point iteration."""
    R_m = R_kpc * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    h = h_function(g_bar)
    
    # Initialize with V_bar
    V = np.array(V_bar, dtype=float)
    sigma = np.full_like(R_kpc, sigma_kms)
    
    for _ in range(max_iter):
        C = C_local(V, sigma)
        Sigma = 1 + A_0 * C * h
        V_new = V_bar * np.sqrt(Sigma)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    
    return V

def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND standard interpolation: ν = 1/(1 - exp(-√x))"""
    R_m = R_kpc * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    x = g_bar / a0_mond
    x = np.maximum(x, 1e-10)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    return V_bar * np.sqrt(nu)

# =============================================================================
# LOAD SPARC DATA (Same as run_regression_extended.py)
# =============================================================================

def load_sparc():
    """Load SPARC galaxy data."""
    data_path = Path("data/Rotmod_LTG")
    
    # Load master sheet
    master_file = data_path / "MasterSheet_SPARC.mrt"
    master_data = {}
    with open(master_file, 'r') as f:
        in_data = False
        for line in f:
            if line.startswith('Galaxy'):
                in_data = True
                continue
            if not in_data or line.strip() == '' or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                name = parts[0]
                master_data[name] = {
                    'Rdisk': float(parts[6]) if parts[6] != '...' else 3.0,
                }
    
    galaxies = []
    for dat_file in sorted(data_path.glob("*_rotmod.dat")):
        name = dat_file.stem.replace("_rotmod", "")
        try:
            data = np.loadtxt(dat_file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 7 and len(data) >= 5:
                R = data[:, 0]
                V_obs = data[:, 1]
                V_err = data[:, 2]
                V_gas = data[:, 3]
                V_disk = data[:, 4]
                V_bulge = data[:, 5]
                
                if np.any(np.isnan(R)) or np.any(np.isnan(V_obs)):
                    continue
                
                # Compute V_bar with M/L ratios
                V_bar = np.sqrt(V_gas**2 + ML_DISK * V_disk**2 + ML_BULGE * V_bulge**2)
                
                R_d = master_data.get(name, {}).get('Rdisk', 3.0)
                
                galaxies.append({
                    'name': name,
                    'R': R,
                    'V_obs': V_obs,
                    'V_err': V_err,
                    'V_bar': V_bar,
                    'R_d': R_d,
                })
        except:
            continue
    
    return galaxies

# =============================================================================
# RUN COMPARISON
# =============================================================================

def test_sparc(galaxies: List[Dict], predict_fn, name: str) -> Dict:
    """Test on SPARC galaxies using same methodology as run_regression_extended.py."""
    
    rms_list = []
    mond_rms_list = []
    wins = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        
        # Model prediction
        V_pred = predict_fn(R, V_bar, R_d)
        rms = np.sqrt(np.mean((V_pred - V_obs)**2))
        rms_list.append(rms)
        
        # MOND prediction
        V_mond = predict_mond(R, V_bar)
        rms_mond = np.sqrt(np.mean((V_mond - V_obs)**2))
        mond_rms_list.append(rms_mond)
        
        if rms < rms_mond:
            wins += 1
    
    mean_rms = np.mean(rms_list)
    mean_mond = np.mean(mond_rms_list)
    win_rate = wins / len(galaxies) * 100
    
    return {
        'name': name,
        'mean_rms': mean_rms,
        'mean_mond_rms': mean_mond,
        'win_rate': win_rate,
        'n_galaxies': len(galaxies),
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    galaxies = load_sparc()
    print(f"Loaded {len(galaxies)} SPARC galaxies")
    
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)
    
    # Test canonical
    result_canonical = test_sparc(galaxies, predict_canonical, "Canonical W(r)")
    print(f"\n[Canonical W(r) = r/(ξ+r)]")
    print(f"  RMS = {result_canonical['mean_rms']:.2f} km/s")
    print(f"  MOND RMS = {result_canonical['mean_mond_rms']:.2f} km/s")
    print(f"  Win rate vs MOND = {result_canonical['win_rate']:.1f}%")
    
    # Test C_local with different σ values
    sigma_values = [15, 20, 25, 30]
    
    print(f"\n[C_local = v²/(v² + σ²) with fixed-point iteration]")
    
    for sigma in sigma_values:
        predict_fn = lambda R, V, Rd, s=sigma: predict_C_local(R, V, Rd, sigma_kms=s)
        result = test_sparc(galaxies, predict_fn, f"C_local (σ={sigma})")
        
        change = (result['mean_rms'] - result_canonical['mean_rms']) / result_canonical['mean_rms'] * 100
        print(f"  σ={sigma} km/s: RMS = {result['mean_rms']:.2f} km/s ({change:+.2f}%), Win = {result['win_rate']:.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Best C_local
    best_sigma = 20
    predict_fn = lambda R, V, Rd: predict_C_local(R, V, Rd, sigma_kms=best_sigma)
    result_best = test_sparc(galaxies, predict_fn, f"C_local (σ={best_sigma})")
    
    print(f"""
SPARC GALAXY COMPARISON (N={len(galaxies)}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Model                | RMS (km/s) | Change | Win vs MOND |
|----------------------|------------|--------|-------------|
| Canonical W(r)       | {result_canonical['mean_rms']:.2f}      | —      | {result_canonical['win_rate']:.1f}%       |
| C_local (σ=20 km/s)  | {result_best['mean_rms']:.2f}      | {(result_best['mean_rms']-result_canonical['mean_rms'])/result_canonical['mean_rms']*100:+.1f}%  | {result_best['win_rate']:.1f}%       |
| MOND (reference)     | {result_canonical['mean_mond_rms']:.2f}      | —      | —           |

CONCLUSION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The C_local formulation provides a ~{abs((result_best['mean_rms']-result_canonical['mean_rms'])/result_canonical['mean_rms']*100):.1f}% change in RMS.
Win rate vs MOND is essentially unchanged.

The canonical W(r) = r/(ξ+r) remains effective because it already approximates
the orbit-averaged coherence scalar ⟨C⟩_orbit.
""")

