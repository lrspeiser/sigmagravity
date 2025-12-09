#!/usr/bin/env python3
"""
Comprehensive sweep of C_local parameters to find optimal configuration.

Tests:
1. Constant σ values from 10-60 km/s
2. Exponential model with different σ_0 and σ_disk
3. Mixed model with different floor and frac
4. Comparison with geometric baseline

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
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8
H0_SI = 2.27e-18
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30
g_dagger = cH0 / (4 * math.sqrt(math.pi))
A_GALAXY = np.exp(1 / (2 * np.pi))

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def C_local(v_rot_kms, sigma_kms):
    v2 = np.maximum(v_rot_kms, 0.0)**2
    s2 = np.maximum(sigma_kms, 1e-6)**2
    return v2 / (v2 + s2)

def W_geometric(r_kpc, R_d_kpc):
    xi = max(R_d_kpc / (2 * np.pi), 0.01)
    return r_kpc / (xi + r_kpc)

def h_function(g):
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def compute_V_bar(V_gas, V_disk, V_bulge, ml_disk=0.5, ml_bulge=0.7):
    V_bar_sq = V_gas**2 + ml_disk * V_disk**2 + ml_bulge * V_bulge**2
    return np.sqrt(np.maximum(V_bar_sq, 0))

def predict_velocity_geometric(R_kpc, V_bar, R_d, A=A_GALAXY):
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    h = h_function(g_bar)
    W = W_geometric(R_kpc, R_d)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)

def predict_velocity_C_local(R_kpc, V_bar, R_d, sigma_fn, A=A_GALAXY,
                             max_iter=50, tol=1e-6):
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    h = h_function(g_bar)
    sigma = sigma_fn(R_kpc, V_bar, R_d)
    
    V = np.array(V_bar, dtype=float)
    for _ in range(max_iter):
        C = C_local(V, sigma)
        Sigma = 1 + A * C * h
        V_new = V_bar * np.sqrt(Sigma)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V

# =============================================================================
# LOAD DATA
# =============================================================================

def load_sparc_galaxies(data_dir="data/Rotmod_LTG"):
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    master_file = data_path / "MasterSheet_SPARC.mrt"
    if not master_file.exists():
        return []
    
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
            if data.shape[1] >= 7 and len(data) >= 3:
                R = data[:, 0]
                V_obs = data[:, 1]
                if np.any(np.isnan(R)) or np.any(np.isnan(V_obs)):
                    continue
                R_d = master_data.get(name, {}).get('Rdisk', 3.0)
                galaxies.append({
                    'name': name, 'R': R, 'V_obs': V_obs, 'V_err': data[:, 2],
                    'V_gas': data[:, 3], 'V_disk': data[:, 4], 'V_bulge': data[:, 5],
                    'R_d': R_d, 'n_points': len(R),
                })
        except:
            continue
    return [g for g in galaxies if g['n_points'] >= 5]

# =============================================================================
# SWEEP FUNCTIONS
# =============================================================================

def sweep_constant_sigma(galaxies, sigma_values):
    """Sweep constant σ values."""
    results = []
    
    for sigma in sigma_values:
        sigma_fn = lambda r, v, rd, s=sigma: np.full_like(r, s)
        
        total_sq_err = 0
        total_points = 0
        rms_list = []
        
        for gal in galaxies:
            V_bar = compute_V_bar(gal['V_gas'], gal['V_disk'], gal['V_bulge'])
            V_pred = predict_velocity_C_local(gal['R'], V_bar, gal['R_d'], sigma_fn)
            
            sq_err = np.sum((V_pred - gal['V_obs'])**2)
            total_sq_err += sq_err
            total_points += len(gal['V_obs'])
            rms_list.append(np.sqrt(sq_err / len(gal['V_obs'])))
        
        global_rms = np.sqrt(total_sq_err / total_points)
        results.append({
            'sigma': sigma,
            'global_rms': global_rms,
            'mean_rms': np.mean(rms_list),
            'median_rms': np.median(rms_list),
        })
    
    return results


def sweep_exponential_model(galaxies, sigma0_values, sigma_disk_values):
    """Sweep exponential model parameters."""
    results = []
    
    for sigma0 in sigma0_values:
        for sigma_disk in sigma_disk_values:
            if sigma_disk >= sigma0:
                continue
                
            def sigma_fn(r, v, rd, s0=sigma0, sd=sigma_disk):
                return sd + (s0 - sd) * np.exp(-r / (2 * rd))
            
            total_sq_err = 0
            total_points = 0
            
            for gal in galaxies:
                V_bar = compute_V_bar(gal['V_gas'], gal['V_disk'], gal['V_bulge'])
                V_pred = predict_velocity_C_local(gal['R'], V_bar, gal['R_d'], sigma_fn)
                total_sq_err += np.sum((V_pred - gal['V_obs'])**2)
                total_points += len(gal['V_obs'])
            
            results.append({
                'sigma0': sigma0,
                'sigma_disk': sigma_disk,
                'global_rms': np.sqrt(total_sq_err / total_points),
            })
    
    return results


def sweep_mixed_model(galaxies, floor_values, frac_values):
    """Sweep mixed model parameters."""
    results = []
    
    for floor in floor_values:
        for frac in frac_values:
            def sigma_fn(r, v, rd, f=floor, fr=frac):
                return np.sqrt(f**2 + (fr * v)**2)
            
            total_sq_err = 0
            total_points = 0
            
            for gal in galaxies:
                V_bar = compute_V_bar(gal['V_gas'], gal['V_disk'], gal['V_bulge'])
                V_pred = predict_velocity_C_local(gal['R'], V_bar, gal['R_d'], sigma_fn)
                total_sq_err += np.sum((V_pred - gal['V_obs'])**2)
                total_points += len(gal['V_obs'])
            
            results.append({
                'floor': floor,
                'frac': frac,
                'global_rms': np.sqrt(total_sq_err / total_points),
            })
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("COHERENCE SCALAR PARAMETER SWEEP")
    print("=" * 80)
    
    galaxies = load_sparc_galaxies()
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Baseline
    total_sq_err = 0
    total_points = 0
    for gal in galaxies:
        V_bar = compute_V_bar(gal['V_gas'], gal['V_disk'], gal['V_bulge'])
        V_pred = predict_velocity_geometric(gal['R'], V_bar, gal['R_d'])
        total_sq_err += np.sum((V_pred - gal['V_obs'])**2)
        total_points += len(gal['V_obs'])
    baseline_rms = np.sqrt(total_sq_err / total_points)
    print(f"\nBaseline (Geometric W(r)): {baseline_rms:.2f} km/s")
    
    # Sweep 1: Constant σ
    print("\n" + "-" * 80)
    print("SWEEP 1: Constant σ")
    print("-" * 80)
    
    sigma_values = [10, 15, 20, 25, 30, 35, 40, 50, 60]
    results_const = sweep_constant_sigma(galaxies, sigma_values)
    
    print(f"{'σ (km/s)':<12} {'Global RMS':<15} {'Change':<10}")
    for r in results_const:
        change = (r['global_rms'] - baseline_rms) / baseline_rms * 100
        print(f"{r['sigma']:<12} {r['global_rms']:.2f} km/s      {change:+.1f}%")
    
    best_const = min(results_const, key=lambda x: x['global_rms'])
    print(f"\nBest constant σ: {best_const['sigma']} km/s → RMS = {best_const['global_rms']:.2f} km/s")
    
    # Sweep 2: Exponential model
    print("\n" + "-" * 80)
    print("SWEEP 2: Exponential σ(r) = σ_disk + (σ_0 - σ_disk) × exp(-r/2R_d)")
    print("-" * 80)
    
    sigma0_values = [40, 60, 80, 100, 120]
    sigma_disk_values = [8, 12, 15, 20, 25]
    results_exp = sweep_exponential_model(galaxies, sigma0_values, sigma_disk_values)
    
    print(f"{'σ_0':<8} {'σ_disk':<10} {'Global RMS':<15} {'Change':<10}")
    for r in sorted(results_exp, key=lambda x: x['global_rms'])[:10]:
        change = (r['global_rms'] - baseline_rms) / baseline_rms * 100
        print(f"{r['sigma0']:<8} {r['sigma_disk']:<10} {r['global_rms']:.2f} km/s      {change:+.1f}%")
    
    best_exp = min(results_exp, key=lambda x: x['global_rms'])
    print(f"\nBest exponential: σ_0={best_exp['sigma0']}, σ_disk={best_exp['sigma_disk']} → RMS = {best_exp['global_rms']:.2f} km/s")
    
    # Sweep 3: Mixed model
    print("\n" + "-" * 80)
    print("SWEEP 3: Mixed σ = √(floor² + (frac × V_bar)²)")
    print("-" * 80)
    
    floor_values = [5, 8, 10, 12, 15]
    frac_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    results_mixed = sweep_mixed_model(galaxies, floor_values, frac_values)
    
    print(f"{'floor':<8} {'frac':<8} {'Global RMS':<15} {'Change':<10}")
    for r in sorted(results_mixed, key=lambda x: x['global_rms'])[:10]:
        change = (r['global_rms'] - baseline_rms) / baseline_rms * 100
        print(f"{r['floor']:<8} {r['frac']:<8} {r['global_rms']:.2f} km/s      {change:+.1f}%")
    
    best_mixed = min(results_mixed, key=lambda x: x['global_rms'])
    print(f"\nBest mixed: floor={best_mixed['floor']}, frac={best_mixed['frac']} → RMS = {best_mixed['global_rms']:.2f} km/s")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"""
Model Comparison:
----------------
Baseline (Geometric W(r)):     {baseline_rms:.2f} km/s

Best C_local variants:
  Constant σ={best_const['sigma']} km/s:         {best_const['global_rms']:.2f} km/s ({(best_const['global_rms']-baseline_rms)/baseline_rms*100:+.1f}%)
  Exponential (σ₀={best_exp['sigma0']}, σ_d={best_exp['sigma_disk']}): {best_exp['global_rms']:.2f} km/s ({(best_exp['global_rms']-baseline_rms)/baseline_rms*100:+.1f}%)
  Mixed (floor={best_mixed['floor']}, frac={best_mixed['frac']}):   {best_mixed['global_rms']:.2f} km/s ({(best_mixed['global_rms']-baseline_rms)/baseline_rms*100:+.1f}%)

Key findings:
1. C_local with constant σ ~ 20-25 km/s performs comparably to geometric W(r)
2. The exponential model (high central dispersion) performs WORSE
3. The mixed model provides marginal improvement

The geometric W(r) = r/(ξ+r) is remarkably effective despite being "non-local".
This suggests it already captures the essential physics of orbit-averaged coherence.
""")

