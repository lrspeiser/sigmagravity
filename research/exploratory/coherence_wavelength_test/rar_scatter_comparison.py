#!/usr/bin/env python3
"""
RAR Scatter Comparison: Σ-Gravity h(g) vs MOND ν(g)

This script computes the Radial Acceleration Relation (RAR) scatter for:
1. Σ-Gravity with h(g) = √(g†/g) × g†/(g†+g)
2. MOND with simple interpolating function ν(g) = 1/(1-e^(-√(g/a₀)))

Lower scatter = better fit to the universal RAR.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import os
import glob
from pathlib import Path

# Physical constants
c = 2.998e8  # m/s
H0 = 70 * 1000 / 3.086e22  # 1/s (70 km/s/Mpc)
G = 6.674e-11  # m³/(kg·s²)

# Critical accelerations
g_dagger = c * H0 / (4 * np.sqrt(np.pi))  # Σ-Gravity: 9.59e-11
a0_mond = 1.2e-10  # MOND

print("=" * 80)
print("RAR SCATTER COMPARISON: Σ-GRAVITY vs MOND")
print("=" * 80)

print(f"\nΣ-Gravity g† = {g_dagger:.4e} m/s²")
print(f"MOND a₀ = {a0_mond:.4e} m/s²")

def load_sparc_galaxy(filepath):
    """Load a single SPARC galaxy rotation curve."""
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        R = data[:, 0]  # kpc
        Vobs = data[:, 1]  # km/s
        errV = data[:, 2]  # km/s
        Vgas = data[:, 3]  # km/s
        Vdisk = data[:, 4]  # km/s
        Vbul = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
        
        return {
            'R': R,
            'Vobs': Vobs,
            'errV': errV,
            'Vgas': Vgas,
            'Vdisk': Vdisk,
            'Vbul': Vbul
        }
    except Exception as e:
        return None

def h_sigma_gravity(g, g_dagger):
    """Σ-Gravity enhancement function."""
    ratio = g_dagger / np.maximum(g, 1e-20)
    return np.sqrt(ratio) * g_dagger / (g_dagger + g)

def nu_mond_simple(g, a0):
    """MOND simple interpolating function."""
    x = g / a0
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))

# Load SPARC data
sparc_path = "/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"
galaxy_files = sorted(glob.glob(os.path.join(sparc_path, "*_rotmod.dat")))

print(f"\nLoading {len(galaxy_files)} SPARC galaxies...")

# Collect all data points for RAR
g_bar_all = []
g_obs_all = []
g_sigma_all = []
g_mond_all = []
galaxy_names = []
errV_all = []

A = np.sqrt(3)  # Galaxy amplitude

for gfile in galaxy_files:
    galaxy = load_sparc_galaxy(gfile)
    if galaxy is None or len(galaxy['R']) < 3:
        continue
    
    name = Path(gfile).stem.replace('_rotmod', '')
    
    try:
        # Convert to SI
        R_m = galaxy['R'] * 3.086e19  # kpc to m
        
        # Baryonic velocity squared
        V_bar_sq = galaxy['Vgas']**2 + galaxy['Vdisk']**2 + galaxy['Vbul']**2
        V_bar = np.sqrt(np.maximum(V_bar_sq, 1e-10))  # km/s
        
        # Baryonic acceleration
        g_bar = (V_bar * 1000)**2 / R_m  # m/s²
        
        # Observed acceleration
        g_obs = (galaxy['Vobs'] * 1000)**2 / R_m  # m/s²
        
        # Skip bad data
        if np.any(np.isnan(g_bar)) or np.any(g_bar <= 0) or np.any(g_obs <= 0):
            continue
        
        # Σ-Gravity prediction
        h = h_sigma_gravity(g_bar, g_dagger)
        R_d = np.median(galaxy['R']) / 2
        xi = (2/3) * R_d
        W = 1 - (xi / (xi + galaxy['R']))**0.5
        Sigma = 1 + A * W * h
        g_sigma = g_bar * Sigma
        
        # MOND prediction
        nu = nu_mond_simple(g_bar, a0_mond)
        g_mond = g_bar * nu
        
        # Store
        g_bar_all.extend(g_bar)
        g_obs_all.extend(g_obs)
        g_sigma_all.extend(g_sigma)
        g_mond_all.extend(g_mond)
        galaxy_names.extend([name] * len(g_bar))
        errV_all.extend(galaxy['errV'])
        
    except Exception as e:
        continue

# Convert to arrays
g_bar_all = np.array(g_bar_all)
g_obs_all = np.array(g_obs_all)
g_sigma_all = np.array(g_sigma_all)
g_mond_all = np.array(g_mond_all)
errV_all = np.array(errV_all)

print(f"Total data points: {len(g_bar_all)}")

# Compute RAR scatter in dex
def compute_rar_scatter(g_pred, g_obs, weights=None):
    """Compute RAR scatter in dex (log10 units)."""
    log_residuals = np.log10(g_obs / g_pred)
    
    # Remove infinities
    mask = np.isfinite(log_residuals)
    log_residuals = log_residuals[mask]
    
    if weights is not None:
        weights = weights[mask]
        mean = np.average(log_residuals, weights=weights)
        variance = np.average((log_residuals - mean)**2, weights=weights)
        return np.sqrt(variance)
    
    return np.std(log_residuals)

# Compute scatter
scatter_sigma = compute_rar_scatter(g_sigma_all, g_obs_all)
scatter_mond = compute_rar_scatter(g_mond_all, g_obs_all)

# Also compute weighted scatter using velocity errors
# Error in g_obs ≈ 2 × g_obs × (errV / Vobs)
weights = 1.0 / np.maximum(errV_all, 1.0)**2

scatter_sigma_weighted = compute_rar_scatter(g_sigma_all, g_obs_all, weights)
scatter_mond_weighted = compute_rar_scatter(g_mond_all, g_obs_all, weights)

print("\n" + "=" * 80)
print("RAR SCATTER RESULTS")
print("=" * 80)

print(f"\n{'Metric':<35} {'Σ-Gravity':<15} {'MOND':<15} {'Winner':<15}")
print("-" * 80)
print(f"{'Unweighted scatter (dex)':<35} {scatter_sigma:<15.4f} {scatter_mond:<15.4f} {'Σ-Gravity' if scatter_sigma < scatter_mond else 'MOND'}")
print(f"{'Weighted scatter (dex)':<35} {scatter_sigma_weighted:<15.4f} {scatter_mond_weighted:<15.4f} {'Σ-Gravity' if scatter_sigma_weighted < scatter_mond_weighted else 'MOND'}")

# Compute scatter in different g_bar regimes
print("\n" + "=" * 80)
print("RAR SCATTER BY ACCELERATION REGIME")
print("=" * 80)

regimes = [
    (1e-13, 1e-11, 'Deep MOND (g < 10⁻¹¹)'),
    (1e-11, 1e-10, 'Transition (10⁻¹¹ < g < 10⁻¹⁰)'),
    (1e-10, 1e-9, 'Near-Newtonian (g > 10⁻¹⁰)')
]

print(f"\n{'Regime':<35} {'N pts':<10} {'Σ-Grav (dex)':<15} {'MOND (dex)':<15}")
print("-" * 80)

for g_lo, g_hi, label in regimes:
    mask = (g_bar_all >= g_lo) & (g_bar_all < g_hi)
    n_pts = np.sum(mask)
    
    if n_pts > 10:
        scatter_s = compute_rar_scatter(g_sigma_all[mask], g_obs_all[mask])
        scatter_m = compute_rar_scatter(g_mond_all[mask], g_obs_all[mask])
        print(f"{label:<35} {n_pts:<10} {scatter_s:<15.4f} {scatter_m:<15.4f}")

# Compute mean residuals (bias)
print("\n" + "=" * 80)
print("MEAN RESIDUALS (BIAS)")
print("=" * 80)

mean_residual_sigma = np.mean(np.log10(g_obs_all / g_sigma_all))
mean_residual_mond = np.mean(np.log10(g_obs_all / g_mond_all))

print(f"\n{'Model':<20} {'Mean log(g_obs/g_pred) [dex]':<30}")
print("-" * 50)
print(f"{'Σ-Gravity':<20} {mean_residual_sigma:<+30.4f}")
print(f"{'MOND':<20} {mean_residual_mond:<+30.4f}")

print("\n(Positive = model under-predicts, Negative = model over-predicts)")

# Compare to literature
print("\n" + "=" * 80)
print("COMPARISON TO LITERATURE")
print("=" * 80)

print(f"""
McGaugh et al. (2016) reported RAR scatter of ~0.13 dex for SPARC.
Lelli et al. (2017) found intrinsic scatter ~0.10 dex after accounting for errors.

Our results:
  Σ-Gravity: {scatter_sigma:.3f} dex
  MOND:      {scatter_mond:.3f} dex

{'Σ-Gravity achieves lower scatter than MOND!' if scatter_sigma < scatter_mond else 'MOND achieves lower scatter than Σ-Gravity.'}
""")

print("=" * 80)
print("CONCLUSION")
print("=" * 80)

improvement = (scatter_mond - scatter_sigma) / scatter_mond * 100

print(f"""
Σ-Gravity with g† = cH₀/(4√π) achieves:
  - RAR scatter: {scatter_sigma:.4f} dex
  - {'Lower' if scatter_sigma < scatter_mond else 'Higher'} scatter than MOND by {abs(improvement):.1f}%

This demonstrates that the Σ-Gravity enhancement function h(g) = √(g†/g) × g†/(g†+g)
{'provides a better description' if scatter_sigma < scatter_mond else 'provides a comparable description'} 
of the universal RAR compared to MOND's simple interpolating function.
""")

print("=" * 80)

