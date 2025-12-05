#!/usr/bin/env python3
"""
Precision g† Fit on Full SPARC Dataset

This script finds the best-fit value of g† and compares it to:
1. The theoretical prediction: g† = cH₀/(4√π) = 9.59×10⁻¹¹ m/s²
2. The MOND empirical value: a₀ = 1.2×10⁻¹⁰ m/s²

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import os
import glob
from pathlib import Path
from scipy.optimize import minimize_scalar

# Physical constants
c = 2.998e8  # m/s
H0 = 70 * 1000 / 3.086e22  # 1/s (70 km/s/Mpc)
G = 6.674e-11  # m³/(kg·s²)

# Theoretical predictions
g_dagger_theory = c * H0 / (4 * np.sqrt(np.pi))  # 9.59e-11
a0_mond = 1.2e-10  # MOND empirical

print("=" * 80)
print("PRECISION g† FIT ON FULL SPARC DATASET")
print("=" * 80)

print(f"\nTheoretical prediction: g† = cH₀/(4√π) = {g_dagger_theory:.4e} m/s²")
print(f"MOND empirical value: a₀ = {a0_mond:.4e} m/s²")

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

def compute_total_chi2(g_dagger, galaxies, A=np.sqrt(3)):
    """Compute total chi-squared for all galaxies."""
    total_chi2 = 0
    n_points = 0
    
    for galaxy in galaxies:
        R_m = galaxy['R'] * 3.086e19  # kpc to m
        
        V_bar_sq = galaxy['Vgas']**2 + galaxy['Vdisk']**2 + galaxy['Vbul']**2
        V_bar = np.sqrt(np.maximum(V_bar_sq, 1e-10))  # km/s
        
        g_bar = (V_bar * 1000)**2 / R_m  # m/s²
        
        if np.any(np.isnan(g_bar)) or np.any(g_bar <= 0):
            continue
        
        # Σ-Gravity prediction
        h = h_sigma_gravity(g_bar, g_dagger)
        R_d = np.median(galaxy['R']) / 2
        xi = (2/3) * R_d
        W = 1 - (xi / (xi + galaxy['R']))**0.5
        Sigma = 1 + A * W * h
        V_pred = V_bar * np.sqrt(Sigma)
        
        # Chi-squared
        err = np.maximum(galaxy['errV'], 1.0)  # Minimum 1 km/s error
        chi2 = np.sum(((galaxy['Vobs'] - V_pred) / err)**2)
        
        total_chi2 += chi2
        n_points += len(galaxy['R'])
    
    return total_chi2, n_points

# Load SPARC data
sparc_path = "/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"
galaxy_files = sorted(glob.glob(os.path.join(sparc_path, "*_rotmod.dat")))

print(f"\nLoading {len(galaxy_files)} SPARC galaxies...")

galaxies = []
for gfile in galaxy_files:
    galaxy = load_sparc_galaxy(gfile)
    if galaxy is not None and len(galaxy['R']) >= 3:
        galaxies.append(galaxy)

print(f"Loaded {len(galaxies)} valid galaxies")

# Grid search for best g†
print("\n" + "=" * 80)
print("GRID SEARCH FOR OPTIMAL g†")
print("=" * 80)

g_dagger_range = np.logspace(-11.5, -9.5, 100)
chi2_values = []

for g_d in g_dagger_range:
    chi2, n_pts = compute_total_chi2(g_d, galaxies)
    chi2_values.append(chi2)

chi2_values = np.array(chi2_values)
best_idx = np.argmin(chi2_values)
best_g_dagger = g_dagger_range[best_idx]

print(f"\nBest-fit g† = {best_g_dagger:.4e} m/s²")
print(f"Minimum χ² = {chi2_values[best_idx]:.1f}")

# Fine-tune with optimization
def objective(log_g_dagger):
    g_d = 10**log_g_dagger
    chi2, _ = compute_total_chi2(g_d, galaxies)
    return chi2

result = minimize_scalar(objective, bounds=(-11.5, -9.5), method='bounded')
best_g_dagger_opt = 10**result.x

print(f"\nOptimized g† = {best_g_dagger_opt:.4e} m/s²")
print(f"Optimized χ² = {result.fun:.1f}")

# Compare to predictions
print("\n" + "=" * 80)
print("COMPARISON TO PREDICTIONS")
print("=" * 80)

chi2_theory, n_pts = compute_total_chi2(g_dagger_theory, galaxies)
chi2_mond, _ = compute_total_chi2(a0_mond, galaxies)
chi2_best, _ = compute_total_chi2(best_g_dagger_opt, galaxies)

print(f"\n{'Model':<30} {'g† [m/s²]':<18} {'χ²':<15} {'Δχ²':<15}")
print("-" * 80)
print(f"{'Best fit':<30} {best_g_dagger_opt:<18.4e} {chi2_best:<15.1f} {0:<15.1f}")
print(f"{'Theory: cH₀/(4√π)':<30} {g_dagger_theory:<18.4e} {chi2_theory:<15.1f} {chi2_theory-chi2_best:<+15.1f}")
print(f"{'MOND: a₀':<30} {a0_mond:<18.4e} {chi2_mond:<15.1f} {chi2_mond-chi2_best:<+15.1f}")

# Calculate ratios
print(f"\n{'Ratios:':<30}")
print(f"  Best-fit / Theory: {best_g_dagger_opt / g_dagger_theory:.3f}")
print(f"  Best-fit / MOND a₀: {best_g_dagger_opt / a0_mond:.3f}")
print(f"  Theory / MOND a₀: {g_dagger_theory / a0_mond:.3f}")

# Statistical significance
dof = n_pts - 1  # degrees of freedom (1 parameter)
print(f"\n{'Statistical Analysis:':<30}")
print(f"  Total data points: {n_pts}")
print(f"  Reduced χ² (best): {chi2_best/dof:.3f}")
print(f"  Reduced χ² (theory): {chi2_theory/dof:.3f}")
print(f"  Reduced χ² (MOND): {chi2_mond/dof:.3f}")

# Confidence interval (Δχ² = 1 for 68% CI on 1 parameter)
print("\n" + "=" * 80)
print("CONFIDENCE INTERVAL")
print("=" * 80)

# Find 68% CI (Δχ² = 1)
target_chi2 = chi2_best + 1.0
g_lower = g_dagger_range[np.where(chi2_values < target_chi2)[0][0]]
g_upper = g_dagger_range[np.where(chi2_values < target_chi2)[0][-1]]

print(f"\n68% Confidence Interval:")
print(f"  g† = {best_g_dagger_opt:.4e} (+{g_upper-best_g_dagger_opt:.2e}, -{best_g_dagger_opt-g_lower:.2e}) m/s²")
print(f"  = {best_g_dagger_opt:.4e} ± {(g_upper-g_lower)/2:.2e} m/s²")

# Check if theory is within CI
if g_lower <= g_dagger_theory <= g_upper:
    print(f"\n✓ Theoretical prediction g† = cH₀/(4√π) is WITHIN 68% CI")
else:
    sigma_away = abs(best_g_dagger_opt - g_dagger_theory) / ((g_upper - g_lower) / 2)
    print(f"\n✗ Theoretical prediction is {sigma_away:.1f}σ away from best fit")

if g_lower <= a0_mond <= g_upper:
    print(f"✓ MOND a₀ is WITHIN 68% CI")
else:
    sigma_away = abs(best_g_dagger_opt - a0_mond) / ((g_upper - g_lower) / 2)
    print(f"✗ MOND a₀ is {sigma_away:.1f}σ away from best fit")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
Best-fit g† = {best_g_dagger_opt:.4e} m/s²

Comparison:
  - Theory (cH₀/4√π): {g_dagger_theory:.4e} m/s² → {100*g_dagger_theory/best_g_dagger_opt:.1f}% of best-fit
  - MOND (a₀):        {a0_mond:.4e} m/s² → {100*a0_mond/best_g_dagger_opt:.1f}% of best-fit

The theoretical prediction g† = cH₀/(4√π) is:
  - Within {abs(best_g_dagger_opt - g_dagger_theory) / best_g_dagger_opt * 100:.1f}% of the best-fit value
  - Δχ² = {chi2_theory - chi2_best:.1f} compared to best-fit

This is {'excellent' if abs(chi2_theory - chi2_best) < 10 else 'good' if abs(chi2_theory - chi2_best) < 50 else 'moderate'} agreement.
""")

print("=" * 80)

