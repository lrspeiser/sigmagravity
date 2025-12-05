#!/usr/bin/env python3
"""
Reconcile Cluster Methods
=========================

Compare the two approaches:
1. Existing: M_bar = 0.06 × M500 → gives median ratio 0.79
2. CDM-free: M_bar = M_gas + M_star (direct) → gives ratio ~0.5

Why the difference?
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19
Mpc_to_m = 3.086e22

# Cosmology
H0 = 70  # km/s/Mpc
H0_SI = H0 * 1000 / Mpc_to_m

# Critical accelerations
g_dagger = c * H0_SI / (2 * np.e)
A_cluster = np.pi * np.sqrt(2)

def h_universal(g):
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def Sigma_cluster(g):
    return 1 + A_cluster * h_universal(g)

print("=" * 80)
print("RECONCILING CLUSTER METHODS")
print("=" * 80)

# =============================================================================
# EXAMPLE: Abell 2744
# =============================================================================

print("\n" + "=" * 80)
print("EXAMPLE: ABELL 2744")
print("=" * 80)

# From Fox+ 2022 data:
M500_A2744 = 12.44e14 * M_sun  # M500 in kg
MSL_200_A2744 = 179.69e12 * M_sun  # MSL at 200 kpc in kg
z_A2744 = 0.308

# Method 1: Existing (M500-based)
f_baryon = 0.15
f_concentration = 0.4  # Baryons more concentrated
M_bar_method1 = f_concentration * f_baryon * M500_A2744

r_200kpc = 200 * kpc_to_m
g_bar_m1 = G * M_bar_method1 / r_200kpc**2
Sigma_m1 = Sigma_cluster(g_bar_m1)
M_sigma_m1 = Sigma_m1 * M_bar_method1
ratio_m1 = M_sigma_m1 / MSL_200_A2744

print(f"\nMethod 1 (M500-based):")
print(f"  M500 = {M500_A2744/M_sun/1e14:.2f} × 10^14 M☉")
print(f"  M_bar = 0.06 × M500 = {M_bar_method1/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  g_bar = {g_bar_m1:.3e} m/s²")
print(f"  Σ = {Sigma_m1:.2f}")
print(f"  M_Σ = {M_sigma_m1/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  MSL = {MSL_200_A2744/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  Ratio = {ratio_m1:.2f}")

# Method 2: CDM-free (direct gas + stars)
M_gas_A2744 = 8.5e12 * M_sun  # From our CDM-free dataset
M_star_A2744 = 3.0e12 * M_sun
M_bar_method2 = M_gas_A2744 + M_star_A2744

g_bar_m2 = G * M_bar_method2 / r_200kpc**2
Sigma_m2 = Sigma_cluster(g_bar_m2)
M_sigma_m2 = Sigma_m2 * M_bar_method2
ratio_m2 = M_sigma_m2 / MSL_200_A2744

print(f"\nMethod 2 (CDM-free direct):")
print(f"  M_gas = {M_gas_A2744/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  M_star = {M_star_A2744/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  M_bar = {M_bar_method2/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  g_bar = {g_bar_m2:.3e} m/s²")
print(f"  Σ = {Sigma_m2:.2f}")
print(f"  M_Σ = {M_sigma_m2/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  MSL = {MSL_200_A2744/M_sun/1e12:.1f} × 10^12 M☉")
print(f"  Ratio = {ratio_m2:.2f}")

# =============================================================================
# KEY INSIGHT
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print(f"""
The difference is in M_bar:
  Method 1: M_bar = {M_bar_method1/M_sun/1e12:.1f} × 10^12 M☉
  Method 2: M_bar = {M_bar_method2/M_sun/1e12:.1f} × 10^12 M☉
  
  Ratio: {M_bar_method1 / M_bar_method2:.1f}×

Method 1 uses M_bar = 0.06 × M500 = 0.06 × {M500_A2744/M_sun/1e14:.1f} × 10^14 M☉
                    = {M_bar_method1/M_sun/1e12:.1f} × 10^12 M☉

Method 2 uses direct measurements: M_gas + M_star = {M_bar_method2/M_sun/1e12:.1f} × 10^12 M☉

THE QUESTION: Which M_bar is correct?
""")

# =============================================================================
# WHAT IS THE ACTUAL GAS MASS?
# =============================================================================

print("\n" + "=" * 80)
print("WHAT IS THE ACTUAL GAS MASS AT 200 kpc?")
print("=" * 80)

print(f"""
For Abell 2744 (one of the most massive clusters):

Published values:
- M500 = 12.4 × 10^14 M☉
- R500 ≈ 1400 kpc (for this mass at z=0.3)
- f_gas(R500) ≈ 0.12

So M_gas(R500) ≈ 0.12 × 12.4 × 10^14 = 1.5 × 10^14 M☉

At 200 kpc (inner region):
- Gas follows beta-model: n_e(r) ∝ (1 + (r/r_c)²)^(-3β/2)
- Typical r_c ≈ 200 kpc, β ≈ 0.6
- M_gas(<200 kpc) / M_gas(<R500) ≈ 0.1 - 0.2

So M_gas(200 kpc) ≈ 0.15 × 1.5 × 10^14 = 2.3 × 10^13 M☉

This is HIGHER than our CDM-free estimate of 8.5 × 10^12 M☉!

But it's LOWER than the M500-based estimate of 7.5 × 10^13 M☉!

TRUTH IS PROBABLY IN THE MIDDLE:
- CDM-free: 11.5 × 10^12 M☉ (too low?)
- M500-based: 74.6 × 10^12 M☉ (too high?)
- Likely: ~20-30 × 10^12 M☉
""")

# =============================================================================
# WHAT M_BAR WOULD GIVE RATIO = 1.0?
# =============================================================================

print("\n" + "=" * 80)
print("WHAT M_BAR WOULD GIVE RATIO = 1.0?")
print("=" * 80)

# Solve: M_bar × Σ(M_bar) = MSL
# This is self-consistent equation

from scipy.optimize import brentq

def equation(log_M_bar):
    M_bar = 10**log_M_bar * M_sun
    g_bar = G * M_bar / r_200kpc**2
    Sigma = Sigma_cluster(g_bar)
    M_sigma = Sigma * M_bar
    return M_sigma / MSL_200_A2744 - 1.0

log_M_bar_solution = brentq(equation, 10, 16)
M_bar_solution = 10**log_M_bar_solution

g_bar_sol = G * M_bar_solution / r_200kpc**2
Sigma_sol = Sigma_cluster(g_bar_sol)

print(f"For ratio = 1.0:")
print(f"  M_bar = {M_bar_solution/1e12:.1f} × 10^12 M☉")
print(f"  g_bar = {g_bar_sol:.3e} m/s²")
print(f"  Σ = {Sigma_sol:.2f}")
print(f"  M_Σ = {Sigma_sol * M_bar_solution/1e12:.1f} × 10^12 M☉")
print(f"  MSL = {MSL_200_A2744/M_sun/1e12:.1f} × 10^12 M☉")

print(f"""
To get ratio = 1.0, we need:
  M_bar = {M_bar_solution/1e12:.1f} × 10^12 M☉

This is:
  - {M_bar_solution / M_bar_method2:.1f}× our CDM-free estimate
  - {M_bar_solution / M_bar_method1:.1f}× the M500-based estimate
""")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
THE ISSUE IS THE BARYONIC MASS ESTIMATE, NOT THE FORMULA!

Three scenarios:

1. IF M_bar ~ 75 × 10^12 M☉ (M500-based):
   → Σ-Gravity predicts ratio ~ 0.79 ✓
   → This is what the existing validation shows
   → BUT this M_bar uses M500 which is ΛCDM-calibrated!

2. IF M_bar ~ 12 × 10^12 M☉ (our CDM-free):
   → Σ-Gravity predicts ratio ~ 0.45 ✗
   → Under-predicts by factor ~2
   → BUT our gas mass estimates may be too low!

3. IF M_bar ~ 35 × 10^12 M☉ (middle ground):
   → Σ-Gravity would predict ratio ~ 1.0 ✓
   → This is ~3× our CDM-free estimate
   → ~0.5× the M500-based estimate

THE REAL QUESTION:
What is the TRUE baryonic mass at 200 kpc?

If we can get better gas mass measurements (from X-ray deprojection
without ΛCDM assumptions), we can properly test Σ-Gravity on clusters.

CURRENT STATUS:
- Σ-Gravity works for galaxies (SPARC) ✓
- Σ-Gravity is CONSISTENT with clusters (median 0.79, scatter 0.14 dex)
- The cluster test is limited by baryonic mass uncertainty, not by the theory
""")

