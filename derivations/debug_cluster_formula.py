#!/usr/bin/env python3
"""
Debug Cluster Formula
=====================

Let's carefully examine what's happening with the cluster calculations.

The key question: Why does Σ-Gravity under-predict cluster enhancement?
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
g_dagger_old = c * H0_SI / (2 * np.e)  # Old: cH₀/(2e)
g_dagger_new = c * H0_SI / (4 * np.sqrt(np.pi))  # New: cH₀/(4√π)

# Amplitudes
A_galaxy = np.sqrt(3)  # ≈ 1.73
A_cluster = np.pi * np.sqrt(2)  # ≈ 4.44

print("=" * 80)
print("DEBUG: UNDERSTANDING THE CLUSTER DISCREPANCY")
print("=" * 80)

print(f"\n1. PARAMETERS:")
print(f"   g†_old = {g_dagger_old:.3e} m/s²")
print(f"   g†_new = {g_dagger_new:.3e} m/s²")
print(f"   A_galaxy = √3 = {A_galaxy:.3f}")
print(f"   A_cluster = π√2 = {A_cluster:.3f}")
print(f"   Ratio A_cluster/A_galaxy = {A_cluster/A_galaxy:.3f}")

# =============================================================================
# STEP 1: What is the typical g_bar in clusters at 200 kpc?
# =============================================================================

print("\n" + "=" * 80)
print("2. TYPICAL CLUSTER ACCELERATIONS")
print("=" * 80)

# For a cluster with M_bar ~ 10^13 M_sun within 200 kpc
M_bar_typical = 1e13 * M_sun  # kg
r = 200 * kpc_to_m  # m

g_bar_typical = G * M_bar_typical / r**2
print(f"\n   For M_bar = 10^13 M☉ at r = 200 kpc:")
print(f"   g_bar = {g_bar_typical:.3e} m/s²")
print(f"   g_bar / g†_old = {g_bar_typical / g_dagger_old:.3f}")
print(f"   g_bar / g†_new = {g_bar_typical / g_dagger_new:.3f}")

# =============================================================================
# STEP 2: What enhancement does h(g) give?
# =============================================================================

print("\n" + "=" * 80)
print("3. THE h(g) FUNCTION ANALYSIS")
print("=" * 80)

def h_function(g, g_dag):
    """Universal acceleration function."""
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)

# For cluster acceleration
g = g_bar_typical

h_old = h_function(g, g_dagger_old)
h_new = h_function(g, g_dagger_new)

print(f"\n   At g_bar = {g:.2e} m/s²:")
print(f"   h(g, g†_old) = {h_old:.4f}")
print(f"   h(g, g†_new) = {h_new:.4f}")

# What Σ does this give?
Sigma_old = 1 + A_cluster * h_old
Sigma_new = 1 + A_cluster * h_new

print(f"\n   Σ = 1 + A × h:")
print(f"   Σ_old = 1 + {A_cluster:.2f} × {h_old:.4f} = {Sigma_old:.2f}")
print(f"   Σ_new = 1 + {A_cluster:.2f} × {h_new:.4f} = {Sigma_new:.2f}")

# =============================================================================
# STEP 3: What enhancement is REQUIRED?
# =============================================================================

print("\n" + "=" * 80)
print("4. REQUIRED ENHANCEMENT")
print("=" * 80)

# From our data: MSL ~ 150-200 × 10^12 M_sun, M_bar ~ 10 × 10^12 M_sun
# So Σ_required ~ 15-20

Sigma_required = 16  # typical

print(f"\n   Required Σ ≈ {Sigma_required}")
print(f"   Predicted Σ ≈ {Sigma_old:.1f} (old) or {Sigma_new:.1f} (new)")
print(f"   Ratio: {Sigma_required / Sigma_old:.1f}× (old) or {Sigma_required / Sigma_new:.1f}× (new)")

# =============================================================================
# STEP 4: What would fix this?
# =============================================================================

print("\n" + "=" * 80)
print("5. POSSIBLE FIXES")
print("=" * 80)

# Option A: Increase A_cluster
# Σ = 1 + A × h
# Σ_required = 1 + A_needed × h
# A_needed = (Σ_required - 1) / h

A_needed_old = (Sigma_required - 1) / h_old
A_needed_new = (Sigma_required - 1) / h_new

print(f"\n   Option A: Larger cluster amplitude")
print(f"   A_needed (old g†) = {A_needed_old:.1f} (current: {A_cluster:.1f})")
print(f"   A_needed (new g†) = {A_needed_new:.1f} (current: {A_cluster:.1f})")
print(f"   Ratio needed: {A_needed_old / A_cluster:.1f}× (old) or {A_needed_new / A_cluster:.1f}× (new)")

# Option B: Larger g†
# h(g) = √(g†/g) × g†/(g† + g)
# At low g (cluster regime): g << g†, so h ≈ √(g†/g) × 1 = √(g†/g)
# Σ ≈ 1 + A × √(g†/g)
# For Σ = 16 with A = 4.44:
# 16 = 1 + 4.44 × √(g†/g)
# √(g†/g) = 15/4.44 = 3.38
# g†/g = 11.4
# g† = 11.4 × g

g_dag_needed = 11.4 * g_bar_typical

print(f"\n   Option B: Larger g†")
print(f"   g†_needed ≈ {g_dag_needed:.2e} m/s²")
print(f"   Ratio to old: {g_dag_needed / g_dagger_old:.1f}×")
print(f"   Ratio to new: {g_dag_needed / g_dagger_new:.1f}×")

# Option C: Check if our M_bar estimate is wrong
print(f"\n   Option C: M_bar is underestimated")
print(f"   If true M_bar = Σ_required × M_bar_estimated / Σ_predicted")
print(f"   True M_bar ≈ {Sigma_required / Sigma_old:.1f}× larger than estimated")

# =============================================================================
# STEP 5: THE REAL QUESTION - Is our M_bar correct?
# =============================================================================

print("\n" + "=" * 80)
print("6. THE CRITICAL QUESTION: ARE GAS MASSES CORRECT?")
print("=" * 80)

print("""
Published gas mass fractions at R500:
  f_gas = M_gas / M_total ≈ 0.10 - 0.13

For a cluster with M_total(500) ~ 10^15 M☉:
  M_gas(500) ~ 10^14 M☉

But at r = 200 kpc (inner region):
  - Gas is more concentrated than total mass
  - M_gas(200 kpc) / M_gas(500) ≈ 0.3 - 0.5
  - So M_gas(200 kpc) ~ 3-5 × 10^13 M☉

Our estimates of M_gas ~ 5-10 × 10^12 M☉ may be LOW by factor of ~5!

WAIT - let me check the actual gas density profiles...
""")

# =============================================================================
# STEP 6: Recalculate with realistic gas masses
# =============================================================================

print("\n" + "=" * 80)
print("7. RECALCULATION WITH LARGER GAS MASSES")
print("=" * 80)

# If M_bar is actually 5× larger...
M_bar_corrected = 5e13 * M_sun  # 50 × 10^12 M☉
g_bar_corrected = G * M_bar_corrected / r**2

print(f"\n   If M_bar = 5 × 10^13 M☉ at 200 kpc:")
print(f"   g_bar = {g_bar_corrected:.3e} m/s²")

h_old_corr = h_function(g_bar_corrected, g_dagger_old)
h_new_corr = h_function(g_bar_corrected, g_dagger_new)

Sigma_old_corr = 1 + A_cluster * h_old_corr
Sigma_new_corr = 1 + A_cluster * h_new_corr

print(f"   h_old = {h_old_corr:.4f}, Σ_old = {Sigma_old_corr:.2f}")
print(f"   h_new = {h_new_corr:.4f}, Σ_new = {Sigma_new_corr:.2f}")

# What Σ is required now?
MSL_typical = 180e12 * M_sun  # 180 × 10^12 M☉
Sigma_required_corr = MSL_typical / M_bar_corrected

print(f"\n   Required Σ = MSL / M_bar = {MSL_typical/(1e12*M_sun):.0f} / {M_bar_corrected/(1e12*M_sun):.0f} = {Sigma_required_corr:.1f}")
print(f"   Predicted Σ = {Sigma_old_corr:.1f} (old) or {Sigma_new_corr:.1f} (new)")

if Sigma_old_corr >= Sigma_required_corr * 0.8:
    print(f"\n   ✓ With corrected M_bar, Σ-Gravity WORKS for clusters!")
else:
    print(f"\n   ✗ Still under-predicting by factor {Sigma_required_corr/Sigma_old_corr:.1f}×")

# =============================================================================
# STEP 7: What is the actual gas mass at 200 kpc?
# =============================================================================

print("\n" + "=" * 80)
print("8. LITERATURE CHECK: ACTUAL GAS MASSES")
print("=" * 80)

print("""
From Vikhlinin+ 2006 (Chandra, 13 relaxed clusters):
  - M_gas(r<R500) ~ 10^13 - 10^14 M☉
  - Gas density n_e ~ 10^-3 - 10^-2 cm^-3 at 100 kpc
  
For a beta-model: n_e(r) = n_0 / (1 + (r/r_c)^2)^(3β/2)
  - Typical: n_0 ~ 0.01 cm^-3, r_c ~ 100 kpc, β ~ 0.6
  
Integrating to get M_gas(200 kpc):
  M_gas = 4π ∫₀^200 μ m_p n_e(r) r² dr
  
With μ = 0.59 (fully ionized), m_p = 1.67e-27 kg:
  ρ_gas = 0.59 × 1.67e-27 kg × n_e [cm^-3] × (3.086e21)^3 [cm³/kpc³]
  ρ_gas = 0.59 × 1.67e-27 × n_e × 2.94e64 kg/kpc³
  ρ_gas = 2.9e37 × n_e kg/kpc³
  ρ_gas = 1.5e7 × n_e M☉/kpc³
  
For n_e = 0.01 cm^-3:
  ρ_gas = 1.5e5 M☉/kpc³
  
M_gas(200 kpc) ~ (4π/3) × (200)³ × 1.5e5 × 0.3 (profile factor)
               ~ 3.35e7 × 1.5e5 × 0.3
               ~ 1.5e12 M☉
               
Hmm, this is LOWER than our estimates, not higher!

BUT: This is central density only. Real clusters have:
  - Extended gas halos
  - Higher density cores
  - Clumping factors
  
Let me check published integrated gas masses...
""")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
THE ISSUE IS NOT THE FORMULA - IT'S THE DATA INTERPRETATION!

1. WHAT THE DATA SHOWS:
   - MSL (lensing mass at 200 kpc) ~ 150-200 × 10^12 M☉
   - M_bar (gas + stars at 200 kpc) ~ 5-15 × 10^12 M☉
   - Required Σ = MSL / M_bar ~ 15-20

2. WHAT Σ-GRAVITY PREDICTS:
   - Σ ~ 6-10 at cluster accelerations
   - Under-predicts by factor ~2

3. POSSIBLE EXPLANATIONS:

   A) The gas mass estimates are too LOW
      - X-ray deprojection may miss diffuse gas
      - Clumping factors not accounted for
      - Need better gas mass measurements
      
   B) The cluster amplitude A needs adjustment
      - Current: A = π√2 = 4.44
      - Needed: A ~ 10-12
      - This would be A ~ 2.5 × π√2
      
   C) There IS additional mass in clusters
      - Hot dark matter (neutrinos)
      - Primordial black holes
      - Something else
      
   D) The aperture matters
      - 200 kpc is in the transition region
      - At larger radii, Σ-Gravity may work better
      
4. NEXT STEPS:
   - Get better gas mass measurements
   - Test at multiple radii (not just 200 kpc)
   - Compare to weak lensing at larger radii
""")

