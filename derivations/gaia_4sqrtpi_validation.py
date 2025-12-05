#!/usr/bin/env python3
"""
Gaia DR3 Milky Way Rotation Curve Validation: New vs Old g† Formula
=====================================================================

Tests g† = cH₀/(4√π) vs g† = cH₀/(2e) on Gaia DR3 Milky Way data.

Uses the binned rotation curve data from vendor/maxdepth_gaia/gaia_bin_residuals.csv
which contains actual Gaia observations of the MW rotation curve.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Physical constants
# ============================================================================
c = 2.998e8          # m/s
H0_SI = 2.27e-18     # 1/s (70 km/s/Mpc)
kpc_to_m = 3.086e19  # m per kpc
km_to_m = 1000.0
G = 4.302e-6         # kpc (km/s)^2 / Msun

# Critical accelerations
g_dagger_old = c * H0_SI / (2 * np.e)        # Old: ≈ 1.25×10⁻¹⁰ m/s²
g_dagger_new = c * H0_SI / (4 * np.sqrt(np.pi))  # New: ≈ 9.60×10⁻¹¹ m/s²

# Model parameters
A_galaxy = np.sqrt(3)              # ≈ 1.732
R_d_MW = 2.6                       # kpc
a0_mond = 1.2e-10                  # m/s²

print("=" * 80)
print("GAIA DR3 MILKY WAY VALIDATION: g† = cH₀/(4√π) vs g† = cH₀/(2e)")
print("=" * 80)
print(f"\nCritical accelerations:")
print(f"  Old (2e):     g† = {g_dagger_old:.3e} m/s²")
print(f"  New (4√π):    g† = {g_dagger_new:.3e} m/s²")
print(f"  MOND a₀:      a₀ = {a0_mond:.3e} m/s²")
print(f"  Ratio new/old: {g_dagger_new/g_dagger_old:.3f}")
print(f"  Ratio new/a₀:  {g_dagger_new/a0_mond:.3f}")

# ============================================================================
# Milky Way Baryonic Model (McGaugh-like)
# ============================================================================
# Masses in Msun, scale lengths in kpc
M_bulge, a_bulge = 9e9, 0.5
M_thin, a_thin, b_thin = 5.5e10, 2.5, 0.3
M_thick, a_thick, b_thick = 1.0e10, 2.5, 0.9
M_HI, a_HI, b_HI = 1.0e10, 7.0, 0.1
M_H2, a_H2, b_H2 = 1.0e9, 1.5, 0.05

def v_baryon(R):
    """Baryonic rotation curve from Miyamoto-Nagai + Hernquist profiles"""
    v2 = (G*M_bulge/(R+a_bulge) + 
          G*M_thin*R**2/(np.sqrt(R**2+(a_thin+b_thin)**2))**3 +
          G*M_thick*R**2/(np.sqrt(R**2+(a_thick+b_thick)**2))**3 +
          G*M_HI*R**2/(np.sqrt(R**2+(a_HI+b_HI)**2))**3 +
          G*M_H2*R**2/(np.sqrt(R**2+(a_H2+b_H2)**2))**3)
    return np.sqrt(np.maximum(v2, 0))

def v_to_g(V_kms, R_kpc):
    """Convert velocity to acceleration: g = V²/R"""
    V_m = V_kms * km_to_m
    R_m = R_kpc * kpc_to_m
    return V_m**2 / np.maximum(R_m, 1e-10)

def g_to_v(g, R_kpc):
    """Convert acceleration to velocity: V = √(gR)"""
    R_m = R_kpc * kpc_to_m
    V_m = np.sqrt(np.maximum(g * R_m, 0))
    return V_m / km_to_m

# ============================================================================
# Σ-Gravity model functions
# ============================================================================
def h_universal(g, g_dagger):
    """Universal acceleration function: h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def C_coherence(r, R_d=R_d_MW):
    """Spatial coherence window: C(R) = 1 - (ξ/(ξ+R))^0.5"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5

def Sigma_derived(r, g_bar, g_dagger, R_d=R_d_MW):
    """Derived Σ-Gravity enhancement"""
    h = h_universal(g_bar, g_dagger)
    C = C_coherence(r, R_d)
    return 1 + A_galaxy * C * h

def v_sigma(R, g_dagger):
    """Σ-Gravity rotation curve"""
    v_b = v_baryon(R)
    g_b = v_to_g(v_b, R)
    Sigma = Sigma_derived(R, g_b, g_dagger)
    return v_b * np.sqrt(Sigma)

# ============================================================================
# MOND model
# ============================================================================
def mond_nu(g):
    """MOND simple interpolation function"""
    g = np.maximum(g, 1e-15)
    return 1 / (1 - np.exp(-np.sqrt(g / a0_mond)))

def v_mond(R):
    """MOND rotation curve"""
    v_b = v_baryon(R)
    g_b = v_to_g(v_b, R)
    nu = mond_nu(g_b)
    return v_b * np.sqrt(nu)

# ============================================================================
# Load Gaia data
# ============================================================================
gaia_csv = Path(__file__).parent.parent / "vendor" / "maxdepth_gaia" / "gaia_bin_residuals.csv"

print(f"\n{'='*80}")
print("Loading Gaia DR3 rotation curve data...")
print(f"{'='*80}")

df = pd.read_csv(gaia_csv)
print(f"Loaded {len(df)} radial bins from {gaia_csv.name}")

# Filter to bins with sufficient stars
df = df[df['N'] >= 10].copy()
print(f"After N≥10 filter: {len(df)} bins")

# Extract data
R_data = df['R_kpc_mid'].values
V_data = df['vphi_kms'].values
V_err = df['vphi_err_kms'].values

print(f"\nRadial range: {R_data.min():.1f} - {R_data.max():.1f} kpc")
print(f"Total stars: {df['N'].sum():,}")

# ============================================================================
# Calculate model predictions
# ============================================================================
print(f"\n{'='*80}")
print("Calculating model predictions...")
print(f"{'='*80}")

V_bar = v_baryon(R_data)
V_sigma_old = v_sigma(R_data, g_dagger_old)
V_sigma_new = v_sigma(R_data, g_dagger_new)
V_mond_pred = v_mond(R_data)

# ============================================================================
# Calculate RMS errors
# ============================================================================
def calc_rms(pred, obs):
    return np.sqrt(np.mean((pred - obs)**2))

def calc_weighted_rms(pred, obs, err):
    weights = 1 / err**2
    return np.sqrt(np.sum(weights * (pred - obs)**2) / np.sum(weights))

rms_bar = calc_rms(V_bar, V_data)
rms_old = calc_rms(V_sigma_old, V_data)
rms_new = calc_rms(V_sigma_new, V_data)
rms_mond = calc_rms(V_mond_pred, V_data)

wrms_bar = calc_weighted_rms(V_bar, V_data, V_err)
wrms_old = calc_weighted_rms(V_sigma_old, V_data, V_err)
wrms_new = calc_weighted_rms(V_sigma_new, V_data, V_err)
wrms_mond = calc_weighted_rms(V_mond_pred, V_data, V_err)

print(f"\n{'Model':<25} {'RMS (km/s)':<15} {'Weighted RMS':<15}")
print("-" * 55)
print(f"{'GR (baryons only)':<25} {rms_bar:>10.2f}      {wrms_bar:>10.2f}")
print(f"{'Σ-Gravity (old 2e)':<25} {rms_old:>10.2f}      {wrms_old:>10.2f}")
print(f"{'Σ-Gravity (new 4√π)':<25} {rms_new:>10.2f}      {wrms_new:>10.2f}")
print(f"{'MOND':<25} {rms_mond:>10.2f}      {wrms_mond:>10.2f}")

# ============================================================================
# Per-bin comparison
# ============================================================================
print(f"\n{'='*80}")
print("Per-bin comparison (Data vs Models)")
print(f"{'='*80}")
print(f"\n{'R (kpc)':<10} {'V_obs':<10} {'V_bar':<10} {'V_old':<10} {'V_new':<10} {'V_MOND':<10} {'N':<8}")
print("-" * 68)

for i in range(len(R_data)):
    print(f"{R_data[i]:>8.2f}  {V_data[i]:>8.1f}  {V_bar[i]:>8.1f}  "
          f"{V_sigma_old[i]:>8.1f}  {V_sigma_new[i]:>8.1f}  {V_mond_pred[i]:>8.1f}  {df.iloc[i]['N']:>6.0f}")

# ============================================================================
# Head-to-head comparison
# ============================================================================
print(f"\n{'='*80}")
print("HEAD-TO-HEAD: New (4√π) vs Old (2e)")
print(f"{'='*80}")

residuals_old = np.abs(V_sigma_old - V_data)
residuals_new = np.abs(V_sigma_new - V_data)

wins_new = np.sum(residuals_new < residuals_old)
wins_old = np.sum(residuals_old < residuals_new)
ties = np.sum(residuals_new == residuals_old)

print(f"\nBin-by-bin wins:")
print(f"  New (4√π) wins: {wins_new}/{len(R_data)} bins")
print(f"  Old (2e) wins:  {wins_old}/{len(R_data)} bins")
print(f"  Ties:           {ties}/{len(R_data)} bins")

# ============================================================================
# Radial breakdown
# ============================================================================
print(f"\n{'='*80}")
print("RADIAL BREAKDOWN")
print(f"{'='*80}")

ranges = [(3, 6, "Inner"), (6, 9, "Mid"), (9, 12, "Outer"), (12, 20, "Far outer")]

for r_min, r_max, label in ranges:
    mask = (R_data >= r_min) & (R_data < r_max)
    if np.sum(mask) == 0:
        continue
    
    rms_old_r = calc_rms(V_sigma_old[mask], V_data[mask])
    rms_new_r = calc_rms(V_sigma_new[mask], V_data[mask])
    rms_mond_r = calc_rms(V_mond_pred[mask], V_data[mask])
    
    winner = "NEW" if rms_new_r < rms_old_r else "OLD"
    diff = (rms_old_r - rms_new_r) / rms_old_r * 100
    
    print(f"\n{label} ({r_min}-{r_max} kpc): {np.sum(mask)} bins")
    print(f"  Old (2e):   {rms_old_r:.2f} km/s")
    print(f"  New (4√π):  {rms_new_r:.2f} km/s  ({winner}, {diff:+.1f}%)")
    print(f"  MOND:       {rms_mond_r:.2f} km/s")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

improvement = (rms_old - rms_new) / rms_old * 100

print(f"\nOverall RMS comparison:")
print(f"  Old (2e):   {rms_old:.2f} km/s")
print(f"  New (4√π):  {rms_new:.2f} km/s")
print(f"  Change:     {improvement:+.1f}%")

if rms_new < rms_old:
    print(f"\n✓ New formula (4√π) is BETTER by {improvement:.1f}%")
elif rms_new > rms_old:
    print(f"\n✗ New formula (4√π) is WORSE by {-improvement:.1f}%")
else:
    print(f"\n= New formula (4√π) is EQUIVALENT")

# Compare to MOND
print(f"\nComparison to MOND ({rms_mond:.2f} km/s):")
if rms_new < rms_mond:
    print(f"  ✓ New Σ-Gravity beats MOND by {(rms_mond-rms_new)/rms_mond*100:.1f}%")
else:
    print(f"  ✗ MOND beats new Σ-Gravity by {(rms_new-rms_mond)/rms_mond*100:.1f}%")

print(f"\n{'='*80}")
print("VALIDATION COMPLETE")
print(f"{'='*80}")

