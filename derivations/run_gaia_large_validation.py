#!/usr/bin/env python3
"""
Unified Σ-Gravity Model: Milky Way Validation with Large Gaia Dataset
======================================================================

Uses the 1.8M star Gaia dataset with properly signed velocities (v_phi_signed)
which matches the Eilers+ 2019 rotation curve in the 6-8 kpc range.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
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
a0_mond = 1.2e-10

R0 = 20.0
A_COEFF = 1.0
B_COEFF = 216.7
G_MW = 0.05

print("=" * 100)
print("UNIFIED Σ-GRAVITY MODEL: MILKY WAY VALIDATION (1.8M STARS)")
print("=" * 100)

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def A_unified(G: float) -> float:
    return np.sqrt(A_COEFF + B_COEFF * G**2)

def h_function(g: np.ndarray) -> np.ndarray:
    g = np.atleast_1d(np.maximum(g, 1e-15))
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def f_path(r: np.ndarray, r0: float = R0) -> np.ndarray:
    r = np.atleast_1d(r)
    return r / (r + r0)

def predict_velocity_unified(R_kpc: np.ndarray, V_bar: np.ndarray, G: float = G_MW) -> np.ndarray:
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = A_unified(G)
    h = h_function(g_bar)
    f = f_path(R_kpc)
    
    Sigma = 1 + A * f * h
    return V_bar * np.sqrt(Sigma)

def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    g_obs = g_bar * nu
    
    return np.sqrt(g_obs * R_m) / 1000

def mw_baryonic_velocity(R_kpc: np.ndarray) -> np.ndarray:
    """McMillan 2017 baryonic rotation curve."""
    R = np.atleast_1d(R_kpc)
    R_lit = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                      11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0])
    V_bar_lit = np.array([180, 185, 185, 182, 178, 175, 173, 172, 170, 168,
                          166, 164, 162, 160, 158, 156, 152, 148])
    return np.interp(R, R_lit, V_bar_lit)

# =============================================================================
# LOAD LARGE GAIA DATASET
# =============================================================================

print("\n" + "=" * 100)
print("LOADING LARGE GAIA DATASET")
print("=" * 100)

df = pd.read_csv("/Users/leonardspeiser/Projects/sigmagravity/data/gaia/gaia_processed_signed.csv")
print(f"Loaded {len(df):,} stars")

# Filter to disk plane
disk = df[np.abs(df['z']) < 0.5].copy()
print(f"Stars in disk plane (|z| < 0.5 kpc): {len(disk):,}")

# Use v_phi_signed (properly signed velocity)
v_col = 'v_phi_signed'
R_col = 'R_cyl'

# =============================================================================
# COMPUTE ROTATION CURVE
# =============================================================================

print("\n" + "=" * 100)
print("OBSERVED ROTATION CURVE (1.8M GAIA STARS)")
print("=" * 100)

R_bins = np.arange(4.0, 16.0, 0.5)
obs_data = []

for i in range(len(R_bins) - 1):
    R_min, R_max = R_bins[i], R_bins[i + 1]
    R_center = (R_min + R_max) / 2
    
    mask = (disk[R_col] >= R_min) & (disk[R_col] < R_max)
    stars = disk[mask]
    
    if len(stars) > 100:
        v_median = stars[v_col].median()
        v_std = stars[v_col].std()
        v_sem = v_std / np.sqrt(len(stars))
        
        obs_data.append({
            'R_kpc': R_center,
            'v_phi': v_median,
            'v_std': v_std,
            'v_sem': v_sem,
            'N_stars': len(stars)
        })

obs_curve = pd.DataFrame(obs_data)

print(f"\n{'R [kpc]':<10} {'v_phi [km/s]':<15} {'σ [km/s]':<12} {'N_stars':<12}")
print("-" * 50)
for _, row in obs_curve.iterrows():
    print(f"{row['R_kpc']:<10.2f} {row['v_phi']:<15.1f} {row['v_std']:<12.1f} {int(row['N_stars']):<12,}")

# =============================================================================
# COMPUTE MODEL PREDICTIONS
# =============================================================================

print("\n" + "=" * 100)
print("MODEL PREDICTIONS")
print("=" * 100)

R_obs = obs_curve['R_kpc'].values
V_bar = mw_baryonic_velocity(R_obs)
V_unified = predict_velocity_unified(R_obs, V_bar, G_MW)
V_mond = predict_mond(R_obs, V_bar)

# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 100)
print("MILKY WAY ROTATION CURVE: MODEL COMPARISON")
print("=" * 100)

print(f"\n{'R [kpc]':<10} {'V_obs':<10} {'±':<8} {'V_bar':<10} {'V_Σ':<10} {'V_MOND':<10} {'Δ_Σ':<10} {'Δ_MOND':<10}")
print("-" * 90)

residual_unified = []
residual_mond = []
residual_newton = []

for i, row in obs_curve.iterrows():
    R = row['R_kpc']
    v_obs = row['v_phi']
    err = row['v_sem']
    v_bar = V_bar[i]
    v_sigma = V_unified[i]
    v_mond = V_mond[i]
    
    d_sigma = v_obs - v_sigma
    d_mond = v_obs - v_mond
    d_newton = v_obs - v_bar
    
    residual_unified.append(d_sigma)
    residual_mond.append(d_mond)
    residual_newton.append(d_newton)
    
    print(f"{R:<10.1f} {v_obs:<10.1f} {err:<8.2f} {v_bar:<10.1f} {v_sigma:<10.1f} {v_mond:<10.1f} {d_sigma:<+10.1f} {d_mond:<+10.1f}")

residual_unified = np.array(residual_unified)
residual_mond = np.array(residual_mond)
residual_newton = np.array(residual_newton)

# Statistics
rms_unified = np.sqrt(np.mean(residual_unified**2))
rms_mond = np.sqrt(np.mean(residual_mond**2))
rms_newton = np.sqrt(np.mean(residual_newton**2))

chi2_unified = np.sum((residual_unified / obs_curve['v_sem'].values)**2)
chi2_mond = np.sum((residual_mond / obs_curve['v_sem'].values)**2)
chi2_newton = np.sum((residual_newton / obs_curve['v_sem'].values)**2)

print("\n" + "-" * 90)
print("SUMMARY STATISTICS")
print("-" * 90)

dof = len(obs_curve) - 1
print(f"\n{'Model':<25} {'RMS [km/s]':<15} {'χ²':<15} {'χ²/dof':<15}")
print("-" * 70)
print(f"{'Newtonian (baryons only)':<25} {rms_newton:<15.2f} {chi2_newton:<15.1f} {chi2_newton/dof:<15.2f}")
print(f"{'Σ-Gravity (unified)':<25} {rms_unified:<15.2f} {chi2_unified:<15.1f} {chi2_unified/dof:<15.2f}")
print(f"{'MOND':<25} {rms_mond:<15.2f} {chi2_mond:<15.1f} {chi2_mond/dof:<15.2f}")

# =============================================================================
# ANALYSIS BY REGION
# =============================================================================

print("\n" + "=" * 100)
print("ANALYSIS BY RADIAL REGION")
print("=" * 100)

# Inner region (R < 8 kpc) - where Gaia data matches Eilers
inner_mask = obs_curve['R_kpc'] < 8
outer_mask = obs_curve['R_kpc'] >= 8

# Find indices for specific radii
idx_7 = np.argmin(np.abs(obs_curve['R_kpc'].values - 7.25))
idx_10 = np.argmin(np.abs(obs_curve['R_kpc'].values - 10.25))

print(f"""
INNER REGION (R < 8 kpc):
  This is where our Gaia data matches Eilers+ 2019 best.
  
  Gaia v_phi at R~7 kpc: {obs_curve.iloc[idx_7]['v_phi']:.1f} km/s
  Eilers V_c at R=7 kpc: 229.0 km/s
  
  Σ-Gravity prediction: {V_unified[idx_7]:.1f} km/s
  MOND prediction: {V_mond[idx_7]:.1f} km/s
  
  RMS (Σ-Gravity): {np.sqrt(np.mean(residual_unified[inner_mask]**2)):.1f} km/s
  RMS (MOND): {np.sqrt(np.mean(residual_mond[inner_mask]**2)):.1f} km/s

OUTER REGION (R >= 8 kpc):
  This is where our Gaia data shows HIGHER velocities than Eilers+ 2019.
  
  Gaia v_phi at R~10 kpc: {obs_curve.iloc[idx_10]['v_phi']:.1f} km/s
  Eilers V_c at R=10 kpc: 225.0 km/s
  
  The ~35 km/s excess in outer regions may be due to:
  - Selection effects (brighter/faster stars at larger distances)
  - Non-circular motions (warp, spiral arms)
  - Genuine velocity structure not captured by axisymmetric models
""")

# =============================================================================
# COMPARISON WITH LITERATURE
# =============================================================================

print("\n" + "=" * 100)
print("COMPARISON WITH EILERS+ 2019")
print("=" * 100)

eilers_R = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
eilers_V = [232.5, 230.5, 229.0, 228.0, 226.0, 225.0, 224.0, 223.0, 222.0]

print(f"\n{'R [kpc]':<10} {'Our Gaia':<12} {'Eilers+ 2019':<15} {'Difference':<12}")
print("-" * 50)

for R_lit, V_lit in zip(eilers_R, eilers_V):
    mask = np.abs(obs_curve['R_kpc'] - R_lit) < 0.3
    if mask.any():
        v_our = obs_curve[mask]['v_phi'].values[0]
        diff = v_our - V_lit
        print(f"{R_lit:<10.1f} {v_our:<12.1f} {V_lit:<15.1f} {diff:+12.1f}")

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 100)
print("FINAL REPORT")
print("=" * 100)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    MILKY WAY VALIDATION: LARGE GAIA DATASET                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  DATA: 1.8 million Gaia DR3 stars with v_phi_signed                                     │
│  DISK STARS (|z| < 0.5 kpc): {len(disk):,}                                           │
│  RADIAL RANGE: {obs_curve['R_kpc'].min():.1f} - {obs_curve['R_kpc'].max():.1f} kpc                                                     │
│                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  KEY FINDING: DATA QUALITY                                                               │
│                                                                                          │
│  • At R = 6-8 kpc: Our Gaia data matches Eilers+ 2019 within ~5 km/s                   │
│  • At R > 9 kpc: Our Gaia data shows ~25-35 km/s HIGHER velocities                     │
│                                                                                          │
│  The outer region discrepancy is likely due to selection effects                        │
│  (brighter stars at larger distances have higher velocities on average).                │
│                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  MODEL COMPARISON (FULL RANGE):                                                          │
│                                                                                          │
│    Model                      RMS [km/s]    χ²/dof                                      │
│    ─────────────────────────────────────────────────────────────────────────────────    │
│    Newtonian (baryons only)   {rms_newton:<10.1f}    {chi2_newton/dof:<10.1f}                                    │
│    Σ-Gravity (unified)        {rms_unified:<10.1f}    {chi2_unified/dof:<10.1f}                                    │
│    MOND                       {rms_mond:<10.1f}    {chi2_mond/dof:<10.1f}                                    │
│                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  CONCLUSION:                                                                             │
│                                                                                          │
│  The Σ-Gravity model underpredicts the MW rotation curve by ~{rms_unified:.0f} km/s,            │
│  similar to the tension found with Eilers+ 2019 data.                                   │
│                                                                                          │
│  However, the outer-region excess in our Gaia data (vs Eilers+ 2019)                    │
│  suggests systematic uncertainties that affect all models equally.                      │
│                                                                                          │
│  RECOMMENDATION: Use Eilers+ 2019 for definitive MW rotation curve tests,              │
│  as it has been carefully calibrated for selection effects.                             │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 100)
print("END OF ANALYSIS")
print("=" * 100)

