#!/usr/bin/env python3
"""
Detailed Milky Way test for 2D framework.
The full regression showed 767 km/s RMS - need to investigate.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Physical constants
c = 3e8
H0 = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

data_dir = Path(__file__).parent.parent / "data"

# Load Gaia data
gaia_file = data_dir / "gaia" / "eilers_apogee_6d_disk.csv"
df = pd.read_csv(gaia_file)

print("Gaia data summary:")
print(f"  Columns: {list(df.columns)}")
print(f"  N points: {len(df)}")
print(f"  R_gal range: {df['R_gal'].min():.2f} - {df['R_gal'].max():.2f} kpc")
print(f"  v_phi range: {df['v_phi'].min():.1f} - {df['v_phi'].max():.1f} km/s")
print(f"  v_phi mean: {df['v_phi'].mean():.1f} km/s")

# The issue: v_phi is NEGATIVE (convention)
print(f"\n  Note: v_phi is negative (rotation convention)")
print(f"  Taking absolute value for V_obs")

# Use absolute value
V_obs = np.abs(df['v_phi'].values)
R = df['R_gal'].values

print(f"\n  |v_phi| range: {V_obs.min():.1f} - {V_obs.max():.1f} km/s")
print(f"  |v_phi| mean: {V_obs.mean():.1f} km/s")

# Filter to reasonable range
mask = (R > 5) & (R < 15) & (V_obs > 150) & (V_obs < 300)
R_filt = R[mask]
V_obs_filt = V_obs[mask]

print(f"\n  After filtering (5 < R < 15, 150 < V < 300):")
print(f"  N points: {len(R_filt)}")
print(f"  R range: {R_filt.min():.2f} - {R_filt.max():.2f} kpc")
print(f"  V range: {V_obs_filt.min():.1f} - {V_obs_filt.max():.1f} km/s")

# Simple MW model
R_d_mw = 2.6  # kpc
M_disk = 5e10 * M_sun
M_bulge = 1e10 * M_sun

def V_bar_mw(R):
    R_m = R * kpc_to_m
    # Exponential disk (Freeman)
    x = R / R_d_mw
    V_disk_sq = G_const * M_disk / (R_d_mw * kpc_to_m) * x**2 * (
        1.6 * (1 - (1 + x) * np.exp(-x)) / x**2  # Approximate Freeman formula
    )
    # Bulge
    V_bulge_sq = G_const * M_bulge / R_m
    return np.sqrt(V_disk_sq + V_bulge_sq) / 1000  # km/s

V_bar = V_bar_mw(R_filt)

print(f"\n  V_bar range: {V_bar.min():.1f} - {V_bar.max():.1f} km/s")
print(f"  V_bar mean: {V_bar.mean():.1f} km/s")

# Test both frameworks
def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi, k):
    xi = max(xi, 0.01)
    return 1 - np.power(xi / (xi + np.asarray(r)), k)

def predict_velocity(R, V_bar, R_d, A, xi_coeff, k):
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    xi = xi_coeff * R_d
    W = W_coherence(R, xi, k)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)

# 1D framework
V_pred_1d = predict_velocity(R_filt, V_bar, R_d_mw, np.sqrt(np.e), 0.5, 0.5)
rms_1d = np.sqrt(np.mean((V_obs_filt - V_pred_1d)**2))

# 2D framework
V_pred_2d = predict_velocity(R_filt, V_bar, R_d_mw, np.exp(1/(2*np.pi)), 1/(2*np.pi), 1.0)
rms_2d = np.sqrt(np.mean((V_obs_filt - V_pred_2d)**2))

print(f"\n  1D Framework: RMS = {rms_1d:.2f} km/s")
print(f"  2D Framework: RMS = {rms_2d:.2f} km/s")

# Sample predictions
print(f"\n  Sample predictions at R = 8, 10, 12 kpc:")
for R_test in [8, 10, 12]:
    V_bar_test = V_bar_mw(R_test)
    V_1d = predict_velocity(R_test, V_bar_test, R_d_mw, np.sqrt(np.e), 0.5, 0.5)
    V_2d = predict_velocity(R_test, V_bar_test, R_d_mw, np.exp(1/(2*np.pi)), 1/(2*np.pi), 1.0)
    print(f"    R={R_test}: V_bar={V_bar_test:.1f}, V_1d={V_1d:.1f}, V_2d={V_2d:.1f} km/s")

# Observed at solar radius
solar_idx = np.argmin(np.abs(R_filt - 8.0))
print(f"\n  At R ≈ 8 kpc (solar): V_obs = {V_obs_filt[solar_idx]:.1f} km/s")

