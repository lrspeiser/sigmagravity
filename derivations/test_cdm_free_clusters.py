#!/usr/bin/env python3
"""
Test Σ-Gravity on CDM-Free Cluster Dataset
==========================================

Compares g† = cH₀/(2e) vs g† = cH₀/(4√π) on clusters with
DIRECTLY MEASURED baryonic masses (no ΛCDM assumptions).

Author: Sigma Gravity Team
Date: December 2025
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

# Cluster amplitude (3D spherical geometry)
A_cluster = np.pi * np.sqrt(2)

print("=" * 80)
print("Σ-GRAVITY TEST ON CDM-FREE CLUSTER DATA")
print("=" * 80)
print(f"\nCritical accelerations:")
print(f"  Old: g† = cH₀/(2e)    = {g_dagger_old:.4e} m/s²")
print(f"  New: g† = cH₀/(4√π)   = {g_dagger_new:.4e} m/s²")
print(f"  Ratio: {g_dagger_old / g_dagger_new:.4f}")
print(f"\nCluster amplitude: A = π√2 = {A_cluster:.4f}")


def h_universal(g, g_dag):
    """Universal acceleration function h(g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)


def Sigma_cluster(g, g_dag):
    """Enhancement factor for clusters (W=1 for lensing)."""
    return 1 + A_cluster * h_universal(g, g_dag)


# Load CDM-free cluster data
data_file = Path(__file__).parent / 'cluster_baryonic_cdm_free.csv'
df = pd.read_csv(data_file)

print(f"\nLoaded {len(df)} clusters with CDM-free baryonic masses")

# =============================================================================
# ANALYSIS: Compare predicted Σ to required Σ
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS: Σ-GRAVITY PREDICTIONS VS REQUIRED ENHANCEMENT")
print("=" * 80)

# Aperture radius
r_aperture = 200  # kpc
r_m = r_aperture * kpc_to_m

results = []

for idx, row in df.iterrows():
    name = row['name']
    z = row['z']
    M_bar = row['M_bar_200kpc_1e12Msun'] * 1e12 * M_sun  # kg
    MSL = row['MSL_200kpc_1e12Msun'] * 1e12 * M_sun  # kg
    Sigma_required = row['Sigma_required']
    
    # Compute baryonic acceleration at r = 200 kpc
    g_bar = G * M_bar / r_m**2
    
    # Compute predicted Σ with old and new g†
    Sigma_old = Sigma_cluster(g_bar, g_dagger_old)
    Sigma_new = Sigma_cluster(g_bar, g_dagger_new)
    
    # Compute predicted lensing mass
    M_pred_old = M_bar * Sigma_old
    M_pred_new = M_bar * Sigma_new
    
    # Compute ratios
    ratio_old = M_pred_old / MSL
    ratio_new = M_pred_new / MSL
    
    results.append({
        'name': name,
        'z': z,
        'M_bar': M_bar / (1e12 * M_sun),
        'MSL': MSL / (1e12 * M_sun),
        'g_bar': g_bar,
        'Sigma_required': Sigma_required,
        'Sigma_old': Sigma_old,
        'Sigma_new': Sigma_new,
        'ratio_old': ratio_old,
        'ratio_new': ratio_new,
    })

results_df = pd.DataFrame(results)

# Print results
print(f"\n{'Cluster':<22} {'Σ_req':<8} {'Σ_old':<8} {'Σ_new':<8} {'Ratio_old':<10} {'Ratio_new':<10}")
print("-" * 80)
for _, row in results_df.iterrows():
    print(f"{row['name']:<22} {row['Sigma_required']:<8.1f} {row['Sigma_old']:<8.1f} {row['Sigma_new']:<8.1f} {row['ratio_old']:<10.3f} {row['ratio_new']:<10.3f}")

# =============================================================================
# STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

# Mean ratios (ideal = 1.0)
mean_ratio_old = results_df['ratio_old'].mean()
mean_ratio_new = results_df['ratio_new'].mean()

# RMS deviation from 1.0
rms_old = np.sqrt(np.mean((results_df['ratio_old'] - 1)**2))
rms_new = np.sqrt(np.mean((results_df['ratio_new'] - 1)**2))

# Scatter (std of ratios)
scatter_old = results_df['ratio_old'].std()
scatter_new = results_df['ratio_new'].std()

# MAE
mae_old = np.mean(np.abs(results_df['ratio_old'] - 1))
mae_new = np.mean(np.abs(results_df['ratio_new'] - 1))

print(f"\nMean predicted/observed ratio (ideal = 1.0):")
print(f"  Old (2e):   {mean_ratio_old:.3f}")
print(f"  New (4√π):  {mean_ratio_new:.3f}")

print(f"\nRMS deviation from 1.0:")
print(f"  Old (2e):   {rms_old:.3f}")
print(f"  New (4√π):  {rms_new:.3f}")

print(f"\nScatter (σ):")
print(f"  Old (2e):   {scatter_old:.3f}")
print(f"  New (4√π):  {scatter_new:.3f}")

print(f"\nMean Absolute Error:")
print(f"  Old (2e):   {mae_old:.3f}")
print(f"  New (4√π):  {mae_new:.3f}")

# =============================================================================
# KEY INSIGHT: Required enhancement vs predicted
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHT: ENHANCEMENT COMPARISON")
print("=" * 80)

mean_Sigma_required = results_df['Sigma_required'].mean()
mean_Sigma_old = results_df['Sigma_old'].mean()
mean_Sigma_new = results_df['Sigma_new'].mean()

print(f"\nMean enhancement factors:")
print(f"  Required (from lensing):  Σ = {mean_Sigma_required:.1f}")
print(f"  Predicted (old, 2e):      Σ = {mean_Sigma_old:.1f}")
print(f"  Predicted (new, 4√π):     Σ = {mean_Sigma_new:.1f}")

print(f"\nBoth formulas under-predict by factor of ~{mean_Sigma_required / mean_Sigma_old:.1f}×")

# =============================================================================
# ANALYSIS: What g† would fit the data?
# =============================================================================

print("\n" + "=" * 80)
print("FITTING: WHAT g† WOULD MATCH THE DATA?")
print("=" * 80)

# For each cluster, solve for g† that gives the required Σ
# Σ = 1 + A × h(g)
# h(g) = √(g†/g) × g†/(g† + g)
# 
# This is transcendental, so we'll search numerically

def find_optimal_g_dagger(g_bar, Sigma_target, A=A_cluster):
    """Find g† that gives the target Σ for given g_bar."""
    from scipy.optimize import brentq
    
    def objective(log_g_dag):
        g_dag = 10**log_g_dag
        Sigma_pred = 1 + A * h_universal(g_bar, g_dag)
        return Sigma_pred - Sigma_target
    
    try:
        # Search over a wide range of g†
        log_g_dag_opt = brentq(objective, -15, -5)
        return 10**log_g_dag_opt
    except:
        return np.nan

# Find optimal g† for each cluster
optimal_g_daggers = []
for _, row in results_df.iterrows():
    g_bar = row['g_bar']
    Sigma_req = row['Sigma_required']
    g_dag_opt = find_optimal_g_dagger(g_bar, Sigma_req)
    optimal_g_daggers.append(g_dag_opt)

results_df['g_dagger_optimal'] = optimal_g_daggers

# Statistics on optimal g†
g_dag_opt_mean = np.nanmean(results_df['g_dagger_optimal'])
g_dag_opt_median = np.nanmedian(results_df['g_dagger_optimal'])
g_dag_opt_std = np.nanstd(results_df['g_dagger_optimal'])

print(f"\nOptimal g† per cluster:")
print(f"  Mean:   {g_dag_opt_mean:.3e} m/s²")
print(f"  Median: {g_dag_opt_median:.3e} m/s²")
print(f"  Std:    {g_dag_opt_std:.3e} m/s²")

print(f"\nComparison to theoretical values:")
print(f"  Old (2e):   {g_dagger_old:.3e} m/s²")
print(f"  New (4√π):  {g_dagger_new:.3e} m/s²")
print(f"  Data fit:   {g_dag_opt_mean:.3e} m/s²")

ratio_to_old = g_dag_opt_mean / g_dagger_old
ratio_to_new = g_dag_opt_mean / g_dagger_new

print(f"\nData-fit g† is {ratio_to_old:.1f}× the old value")
print(f"Data-fit g† is {ratio_to_new:.1f}× the new value")

# What cosmological constant would give this g†?
# g† = cH₀/X → X = cH₀/g†
X_fit = c * H0_SI / g_dag_opt_mean
print(f"\nIf g† = cH₀/X, then X = {X_fit:.2f}")
print(f"Compare to: 2e = {2*np.e:.2f}, 4√π = {4*np.sqrt(np.pi):.2f}")

# =============================================================================
# CONCLUSION
# =============================================================================

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
FINDINGS FROM CDM-FREE CLUSTER DATA:

1. Both formulas significantly UNDER-PREDICT the required enhancement
   - Required: Σ ≈ 16 (from direct M_bar and lensing)
   - Predicted: Σ ≈ 2-3 (from both old and new g†)

2. The data-fit g† is ~10× larger than theoretical predictions
   - This suggests clusters need MORE enhancement than galaxies
   
3. POSSIBLE INTERPRETATIONS:
   a) Σ-Gravity needs a scale-dependent g† (larger at cluster scales)
   b) The cluster amplitude A = π√2 is too small for clusters
   c) There IS additional mass (dark matter) in clusters
   d) Gas mass measurements are systematically underestimated

4. IMPORTANT: The OLD formula (2e) gives slightly MORE enhancement
   than the NEW formula (4√π), so it actually performs BETTER on clusters!
   
   This is because larger g† → more enhancement at low accelerations.
""")

# Save results
output_file = Path(__file__).parent / 'cdm_free_cluster_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

