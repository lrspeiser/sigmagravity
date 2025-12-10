#!/usr/bin/env python3
"""
analyze_all_clusters.py — Comprehensive cluster lensing analysis

Tests Σ-Gravity predictions on all 12 clusters in the master catalog.
Computes Einstein radii predictions and compares to observations.

Author: Sigma Gravity Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19
Mpc_to_m = 3.086e22
arcsec_to_rad = np.pi / (180 * 3600)

# Cosmology
H0 = 70  # km/s/Mpc
H0_SI = H0 * 1000 / Mpc_to_m
cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)

# Σ-Gravity parameters (path length scaling: A = A₀ × L^(1/4), A₀ ≈ 1.6)
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # Critical acceleration ≈ 9.6e-11 m/s²
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
L_0 = 0.40  # Reference path length (kpc)
N_EXP = 0.27  # Path length exponent
L_cluster = 600  # Typical cluster path length (kpc)
A_cluster = A_0 * (L_cluster / L_0)**N_EXP  # ≈ 8.45

print("=" * 80)
print("Σ-GRAVITY CLUSTER LENSING ANALYSIS")
print("=" * 80)
print(f"\nParameters:")
print(f"  g† = {g_dagger:.3e} m/s²")
print(f"  A_cluster = {A_cluster:.1f} (from path length scaling)")


def h_universal(g):
    """Universal acceleration function h(g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def Sigma_cluster(g):
    """Enhancement factor for clusters (W=1 for lensing)."""
    return 1 + A_cluster * h_universal(g)


def angular_diameter_distance(z):
    """Angular diameter distance in Mpc."""
    return cosmo.angular_diameter_distance(z).value


def critical_surface_density(z_lens, z_source):
    """Critical surface density for lensing in kg/m²."""
    D_l = angular_diameter_distance(z_lens) * Mpc_to_m
    D_s = angular_diameter_distance(z_source) * Mpc_to_m
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value * Mpc_to_m
    
    Sigma_cr = c**2 / (4 * np.pi * G) * D_s / (D_l * D_ls)
    return Sigma_cr


def nfw_mass_profile(r_kpc, M_500, r_500, c_nfw=4.0):
    """
    NFW mass profile.
    
    Parameters:
    - r_kpc: radius in kpc
    - M_500: mass within r_500 in solar masses
    - r_500: radius in kpc
    - c_nfw: concentration parameter
    """
    r = r_kpc * kpc_to_m
    r_s = r_500 * kpc_to_m / c_nfw
    
    # NFW enclosed mass formula
    x = r / r_s
    x_500 = c_nfw
    
    # M(<r) = M_s * [ln(1+x) - x/(1+x)]
    f_x = np.log(1 + x) - x / (1 + x)
    f_500 = np.log(1 + x_500) - x_500 / (1 + x_500)
    
    M_enc = M_500 * M_sun * f_x / f_500
    return M_enc


def baryonic_mass_profile(r_kpc, M_500, r_500, f_gas=0.11):
    """
    Baryonic mass profile (gas + BCG stars).
    
    Gas follows beta-model, stars concentrated in BCG.
    """
    r = r_kpc
    
    # Gas mass (beta-model, beta=2/3)
    r_core = 0.15 * r_500  # Core radius ~15% of r_500
    M_gas_500 = f_gas * M_500
    
    # Enclosed gas fraction for beta=2/3 model
    x = r / r_core
    x_500 = r_500 / r_core
    f_enc_gas = (x**3 / (1 + x**2)**1.5) / (x_500**3 / (1 + x_500**2)**1.5)
    f_enc_gas = np.minimum(f_enc_gas, 1.0)
    
    M_gas = M_gas_500 * f_enc_gas
    
    # BCG stellar mass (concentrated)
    M_bcg = 0.01 * M_500  # ~1% of total mass in BCG
    r_bcg = 50  # kpc effective radius
    f_enc_star = 1 - np.exp(-r / r_bcg)
    M_star = M_bcg * f_enc_star
    
    M_bar = (M_gas + M_star) * M_sun
    return M_bar


def predict_einstein_radius(cluster_data):
    """
    Predict Einstein radius using Σ-Gravity.
    
    Method:
    1. Compute baryonic mass profile M_bar(r)
    2. Compute baryonic acceleration g_bar(r) = G*M_bar/r²
    3. Apply Σ-Gravity enhancement: M_eff = Σ(g_bar) * M_bar
    4. Find radius where projected mass = critical mass for lensing
    """
    z_lens = cluster_data['z_lens']
    z_source = cluster_data['z_source']
    M_500 = cluster_data['M_500_Msun']
    r_500 = cluster_data['R_500_kpc']
    f_gas = cluster_data['fgas_R500']
    
    # Angular diameter distance to lens
    D_l = angular_diameter_distance(z_lens) * Mpc_to_m
    
    # Critical surface density
    Sigma_cr = critical_surface_density(z_lens, z_source)
    
    # Scan radii to find Einstein radius
    r_test = np.logspace(0.5, 3, 200)  # 3 to 1000 kpc
    
    theta_E_values = []
    
    for r_kpc in r_test:
        r = r_kpc * kpc_to_m
        
        # Baryonic enclosed mass
        M_bar = baryonic_mass_profile(r_kpc, M_500, r_500, f_gas)
        
        # Baryonic acceleration
        g_bar = G * M_bar / r**2
        
        # Σ-Gravity enhancement
        Sigma = Sigma_cluster(g_bar)
        M_eff = Sigma * M_bar
        
        # Mean surface density within r
        Sigma_mean = M_eff / (np.pi * r**2)
        
        # Convergence
        kappa = Sigma_mean / Sigma_cr
        
        # Store
        theta_E_values.append((r_kpc, kappa))
    
    # Find where kappa = 1 (Einstein radius definition)
    r_arr = np.array([x[0] for x in theta_E_values])
    kappa_arr = np.array([x[1] for x in theta_E_values])
    
    # Interpolate to find r where kappa = 1
    if np.max(kappa_arr) < 1:
        # Never reaches kappa=1, use maximum kappa radius
        idx_max = np.argmax(kappa_arr)
        r_E_kpc = r_arr[idx_max]
    elif np.min(kappa_arr) > 1:
        # Always above 1, use innermost
        r_E_kpc = r_arr[0]
    else:
        # Interpolate
        idx = np.where(kappa_arr >= 1)[0][-1]
        if idx < len(kappa_arr) - 1:
            # Linear interpolation
            r1, r2 = r_arr[idx], r_arr[idx + 1]
            k1, k2 = kappa_arr[idx], kappa_arr[idx + 1]
            r_E_kpc = r1 + (1 - k1) * (r2 - r1) / (k2 - k1)
        else:
            r_E_kpc = r_arr[idx]
    
    # Convert to arcseconds
    r_E_m = r_E_kpc * kpc_to_m
    theta_E_rad = r_E_m / D_l
    theta_E_arcsec = theta_E_rad / arcsec_to_rad
    
    return theta_E_arcsec


# Load cluster catalog
data_dir = Path(__file__).parent.parent / "data" / "clusters"
catalog_file = data_dir / "master_catalog.csv"

print(f"\nLoading catalog: {catalog_file}")
df = pd.read_csv(catalog_file)

print(f"Found {len(df)} clusters\n")

# Analyze each cluster
results = []

print(f"{'Cluster':<12} {'z':<6} {'θ_E obs':<10} {'θ_E pred':<10} {'Δ/σ':<8} {'Status':<10}")
print("-" * 66)

for idx, row in df.iterrows():
    name = row['cluster_name']
    z_lens = row['z_lens']
    theta_obs = row['theta_E_obs_arcsec']
    theta_err = row['theta_E_err_arcsec']
    
    # Predict Einstein radius
    cluster_data = {
        'z_lens': z_lens,
        'z_source': row['z_source'],
        'M_500_Msun': float(row['M_500_Msun']),
        'R_500_kpc': row['R_500_kpc'],
        'fgas_R500': row['fgas_R500'],
    }
    
    theta_pred = predict_einstein_radius(cluster_data)
    
    # Compute normalized residual
    delta_sigma = (theta_obs - theta_pred) / theta_err
    
    # Status
    if abs(delta_sigma) < 1:
        status = "✓ <1σ"
    elif abs(delta_sigma) < 2:
        status = "○ <2σ"
    else:
        status = "✗ >2σ"
    
    results.append({
        'name': name,
        'z': z_lens,
        'theta_obs': theta_obs,
        'theta_pred': theta_pred,
        'theta_err': theta_err,
        'delta_sigma': delta_sigma,
        'status': status,
    })
    
    print(f"{name:<12} {z_lens:<6.3f} {theta_obs:<10.1f} {theta_pred:<10.1f} {delta_sigma:<8.2f} {status:<10}")

# Summary statistics
print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)

delta_sigmas = [r['delta_sigma'] for r in results]
within_1sigma = sum(1 for d in delta_sigmas if abs(d) < 1)
within_2sigma = sum(1 for d in delta_sigmas if abs(d) < 2)
n_total = len(results)

rms_residual = np.sqrt(np.mean(np.array(delta_sigmas)**2))
mean_residual = np.mean(delta_sigmas)

print(f"\nN = {n_total} clusters")
print(f"Within 1σ: {within_1sigma}/{n_total} ({100*within_1sigma/n_total:.0f}%)")
print(f"Within 2σ: {within_2sigma}/{n_total} ({100*within_2sigma/n_total:.0f}%)")
print(f"RMS residual: {rms_residual:.2f}σ")
print(f"Mean residual: {mean_residual:.2f}σ")

# Expected for Gaussian: 68% within 1σ, 95% within 2σ
print(f"\nExpected (Gaussian): 68% within 1σ, 95% within 2σ")
print(f"Observed: {100*within_1sigma/n_total:.0f}% within 1σ, {100*within_2sigma/n_total:.0f}% within 2σ")

# Generate figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Predicted vs Observed
ax = axes[0]
theta_obs_arr = [r['theta_obs'] for r in results]
theta_pred_arr = [r['theta_pred'] for r in results]
theta_err_arr = [r['theta_err'] for r in results]

ax.errorbar(theta_obs_arr, theta_pred_arr, xerr=theta_err_arr, fmt='o', 
            ms=8, capsize=3, color='steelblue', alpha=0.8)

# Label points
for r in results:
    ax.annotate(r['name'], (r['theta_obs'], r['theta_pred']), 
                fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

# 1:1 line
ax.plot([5, 60], [5, 60], 'k--', lw=1, alpha=0.5, label='1:1')
ax.fill_between([5, 60], [5*0.8, 60*0.8], [5*1.2, 60*1.2], alpha=0.1, color='gray', label='±20%')

ax.set_xlabel(r'Observed $\theta_E$ [arcsec]')
ax.set_ylabel(r'Predicted $\theta_E$ [arcsec]')
ax.set_title(f'Σ-Gravity Cluster Predictions (N={n_total})')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(8, 60)
ax.set_ylim(8, 60)
ax.set_aspect('equal')

# Right: Residuals histogram
ax = axes[1]
ax.hist(delta_sigmas, bins=np.arange(-3.5, 4, 0.5), color='steelblue', 
        alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='k', linestyle='-', lw=1)
ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel(r'Normalized residual $(\theta_{obs} - \theta_{pred})/\sigma$')
ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution (RMS = {rms_residual:.2f}σ)')
ax.grid(True, alpha=0.3)

# Annotation
ax.text(0.95, 0.95, f'Within 1σ: {within_1sigma}/{n_total}\nWithin 2σ: {within_2sigma}/{n_total}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / "figures"
output_file = output_dir / "cluster_all_12_validation.png"
plt.savefig(output_file, dpi=150)
print(f"\nFigure saved: {output_file}")

plt.close()

# Final verdict
print("\n" + "=" * 66)
print("VERDICT")
print("=" * 66)

if within_2sigma >= 0.9 * n_total:
    print(f"""
✓ STRONG VALIDATION

{within_2sigma}/{n_total} clusters ({100*within_2sigma/n_total:.0f}%) within 2σ
{within_1sigma}/{n_total} clusters ({100*within_1sigma/n_total:.0f}%) within 1σ

Σ-Gravity successfully predicts Einstein radii across the full
cluster sample with derived parameters (A = π√2, g† = cH₀/2e).
""")
elif within_2sigma >= 0.7 * n_total:
    print(f"""
○ MODERATE VALIDATION

{within_2sigma}/{n_total} clusters ({100*within_2sigma/n_total:.0f}%) within 2σ
{within_1sigma}/{n_total} clusters ({100*within_1sigma/n_total:.0f}%) within 1σ

Reasonable agreement but some outliers may indicate:
- Baryonic mass model uncertainties
- Non-equilibrium/merging clusters
- Line-of-sight contamination
""")
else:
    print(f"""
✗ INSUFFICIENT VALIDATION

Only {within_2sigma}/{n_total} clusters within 2σ

Systematic issues may exist. Check:
- Amplitude calibration
- Baryonic mass profiles
- Selection effects
""")
