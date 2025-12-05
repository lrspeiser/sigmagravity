#!/usr/bin/env python3
"""
analyze_fox2022_clusters.py — Σ-Gravity validation on Fox+ 2022 cluster sample

Tests Σ-Gravity predictions on 75 clusters from Fox+ 2022 (ApJ 928, 87).
Uses strong lensing masses at 200 kpc aperture (MSL_200kpc) and Einstein radii.

Key advantage: This dataset has independent M500 from SZ/X-ray, allowing us to
compute baryonic mass (f_gas ~ 0.12) and compare enhanced mass to lensing mass.

Author: Sigma Gravity Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM

# Physical constants
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19
Mpc_to_m = 3.086e22

# Cosmology
H0 = 70  # km/s/Mpc
H0_SI = H0 * 1000 / Mpc_to_m
cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)

# Σ-Gravity parameters (UPDATED December 2025)
# New formula: g† = cH₀/(4√π) - purely geometric derivation
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # Critical acceleration
A_cluster = np.pi * np.sqrt(2)  # Cluster amplitude (3D geometry)

print("=" * 80)
print("Σ-GRAVITY CLUSTER VALIDATION: Fox+ 2022 Sample")
print("=" * 80)
print(f"\nParameters:")
print(f"  g† = cH₀/(4√π) = {g_dagger:.3e} m/s²")
print(f"  A_cluster = π√2 = {A_cluster:.3f}")


def h_universal(g):
    """Universal acceleration function h(g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def Sigma_cluster(g):
    """Enhancement factor for clusters (W=1 for lensing)."""
    return 1 + A_cluster * h_universal(g)


# Load Fox+ 2022 data
data_dir = Path(__file__).parent.parent / "data" / "clusters"
df = pd.read_csv(data_dir / "fox2022_unique_clusters.csv")

print(f"\nLoaded {len(df)} unique clusters from Fox+ 2022")

# Filter to clusters with both M500 (for baryonic mass) and MSL_200kpc (for comparison)
df_valid = df[df['M500_1e14Msun'].notna() & df['MSL_200kpc_1e12Msun'].notna()].copy()
print(f"Clusters with both M500 and MSL_200kpc: {len(df_valid)}")

# Further filter to high-quality (spectroscopic redshifts)
df_specz = df_valid[df_valid['spec_z_constraint'] == 'yes'].copy()
print(f"With spectroscopic redshift constraints: {len(df_specz)}")

# Use high-quality clusters (spectroscopic redshifts)
df_analysis = df_specz.copy()

# Also filter out very low mass clusters where M500 errors are huge
df_analysis = df_analysis[df_analysis['M500_1e14Msun'] > 2.0].copy()
print(f"After M500 > 2×10¹⁴ M☉ cut: {len(df_analysis)}")

# =============================================================================
# ANALYSIS: Compare Σ-enhanced baryonic mass to strong lensing mass
# =============================================================================
# 
# Strategy:
# 1. M500 is total mass (mostly DM in standard picture)
# 2. Baryonic mass ~ f_gas * M500 + f_star * M500 ≈ 0.15 * M500
# 3. Compute g_bar at r=200 kpc from baryonic mass
# 4. Apply Σ-Gravity enhancement
# 5. Compare to MSL_200kpc (strong lensing mass at 200 kpc)

print("\n" + "=" * 80)
print("ANALYSIS: Baryonic mass enhancement vs strong lensing mass")
print("=" * 80)

# Baryonic fraction (gas + stars)
f_baryon = 0.15  # Typical: ~12% gas + ~3% stars within R500

results = []

print(f"\n{'Cluster':<25} {'z':<6} {'M_bar':<10} {'Σ':<6} {'M_Σ':<12} {'MSL_200':<12} {'Ratio':<8} {'Δ/σ':<8}")
print("-" * 100)

for idx, row in df_analysis.iterrows():
    cluster = row['cluster']
    z = row['z_lens']
    
    # Total mass within R500
    M500 = row['M500_1e14Msun'] * 1e14 * M_sun  # Convert to kg
    
    # Baryonic mass estimate at 200 kpc
    # Note: We use 200 kpc aperture matching MSL measurement
    # Baryonic mass is more concentrated than total mass
    # Approximate: M_bar(200kpc) ~ 0.3 * f_baryon * M500
    # (gas follows beta-model, concentrated toward center)
    M_bar_200 = 0.4 * f_baryon * M500  # kg
    
    # Baryonic acceleration at 200 kpc
    r_200kpc = 200 * kpc_to_m
    g_bar = G * M_bar_200 / r_200kpc**2
    
    # Σ-Gravity enhancement
    Sigma = Sigma_cluster(g_bar)
    M_sigma = Sigma * M_bar_200
    
    # Observed strong lensing mass at 200 kpc
    MSL_200 = row['MSL_200kpc_1e12Msun'] * 1e12 * M_sun
    MSL_err_lo = row['e_MSL_lo'] * 1e12 * M_sun
    MSL_err_hi = row['e_MSL_hi'] * 1e12 * M_sun
    MSL_err = (MSL_err_lo + MSL_err_hi) / 2  # Symmetric error
    
    # Ratio of predicted to observed
    ratio = M_sigma / MSL_200
    
    # Normalized residual
    delta_sigma = (M_sigma - MSL_200) / MSL_err if MSL_err > 0 else np.nan
    
    results.append({
        'cluster': cluster,
        'z': z,
        'M_bar_200': M_bar_200 / M_sun,
        'g_bar': g_bar,
        'Sigma': Sigma,
        'M_sigma': M_sigma / M_sun,
        'MSL_200': MSL_200 / M_sun,
        'MSL_err': MSL_err / M_sun,
        'ratio': ratio,
        'delta_sigma': delta_sigma,
        'M500': row['M500_1e14Msun'],
    })
    
    print(f"{cluster:<25} {z:<6.3f} {M_bar_200/M_sun/1e12:<10.1f} {Sigma:<6.1f} {M_sigma/M_sun/1e12:<12.1f} {MSL_200/M_sun/1e12:<12.1f} {ratio:<8.2f} {delta_sigma:<8.1f}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

n_total = len(results_df)
ratios = results_df['ratio'].values
log_ratios = np.log10(ratios)

# Basic stats
mean_ratio = np.mean(ratios)
median_ratio = np.median(ratios)
std_ratio = np.std(ratios)
scatter_dex = np.std(log_ratios)

print(f"\nN = {n_total} clusters")
print(f"\nRatio statistics (M_Σ / MSL_200):")
print(f"  Mean:   {mean_ratio:.3f}")
print(f"  Median: {median_ratio:.3f}")
print(f"  Std:    {std_ratio:.3f}")
print(f"  Scatter: {scatter_dex:.3f} dex")

# Coverage (how many within 1σ, 2σ)
delta_sigmas = results_df['delta_sigma'].dropna().values
within_1sigma = np.sum(np.abs(delta_sigmas) < 1)
within_2sigma = np.sum(np.abs(delta_sigmas) < 2)
within_3sigma = np.sum(np.abs(delta_sigmas) < 3)

print(f"\nResidual coverage:")
print(f"  Within 1σ: {within_1sigma}/{len(delta_sigmas)} ({100*within_1sigma/len(delta_sigmas):.0f}%)")
print(f"  Within 2σ: {within_2sigma}/{len(delta_sigmas)} ({100*within_2sigma/len(delta_sigmas):.0f}%)")
print(f"  Within 3σ: {within_3sigma}/{len(delta_sigmas)} ({100*within_3sigma/len(delta_sigmas):.0f}%)")

# Check if we're systematically over/under predicting
print(f"\nSystematic bias:")
mean_log_ratio = np.mean(log_ratios)
print(f"  Mean log(ratio): {mean_log_ratio:.3f} dex")
if mean_log_ratio > 0.1:
    print(f"  → Systematically OVER-predicting by factor {10**mean_log_ratio:.2f}")
elif mean_log_ratio < -0.1:
    print(f"  → Systematically UNDER-predicting by factor {10**(-mean_log_ratio):.2f}")
else:
    print(f"  → No significant systematic bias")

# =============================================================================
# CALIBRATION CHECK: What f_baryon gives best fit?
# =============================================================================

print("\n" + "=" * 80)
print("CALIBRATION: Testing different baryonic fractions")
print("=" * 80)

for f_test in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    ratios_test = []
    for idx, row in df_analysis.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14 * M_sun
        M_bar_200 = 0.4 * f_test * M500
        r_200kpc = 200 * kpc_to_m
        g_bar = G * M_bar_200 / r_200kpc**2
        Sigma = Sigma_cluster(g_bar)
        M_sigma = Sigma * M_bar_200
        MSL_200 = row['MSL_200kpc_1e12Msun'] * 1e12 * M_sun
        ratios_test.append(M_sigma / MSL_200)
    
    median_r = np.median(ratios_test)
    scatter = np.std(np.log10(ratios_test))
    print(f"  f_baryon = {f_test:.2f}: median ratio = {median_r:.3f}, scatter = {scatter:.3f} dex")

# =============================================================================
# GENERATE FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Predicted vs Observed mass
ax = axes[0]
M_sigma_arr = results_df['M_sigma'].values / 1e12
MSL_arr = results_df['MSL_200'].values / 1e12
MSL_err_arr = results_df['MSL_err'].values / 1e12

ax.errorbar(MSL_arr, M_sigma_arr, xerr=MSL_err_arr, fmt='o', ms=5, alpha=0.6, 
            color='steelblue', capsize=2, elinewidth=0.5)

# 1:1 line
ax.plot([10, 400], [10, 400], 'k--', lw=1, alpha=0.5, label='1:1')
ax.fill_between([10, 400], [10*0.5, 400*0.5], [10*2, 400*2], alpha=0.1, color='gray')

ax.set_xlabel(r'MSL(200 kpc) [$10^{12}$ M$_\odot$]')
ax.set_ylabel(r'Σ-Gravity M$_\Sigma$ [$10^{12}$ M$_\odot$]')
ax.set_title(f'Predicted vs Observed (N={n_total})')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(30, 400)
ax.set_ylim(30, 400)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Ratio vs redshift
ax = axes[1]
z_arr = results_df['z'].values
ax.scatter(z_arr, ratios, c='steelblue', alpha=0.6, s=30)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
ax.axhline(y=median_ratio, color='coral', linestyle='-', alpha=0.7, label=f'Median = {median_ratio:.2f}')
ax.fill_between([0, 1.2], [0.5, 0.5], [2, 2], alpha=0.1, color='gray')

ax.set_xlabel('Redshift z')
ax.set_ylabel(r'M$_\Sigma$ / MSL(200 kpc)')
ax.set_title('Ratio vs Redshift')
ax.set_xlim(0.1, 1.0)
ax.set_ylim(0, 2.5)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 3: Histogram of log ratios
ax = axes[2]
ax.hist(log_ratios, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='k', linestyle='--', lw=1)
ax.axvline(x=mean_log_ratio, color='coral', linestyle='-', lw=2, label=f'Mean = {mean_log_ratio:.2f} dex')

ax.set_xlabel(r'$\log_{10}$(M$_\Sigma$ / MSL)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution (scatter = {scatter_dex:.2f} dex)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent / "figures"
output_file = output_dir / "cluster_fox2022_validation.png"
plt.savefig(output_file, dpi=150)
print(f"\nFigure saved: {output_file}")

plt.close()

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if 0.7 < median_ratio < 1.4 and scatter_dex < 0.3:
    print(f"""
✓ GOOD AGREEMENT

Median ratio: {median_ratio:.2f}
Scatter: {scatter_dex:.2f} dex

Σ-Gravity successfully predicts cluster lensing masses at 200 kpc
using only baryonic mass estimates and the derived formula
(A = π√2, g† = cH₀/2e).
""")
elif 0.5 < median_ratio < 2.0:
    print(f"""
○ MODERATE AGREEMENT

Median ratio: {median_ratio:.2f}
Scatter: {scatter_dex:.2f} dex

Reasonable order-of-magnitude agreement but systematic offset indicates:
- Baryonic mass fraction may need refinement
- Mass concentration profiles need better modeling
- Some clusters may be non-equilibrium

The formula captures the right physics but calibration could improve.
""")
else:
    print(f"""
✗ POOR AGREEMENT

Median ratio: {median_ratio:.2f}
Scatter: {scatter_dex:.2f} dex

Systematic mismatch suggests:
- Baryonic mass model is inadequate
- Amplitude A_cluster may need recalibration
- Additional physics (substructure, merger state) not captured
""")

# Save results
results_df.to_csv(output_dir.parent / "data" / "clusters" / "fox2022_sigma_results.csv", index=False)
print(f"\nResults saved to data/clusters/fox2022_sigma_results.csv")
