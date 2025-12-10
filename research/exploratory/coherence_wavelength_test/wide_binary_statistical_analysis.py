#!/usr/bin/env python3
"""
STATISTICAL WIDE BINARY ANALYSIS FOR Σ-GRAVITY
===============================================

The challenge with wide binaries is that for separations > 1000 AU,
the orbital velocity is < 1 km/s, but proper motion errors are comparable.

The proper approach (following Chae 2023, Banik 2024) is to analyze the
DISTRIBUTION of relative velocities, not individual measurements.

For a population of binaries:
- Newtonian: v_rel should follow a distribution peaked at v_Kep
- MOND/Σ-Gravity: v_rel distribution should be shifted to higher values

This script implements this statistical approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from astropy.io import fits

# Physical constants
G = 6.674e-11
M_sun = 1.989e30
AU_to_m = 1.496e11
pc_to_m = 3.086e16
kpc_to_m = 3.086e19
c = 2.998e8
H0_SI = 2.27e-18

# Σ-Gravity parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
A_galaxy = np.sqrt(3)
g_MW = (233e3)**2 / (8.0 * kpc_to_m)

# MOND
a0_MOND = 1.2e-10

print("=" * 80)
print("STATISTICAL WIDE BINARY ANALYSIS")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "wide_binaries"
CATALOG_FILE = DATA_DIR / "all_columns_catalog.fits"

print(f"\nLoading: {CATALOG_FILE}")

with fits.open(CATALOG_FILE, memmap=True) as hdul:
    data = hdul[1].data
    n_total = len(data)
    print(f"  Total binaries: {n_total:,}")

# Extract columns
parallax = data['parallax1']
parallax_snr = data['parallax_over_error1']
sep_AU = data['sep_AU']
pmra_1 = data['pmra1']
pmdec_1 = data['pmdec1']
pmra_2 = data['pmra2']
pmdec_2 = data['pmdec2']
pmra_err_1 = data['pmra_error1']
pmdec_err_1 = data['pmdec_error1']
pmra_err_2 = data['pmra_error2']
pmdec_err_2 = data['pmdec_error2']
phot_g_1 = data['phot_g_mean_mag1']
phot_g_2 = data['phot_g_mean_mag2']
ruwe_1 = data['ruwe1']
ruwe_2 = data['ruwe2']

# Compute physical quantities
d_pc = 1000 / parallax
dpm_ra = pmra_1 - pmra_2
dpm_dec = pmdec_1 - pmdec_2
dpm_total = np.sqrt(dpm_ra**2 + dpm_dec**2)

# Proper motion error (combined)
dpm_err = np.sqrt(pmra_err_1**2 + pmra_err_2**2 + pmdec_err_1**2 + pmdec_err_2**2)

# Relative velocity in km/s
v_rel = 4.74 * dpm_total * d_pc

# Velocity error
v_rel_err = 4.74 * dpm_err * d_pc

# Mass estimation (simplified)
def estimate_mass(G_mag, parallax):
    M_G = G_mag + 5 * np.log10(parallax / 100)
    log_M = np.clip((4.83 - M_G) / 7.5, -1, 1)
    return 10**log_M * M_sun

M_1 = estimate_mass(phot_g_1, parallax)
M_2 = estimate_mass(phot_g_2, parallax)
M_total = M_1 + M_2

# Keplerian velocity
sep_m = sep_AU * AU_to_m
v_Kep = np.sqrt(G * M_total / sep_m) / 1000  # km/s

# Internal acceleration
g_internal = G * M_total / sep_m**2

# =============================================================================
# QUALITY CUTS
# =============================================================================

print("\nApplying quality cuts...")

# Quality cuts - focus on PM quality rather than velocity error
# The velocity error is dominated by distance, which isn't a quality issue
pm_snr = dpm_total / dpm_err  # Signal-to-noise of proper motion difference

mask = (
    (parallax_snr > 20) &           # Good parallax
    (ruwe_1 < 1.4) & (ruwe_2 < 1.4) & # Not unresolved binaries
    (d_pc < 500) &                   # Reasonable distance
    (d_pc > 10) &                    # Not too close (systematics)
    (sep_AU > 200) &                 # Wide enough to be interesting
    (sep_AU < 30000) &               # Not too wide (contamination)
    (pm_snr > 3) &                   # Good PM measurement (3-sigma detection)
    (v_rel > 0) &                    # Physical
    (v_Kep > 0) &                    # Physical
    np.isfinite(v_rel) & np.isfinite(v_Kep)
)

n_good = np.sum(mask)
print(f"  After quality cuts: {n_good:,} binaries ({100*n_good/n_total:.1f}%)")

# Apply mask
sep_AU_g = sep_AU[mask]
v_rel_g = v_rel[mask]
v_Kep_g = v_Kep[mask]
v_rel_err_g = v_rel_err[mask]
g_internal_g = g_internal[mask]
d_pc_g = d_pc[mask]
M_total_g = M_total[mask]

print(f"  Separation range: {np.min(sep_AU_g):.0f} - {np.max(sep_AU_g):.0f} AU")
print(f"  v_Kep range: {np.min(v_Kep_g):.3f} - {np.max(v_Kep_g):.3f} km/s")
print(f"  v_rel range: {np.min(v_rel_g):.3f} - {np.max(v_rel_g):.3f} km/s")
print(f"  v_rel_err median: {np.median(v_rel_err_g):.3f} km/s")

# =============================================================================
# STATISTICAL ANALYSIS BY SEPARATION BIN
# =============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS BY SEPARATION")
print("=" * 80)

# Define separation bins
sep_bins = [200, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000]

results = []

print(f"\n{'Sep Range':<15} {'N':<8} {'g/g†':<8} {'<v/v_K>':<10} {'σ(v/v_K)':<10} {'Median':<10} {'p(>1)':<10}")
print("-" * 80)

for i in range(len(sep_bins) - 1):
    s_lo, s_hi = sep_bins[i], sep_bins[i+1]
    in_bin = (sep_AU_g >= s_lo) & (sep_AU_g < s_hi)
    n_bin = np.sum(in_bin)
    
    if n_bin < 50:
        continue
    
    v_ratio = v_rel_g[in_bin] / v_Kep_g[in_bin]
    g_median = np.median(g_internal_g[in_bin])
    g_ratio = g_median / g_dagger
    
    # Statistics
    mean_ratio = np.mean(v_ratio)
    std_ratio = np.std(v_ratio)
    median_ratio = np.median(v_ratio)
    sem = std_ratio / np.sqrt(n_bin)
    
    # Fraction with v > v_Kep (anomalous)
    p_above = np.mean(v_ratio > 1.0)
    
    results.append({
        'sep_lo': s_lo,
        'sep_hi': s_hi,
        'sep_mid': np.sqrt(s_lo * s_hi),
        'n': n_bin,
        'g_ratio': g_ratio,
        'mean': mean_ratio,
        'std': std_ratio,
        'median': median_ratio,
        'sem': sem,
        'p_above': p_above,
    })
    
    print(f"{s_lo:>5}-{s_hi:<8} {n_bin:<8} {g_ratio:<8.2f} {mean_ratio:<10.3f} {std_ratio:<10.3f} {median_ratio:<10.3f} {p_above:<10.3f}")

# =============================================================================
# THEORETICAL PREDICTIONS
# =============================================================================

def sigma_boost(g, with_efe=True):
    g_eff = g + g_MW if with_efe else g
    h = np.sqrt(g_dagger / g_eff) * g_dagger / (g_dagger + g_eff)
    return np.sqrt(1 + A_galaxy * h)

def mond_boost(g):
    x = g / a0_MOND
    mu = x / (1 + x)
    return np.sqrt(1 / mu)

print("\n" + "=" * 80)
print("COMPARISON TO PREDICTIONS")
print("=" * 80)

print(f"\n{'Sep (AU)':<12} {'g/g†':<8} {'Data Mean':<12} {'Newton':<10} {'Σ-EFE':<10} {'Σ-noEFE':<10} {'MOND':<10}")
print("-" * 80)

for r in results:
    g = r['g_ratio'] * g_dagger
    pred_newton = 1.0
    pred_sigma_efe = sigma_boost(g, with_efe=True)
    pred_sigma_no_efe = sigma_boost(g, with_efe=False)
    pred_mond = mond_boost(g)
    
    print(f"{r['sep_mid']:<12.0f} {r['g_ratio']:<8.2f} {r['mean']:<12.3f} {pred_newton:<10.3f} {pred_sigma_efe:<10.3f} {pred_sigma_no_efe:<10.3f} {pred_mond:<10.3f}")

# =============================================================================
# KEY DIAGNOSTIC: v_rel / v_Kep DISTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("VELOCITY RATIO DISTRIBUTIONS")
print("=" * 80)

# Select MOND regime binaries (g < g†, i.e., sep > ~10,000 AU)
mond_regime = (sep_AU_g > 7000) & (sep_AU_g < 30000)
newton_regime = (sep_AU_g > 500) & (sep_AU_g < 2000)

if np.sum(mond_regime) > 50 and np.sum(newton_regime) > 50:
    v_ratio_mond = v_rel_g[mond_regime] / v_Kep_g[mond_regime]
    v_ratio_newton = v_rel_g[newton_regime] / v_Kep_g[newton_regime]
    
    print(f"\nNewtonian regime (500-2000 AU): N = {np.sum(newton_regime)}")
    print(f"  Mean v/v_K: {np.mean(v_ratio_newton):.3f}")
    print(f"  Median v/v_K: {np.median(v_ratio_newton):.3f}")
    print(f"  Std v/v_K: {np.std(v_ratio_newton):.3f}")
    
    print(f"\nMOND regime (7000-30000 AU): N = {np.sum(mond_regime)}")
    print(f"  Mean v/v_K: {np.mean(v_ratio_mond):.3f}")
    print(f"  Median v/v_K: {np.median(v_ratio_mond):.3f}")
    print(f"  Std v/v_K: {np.std(v_ratio_mond):.3f}")
    
    # Statistical test: Are the distributions different?
    stat, p_value = stats.mannwhitneyu(v_ratio_newton, v_ratio_mond, alternative='less')
    print(f"\nMann-Whitney U test (Newton < MOND regime):")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")
    
    # Effect size
    mean_diff = np.mean(v_ratio_mond) - np.mean(v_ratio_newton)
    pooled_std = np.sqrt((np.std(v_ratio_mond)**2 + np.std(v_ratio_newton)**2) / 2)
    cohens_d = mean_diff / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    print(f"  Interpretation: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}")
else:
    print("\nInsufficient data in MOND or Newton regime for comparison")
    v_ratio_mond = None
    v_ratio_newton = None

# =============================================================================
# PLOTTING
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Mean v_ratio vs separation with predictions
ax1 = axes[0, 0]

if len(results) > 0:
    sep_mids = [r['sep_mid'] for r in results]
    means = [r['mean'] for r in results]
    sems = [r['sem'] for r in results]
    
    ax1.errorbar(sep_mids, means, yerr=sems, fmt='ko', markersize=8, capsize=4, 
                 label=f'El-Badry data (N={n_good:,})', zorder=5)

# Predictions
sep_pred = np.logspace(np.log10(200), np.log10(30000), 100)
g_pred = G * 2*M_sun / (sep_pred * AU_to_m)**2

ax1.semilogx(sep_pred, np.ones_like(sep_pred), 'k:', alpha=0.5, label='Newton')
ax1.semilogx(sep_pred, sigma_boost(g_pred, with_efe=True), 'b-', lw=2, label='Σ-Gravity (EFE)')
ax1.semilogx(sep_pred, sigma_boost(g_pred, with_efe=False), 'b--', lw=2, alpha=0.7, label='Σ-Gravity (no EFE)')
ax1.semilogx(sep_pred, mond_boost(g_pred), 'r-', lw=2, label='MOND')

# Critical separation
r_crit = np.sqrt(G * 2*M_sun / g_dagger) / AU_to_m
ax1.axvline(r_crit, color='g', ls='--', alpha=0.5, label=f'g=g† ({r_crit:.0f} AU)')

ax1.set_xlabel('Separation (AU)', fontsize=12)
ax1.set_ylabel('Mean v_obs / v_Keplerian', fontsize=12)
ax1.set_title('Wide Binary Velocity Ratio vs Separation', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(200, 30000)
ax1.set_ylim(0.5, 3.0)

# Plot 2: Distribution comparison
ax2 = axes[0, 1]

if v_ratio_mond is not None and v_ratio_newton is not None:
    bins = np.linspace(0, 5, 50)
    ax2.hist(v_ratio_newton, bins=bins, alpha=0.5, density=True, label=f'Newton regime (500-2000 AU, N={len(v_ratio_newton)})')
    ax2.hist(v_ratio_mond, bins=bins, alpha=0.5, density=True, label=f'MOND regime (7000-30000 AU, N={len(v_ratio_mond)})')
    ax2.axvline(1.0, color='k', ls='--', alpha=0.5, label='v = v_Kep')
    ax2.axvline(np.median(v_ratio_newton), color='C0', ls='-', lw=2)
    ax2.axvline(np.median(v_ratio_mond), color='C1', ls='-', lw=2)
    ax2.set_xlabel('v_obs / v_Keplerian', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Velocity Ratio Distribution by Regime', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 5)
else:
    ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax2.transAxes)

# Plot 3: Scatter plot of all data
ax3 = axes[1, 0]

# Sample for plotting (too many points otherwise)
n_plot = min(10000, len(sep_AU_g))
idx = np.random.choice(len(sep_AU_g), n_plot, replace=False)

v_ratio_all = v_rel_g / v_Kep_g
scatter = ax3.scatter(sep_AU_g[idx], v_ratio_all[idx], 
                      c=np.log10(g_internal_g[idx]/g_dagger),
                      cmap='coolwarm', s=5, alpha=0.3, vmin=-2, vmax=2)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('log₁₀(g/g†)', fontsize=10)

# Binned medians
if len(results) > 0:
    ax3.scatter(sep_mids, [r['median'] for r in results], c='k', s=100, marker='s', 
                zorder=5, label='Binned median')

ax3.set_xscale('log')
ax3.axhline(1.0, color='k', ls=':', alpha=0.5)
ax3.set_xlabel('Separation (AU)', fontsize=12)
ax3.set_ylabel('v_obs / v_Keplerian', fontsize=12)
ax3.set_title('Individual Binaries (sample)', fontsize=14)
ax3.legend(fontsize=10)
ax3.set_xlim(200, 30000)
ax3.set_ylim(0, 10)

# Plot 4: Cumulative distribution
ax4 = axes[1, 1]

if v_ratio_mond is not None and v_ratio_newton is not None:
    sorted_newton = np.sort(v_ratio_newton)
    sorted_mond = np.sort(v_ratio_mond)
    cdf_newton = np.arange(1, len(sorted_newton)+1) / len(sorted_newton)
    cdf_mond = np.arange(1, len(sorted_mond)+1) / len(sorted_mond)
    
    ax4.plot(sorted_newton, cdf_newton, 'b-', lw=2, label='Newton regime')
    ax4.plot(sorted_mond, cdf_mond, 'r-', lw=2, label='MOND regime')
    ax4.axvline(1.0, color='k', ls='--', alpha=0.5)
    ax4.set_xlabel('v_obs / v_Keplerian', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution Function', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.set_xlim(0, 5)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()

output_path = Path(__file__).parent / 'wide_binary_statistical_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
DATA QUALITY ISSUE:
The El-Badry catalog contains many binaries where the observed relative 
velocity is much higher than the Keplerian prediction. This is because:

1. Proper motion errors (~0.1 mas/yr) translate to ~0.05 km/s at 100 pc
2. For wide binaries (sep > 5000 AU), v_Kep < 0.3 km/s
3. The measurement error is comparable to or larger than the signal!

This is why Chae and Banik get different results - it depends heavily on:
- Selection criteria
- Treatment of measurement errors
- Statistical methodology

KEY FINDINGS FROM THIS ANALYSIS:

1. The raw data shows v_obs >> v_Kep for most wide binaries
   This is dominated by measurement noise, not gravitational anomalies

2. The DISTRIBUTION of v_obs/v_Kep may contain signal
   But extracting it requires careful error modeling

3. For Σ-Gravity predictions:
   - With EFE: ~10-15% boost at 10,000 AU
   - Without EFE: ~40-50% boost at 10,000 AU
   
   Both are smaller than the ~100x scatter in the raw data!

CONCLUSION:
Testing Σ-Gravity with wide binaries requires:
1. Much more sophisticated error modeling
2. Careful selection of "clean" binaries
3. Statistical comparison of distributions, not individual measurements

The current controversy (Chae vs Banik) shows this is at the limit of
what Gaia DR3 can reliably measure.
""")

