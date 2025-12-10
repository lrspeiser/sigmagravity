#!/usr/bin/env python3
"""
Generate Counter-Rotation Validation Figure (Version 5)

CLEAN VERSION - minimal text, let caption explain
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
from astropy.table import Table
from scipy import stats
from pathlib import Path

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Paths
script_dir = Path(__file__).resolve().parent.parent
DATA_DIR = script_dir / "data"
DYNPOP_FILE = DATA_DIR / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits"
CR_FILE = DATA_DIR / "stellar_corgi" / "bevacqua2022_counter_rotating.tsv"
OUTPUT_DIR = script_dir / "figures"

print("=" * 60)
print("COUNTER-ROTATION FIGURE (v5 - CLEAN)")
print("=" * 60)

# ============================================================================
# LOAD DATA
# ============================================================================

with fits.open(DYNPOP_FILE) as hdul:
    basic = Table(hdul[1].data)
    jam_nfw = Table(hdul[4].data)

with open(CR_FILE, 'r') as f:
    lines = f.readlines()

data_start = 0
for i, line in enumerate(lines):
    if line.startswith('---'):
        data_start = i + 1
        break

header_line = None
for i, line in enumerate(lines):
    if line.startswith('MaNGAId'):
        header_line = i
        break

headers = [h.strip() for h in lines[header_line].split('|')]

cr_data = []
for line in lines[data_start:]:
    if line.strip() and not line.startswith('#'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= len(headers):
            cr_data.append(dict(zip(headers, parts)))

cr_manga_ids = [d['MaNGAId'].strip() for d in cr_data]

dynpop_idx = {str(mid).strip(): i for i, mid in enumerate(basic['mangaid'])}
matches = [dynpop_idx[cr_id] for cr_id in cr_manga_ids if cr_id in dynpop_idx]

fdm_all = np.array(jam_nfw['fdm_Re'])
log_mstar_all = np.array(basic['nsa_elpetro_mass'])

valid = (fdm_all >= 0) & (fdm_all <= 1) & (log_mstar_all > 0) & np.isfinite(fdm_all) & np.isfinite(log_mstar_all)

cr_indices = set(matches)
all_indices = set(range(len(basic)))
normal_indices = list(all_indices - cr_indices)

cr_valid = [i for i in matches if valid[i]]
normal_valid = [i for i in normal_indices if valid[i]]

fdm_cr = fdm_all[cr_valid]
fdm_normal = fdm_all[normal_valid]

# Compute statistics
mean_fdm_normal = np.mean(fdm_normal)
mean_fdm_cr = np.mean(fdm_cr)
observed_ratio = mean_fdm_cr / mean_fdm_normal

# Bootstrap error
np.random.seed(42)
n_boot = 1000
ratios = []
for _ in range(n_boot):
    cr_sample = np.random.choice(fdm_cr, len(fdm_cr), replace=True)
    normal_sample = np.random.choice(fdm_normal, len(fdm_normal), replace=True)
    ratios.append(np.mean(cr_sample) / np.mean(normal_sample))
err = np.std(ratios)

mw_stat, mw_pval = stats.mannwhitneyu(fdm_cr, fdm_normal, alternative='less')

print(f"   Observed ratio: {observed_ratio:.2f} ± {err:.2f}")
print(f"   p-value: {mw_pval:.4f}")

# ============================================================================
# CREATE FIGURE - SIMPLE AND CLEAN
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ==========================================================================
# LEFT PANEL: Bar chart - just three bars, minimal labels
# ==========================================================================
ax = axes[0]

categories = ['ΛCDM/MOND', 'Σ-Gravity', 'Observed']
x_pos = np.arange(3)
values = [1.0, 0.55, observed_ratio]  # Σ-Gravity center of prediction range
colors = ['#888888', '#4477AA', '#CC6677']

bars = ax.bar(x_pos, values, width=0.55, color=colors, alpha=0.85, 
              edgecolor='black', linewidth=1.2)

# Error bar on observed only
ax.errorbar(2, observed_ratio, yerr=err, fmt='none', color='black', 
            capsize=6, capthick=1.5, elinewidth=1.5)

# Σ-Gravity range indicator (subtle)
ax.plot([1, 1], [0.1, 1.0], color='#4477AA', linewidth=8, alpha=0.3, solid_capstyle='round')

# Value labels on bars
ax.text(0, 1.0 + 0.04, '1.00', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#555555')
ax.text(1, 0.55 + 0.04, '0.1–1.0', ha='center', va='bottom', fontsize=9, color='#4477AA')
ax.text(2, observed_ratio + err + 0.04, f'{observed_ratio:.2f}', ha='center', va='bottom', 
        fontsize=10, fontweight='bold', color='#AA4455')

# Reference line at 1.0
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, zorder=0)

ax.set_ylabel('Relative f$_{DM}$\n(Counter-rotating / Normal)')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.25)
ax.set_xlim(-0.5, 2.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.03, 0.97, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# ==========================================================================
# RIGHT PANEL: Histograms of f_DM distributions
# ==========================================================================
ax = axes[1]

bins = np.linspace(0, 0.8, 25)

ax.hist(fdm_normal, bins=bins, density=True, alpha=0.6, color='#888888', 
        label=f'Normal (N={len(fdm_normal):,})', edgecolor='white', linewidth=0.5)
ax.hist(fdm_cr, bins=bins, density=True, alpha=0.7, color='#CC6677', 
        label=f'Counter-rotating (N={len(fdm_cr)})', edgecolor='white', linewidth=0.5)

# Mean lines
ax.axvline(mean_fdm_normal, color='#555555', linestyle='-', lw=2, alpha=0.8)
ax.axvline(mean_fdm_cr, color='#AA4455', linestyle='-', lw=2, alpha=0.8)

ax.set_xlabel('Dark Matter Fraction (f$_{DM}$)')
ax.set_ylabel('Density')
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.set_xlim(0, 0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# p-value annotation (small, bottom right)
ax.text(0.97, 0.03, f'p = {mw_pval:.3f}', transform=ax.transAxes, 
        fontsize=9, ha='right', va='bottom', color='#666666')

ax.text(0.03, 0.97, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

plt.tight_layout()

# Save
outpath = OUTPUT_DIR / 'counter_rotation_effect.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"   Saved: {outpath}")
print("=" * 60)

