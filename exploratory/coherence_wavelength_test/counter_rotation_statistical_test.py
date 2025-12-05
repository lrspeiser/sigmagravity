#!/usr/bin/env python3
"""
Statistical Test of Counter-Rotation Prediction

This script cross-matches:
1. MaNGA DynPop catalog (dynamical masses, f_DM for ~10,000 galaxies)
2. Bevacqua et al. 2022 counter-rotating galaxy catalog (64 galaxies)

And tests Σ-Gravity's prediction that counter-rotating galaxies should show
REDUCED gravitational enhancement (lower f_DM) compared to normal galaxies.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import stats
import os

# Paths
DATA_DIR = "/Users/leonardspeiser/Projects/sigmagravity/data"
DYNPOP_FILE = f"{DATA_DIR}/manga_dynpop/SDSSDR17_MaNGA_JAM.fits"
CR_FILE = f"{DATA_DIR}/stellar_corgi/bevacqua2022_counter_rotating.tsv"

print("=" * 80)
print("COUNTER-ROTATING GALAXY STATISTICAL TEST")
print("Σ-Gravity Prediction: CR galaxies should have LOWER f_DM")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading data...")

# Load DynPop catalog
with fits.open(DYNPOP_FILE) as hdul:
    # HDU 1: Basic properties (mangaid, stellar mass, etc.)
    basic = Table(hdul[1].data)
    
    # HDU 4: JAM results with NFW dark matter (beta_z anisotropy)
    # This has fdm_Re (dark matter fraction within Re)
    jam_nfw = Table(hdul[4].data)
    
    print(f"   DynPop: {len(basic)} galaxies")
    print(f"   Columns with f_DM: fdm_Re in HDU 4")

# Load counter-rotating catalog
with open(CR_FILE, 'r') as f:
    lines = f.readlines()

# Find data start
data_start = 0
for i, line in enumerate(lines):
    if line.startswith('---'):
        data_start = i + 1
        break

# Parse header
header_line = None
for i, line in enumerate(lines):
    if line.startswith('MaNGAId'):
        header_line = i
        break

headers = [h.strip() for h in lines[header_line].split('|')]

# Parse data
cr_data = []
for line in lines[data_start:]:
    if line.strip() and not line.startswith('#'):
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= len(headers):
            cr_data.append(dict(zip(headers, parts)))

cr_manga_ids = [d['MaNGAId'].strip() for d in cr_data]
print(f"   Counter-rotating: {len(cr_manga_ids)} galaxies")

# ============================================================================
# CROSS-MATCH
# ============================================================================

print("\n2. Cross-matching catalogs...")

# Create lookup dict for DynPop
dynpop_idx = {str(mid).strip(): i for i, mid in enumerate(basic['mangaid'])}

# Find matches
matches = []
for cr_id in cr_manga_ids:
    if cr_id in dynpop_idx:
        matches.append(dynpop_idx[cr_id])

print(f"   Found {len(matches)} matches out of {len(cr_manga_ids)} CR galaxies")

if len(matches) == 0:
    print("\n   DEBUG: Sample IDs")
    print(f"   CR IDs: {cr_manga_ids[:5]}")
    print(f"   DynPop IDs: {list(basic['mangaid'][:5])}")
    
    # Try with different formatting
    dynpop_ids_clean = [str(mid).strip() for mid in basic['mangaid']]
    for cr_id in cr_manga_ids[:5]:
        found = cr_id in dynpop_ids_clean
        print(f"   '{cr_id}' in DynPop: {found}")

# ============================================================================
# EXTRACT f_DM DATA
# ============================================================================

print("\n3. Extracting dark matter fractions...")

# Get f_DM for all galaxies
fdm_all = np.array(jam_nfw['fdm_Re'])
log_mstar_all = np.array(basic['nsa_elpetro_mass'])

# Filter valid data (f_DM between 0 and 1, valid stellar mass)
valid = (fdm_all >= 0) & (fdm_all <= 1) & (log_mstar_all > 0) & np.isfinite(fdm_all) & np.isfinite(log_mstar_all)
print(f"   Valid f_DM measurements: {np.sum(valid)} out of {len(fdm_all)}")

# Get indices for CR and normal galaxies
cr_indices = set(matches)
all_indices = set(range(len(basic)))
normal_indices = list(all_indices - cr_indices)

# Filter by valid data
cr_valid = [i for i in matches if valid[i]]
normal_valid = [i for i in normal_indices if valid[i]]

print(f"   CR galaxies with valid f_DM: {len(cr_valid)}")
print(f"   Normal galaxies with valid f_DM: {len(normal_valid)}")

# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

if len(cr_valid) > 5:
    print("\n4. Statistical comparison...")
    
    fdm_cr = fdm_all[cr_valid]
    fdm_normal = fdm_all[normal_valid]
    
    mstar_cr = log_mstar_all[cr_valid]
    mstar_normal = log_mstar_all[normal_valid]
    
    print(f"\n   Counter-rotating galaxies (N={len(fdm_cr)}):")
    print(f"     f_DM mean:   {np.mean(fdm_cr):.3f}")
    print(f"     f_DM median: {np.median(fdm_cr):.3f}")
    print(f"     f_DM std:    {np.std(fdm_cr):.3f}")
    print(f"     log M* mean: {np.mean(mstar_cr):.2f}")
    
    print(f"\n   Normal galaxies (N={len(fdm_normal)}):")
    print(f"     f_DM mean:   {np.mean(fdm_normal):.3f}")
    print(f"     f_DM median: {np.median(fdm_normal):.3f}")
    print(f"     f_DM std:    {np.std(fdm_normal):.3f}")
    print(f"     log M* mean: {np.mean(mstar_normal):.2f}")
    
    # Statistical tests
    print("\n   Statistical tests:")
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(fdm_cr, fdm_normal)
    print(f"     KS test: statistic={ks_stat:.3f}, p-value={ks_pval:.4f}")
    
    # Mann-Whitney U test
    mw_stat, mw_pval = stats.mannwhitneyu(fdm_cr, fdm_normal, alternative='less')
    print(f"     Mann-Whitney U (CR < Normal): statistic={mw_stat:.0f}, p-value={mw_pval:.4f}")
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(fdm_cr, fdm_normal)
    print(f"     T-test: statistic={t_stat:.3f}, p-value={t_pval:.4f}")
    
    # ============================================================================
    # MASS-MATCHED COMPARISON
    # ============================================================================
    
    print("\n5. Mass-matched comparison...")
    
    # Match by stellar mass (within 0.2 dex)
    matched_normal_fdm = []
    for i, cr_idx in enumerate(cr_valid):
        cr_mass = log_mstar_all[cr_idx]
        
        # Find normal galaxies with similar mass
        mass_matched = [j for j in normal_valid 
                        if abs(log_mstar_all[j] - cr_mass) < 0.2]
        
        if len(mass_matched) > 0:
            # Take median f_DM of matched sample
            matched_fdm = np.median(fdm_all[mass_matched])
            matched_normal_fdm.append(matched_fdm)
    
    if len(matched_normal_fdm) > 5:
        matched_normal_fdm = np.array(matched_normal_fdm)
        
        print(f"   Mass-matched comparison (N={len(matched_normal_fdm)}):")
        print(f"     CR f_DM mean:           {np.mean(fdm_cr[:len(matched_normal_fdm)]):.3f}")
        print(f"     Matched normal f_DM:    {np.mean(matched_normal_fdm):.3f}")
        print(f"     Difference:             {np.mean(fdm_cr[:len(matched_normal_fdm)]) - np.mean(matched_normal_fdm):.3f}")
        
        # Paired test
        paired_t, paired_p = stats.ttest_rel(fdm_cr[:len(matched_normal_fdm)], matched_normal_fdm)
        print(f"     Paired t-test: t={paired_t:.3f}, p={paired_p:.4f}")
    
    # ============================================================================
    # INTERPRETATION
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    diff = np.mean(fdm_cr) - np.mean(fdm_normal)
    
    print(f"""
Σ-Gravity Prediction:
  Counter-rotating galaxies should have LOWER f_DM due to disrupted coherence.
  
Observed:
  f_DM(CR) - f_DM(Normal) = {diff:.3f}
  
Result:
""")
    
    if diff < -0.05 and mw_pval < 0.05:
        print("  ✓ SUPPORTS Σ-GRAVITY: CR galaxies have significantly lower f_DM")
        status = "SUPPORTS"
    elif diff > 0.05 and mw_pval < 0.05:
        print("  ✗ CONTRADICTS Σ-GRAVITY: CR galaxies have higher f_DM")
        status = "CONTRADICTS"
    else:
        print("  ~ INCONCLUSIVE: No significant difference detected")
        status = "INCONCLUSIVE"
    
    print(f"""
Note: ΛCDM and MOND predict NO difference in f_DM for counter-rotating galaxies.
If f_DM(CR) < f_DM(Normal), this uniquely supports Σ-Gravity's coherence mechanism.
""")

else:
    print("\n   ERROR: Insufficient matched data for statistical analysis")
    print("   Need to investigate ID matching issue")
    status = "ERROR"

print("=" * 80)
print(f"FINAL STATUS: {status}")
print("=" * 80)
