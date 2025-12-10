#!/usr/bin/env python3
"""Debug LSB vs HSB test."""
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
SPARC_TABLE = DATA_DIR / 'sparc' / 'Table1_SPARC.dat'
ROTMOD_DIR = DATA_DIR / 'Rotmod_LTG'

# Load galaxies
galaxies = {}
with open(SPARC_TABLE, 'r') as f:
    for line in f:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        name = parts[0]
        try:
            galaxies[name] = {
                'Rdisk': float(parts[8]),
                'SBdisk': float(parts[9]),
            }
        except:
            continue

# Load rotation curves
rotation_curves = {}
for rc_file in ROTMOD_DIR.glob('*_rotmod.dat'):
    name = rc_file.stem.replace('_rotmod', '')
    try:
        data = []
        with open(rc_file, 'r') as f:
            for line in f:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    data.append([float(p) for p in parts[:6]])
        if len(data) > 0:
            data = np.array(data)
            rotation_curves[name] = {'R': data[:, 0], 'v_obs': data[:, 1], 'v_gas': data[:, 3], 'v_disk': data[:, 4], 'v_bul': data[:, 5]}
    except:
        pass

# Match
matched = set(galaxies.keys()) & set(rotation_curves.keys())
print(f'Matched: {len(matched)} galaxies')

# Get SB distribution
sb_values = [galaxies[n]['SBdisk'] for n in matched if galaxies[n]['SBdisk'] > 0]
print(f'SB range: {min(sb_values):.1f} - {max(sb_values):.1f} L/pc^2')
print(f'SB percentiles: 10th={np.percentile(sb_values, 10):.1f}, 25th={np.percentile(sb_values, 25):.1f}, 50th={np.percentile(sb_values, 50):.1f}, 75th={np.percentile(sb_values, 75):.1f}, 90th={np.percentile(sb_values, 90):.1f}')

# Compute K and x for all matched galaxies
all_x = []
all_K = []
all_sb = []
for name in matched:
    Rdisk = galaxies[name]['Rdisk']
    SB = galaxies[name]['SBdisk']
    if Rdisk <= 0 or SB <= 0:
        continue
    rc = rotation_curves[name]
    R = rc['R']
    v_obs = rc['v_obs']
    v_bar = np.sqrt(rc['v_gas']**2 + rc['v_disk']**2 + rc['v_bul']**2)
    v_bar = np.maximum(v_bar, 1.0)
    K = (v_obs / v_bar)**2
    x = R / Rdisk
    for i in range(len(K)):
        if 0.1 < K[i] < 50 and 0.1 < x[i] < 10:
            all_x.append(x[i])
            all_K.append(K[i])
            all_sb.append(SB)

print(f'\nTotal data points: {len(all_x)}')
all_x = np.array(all_x)
all_K = np.array(all_K)
all_sb = np.array(all_sb)

# Split by SB percentiles  
sb_25 = np.percentile(sb_values, 25)
sb_75 = np.percentile(sb_values, 75)
lsb_mask = all_sb < sb_25
hsb_mask = all_sb > sb_75

print(f'LSB points (SB < {sb_25:.1f}): {lsb_mask.sum()}')
print(f'HSB points (SB > {sb_75:.1f}): {hsb_mask.sum()}')

# Compare K at fixed x
x_bins = np.linspace(0.5, 5.0, 10)
print(f'\nK comparison at fixed x:')
print(f'x_bin        LSB K (n)          HSB K (n)          Diff')
print('-'*60)
for i in range(len(x_bins)-1):
    x_lo, x_hi = x_bins[i], x_bins[i+1]
    lsb_in_bin = (lsb_mask) & (all_x >= x_lo) & (all_x < x_hi)
    hsb_in_bin = (hsb_mask) & (all_x >= x_lo) & (all_x < x_hi)
    if lsb_in_bin.sum() > 3 and hsb_in_bin.sum() > 3:
        lsb_K = np.median(all_K[lsb_in_bin])
        hsb_K = np.median(all_K[hsb_in_bin])
        diff = lsb_K - hsb_K
        print(f'[{x_lo:.1f},{x_hi:.1f}]     {lsb_K:.2f} ({lsb_in_bin.sum()})         {hsb_K:.2f} ({hsb_in_bin.sum()})         {diff:+.2f}')
    else:
        print(f'[{x_lo:.1f},{x_hi:.1f}]     ({lsb_in_bin.sum()})              ({hsb_in_bin.sum()})             -')
