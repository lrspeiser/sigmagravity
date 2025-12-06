#!/usr/bin/env python3
"""
Complete Galaxy-by-Galaxy Breakdown: Σ-Gravity vs MOND
=======================================================

Outputs detailed results for every galaxy to enable analysis of
what makes certain galaxies fit well or poorly.

Uses the latest Σ-Gravity formulas from README.md:
- g† = cH₀/(4√π) ≈ 9.60 × 10⁻¹¹ m/s²
- h(gN) = √(g†/gN) × g†/(g† + gN)
- W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)Rd
- A = √3 for disk galaxies
- Σ = 1 + A × W(r) × h(gN)
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS (from README.md)
# =============================================================================

c = 2.998e8  # m/s
H0_SI = 2.27e-18  # s^-1 (H0 ≈ 70 km/s/Mpc)
kpc_to_m = 3.086e19

# Critical acceleration: g† = cH₀/(4√π) ≈ 9.60 × 10⁻¹¹ m/s²
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))

# MOND acceleration for comparison
a0_mond = 1.2e-10

# Amplitude A = √3 for disk galaxies (3 torsion modes)
A_galaxy = math.sqrt(3)

# =============================================================================
# MODEL FUNCTIONS (from README.md §2.8, §2.7, §2.10)
# =============================================================================

def h_function(g_N):
    """
    Acceleration dependence function (README.md §2.8):
    h(gN) = √(g†/gN) × g†/(g† + gN)
    """
    g_N = np.atleast_1d(np.maximum(g_N, 1e-15))
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)


def coherence_window(r, R_d):
    """
    Coherence window function (README.md §2.7):
    W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)Rd
    """
    xi = (2.0 / 3.0) * R_d
    return 1.0 - np.sqrt(xi / (xi + r))


def predict_sigma_gravity(R, V_bar, R_d):
    """
    Σ-Gravity prediction (README.md §2.10):
    Σ = 1 + A × W(r) × h(gN)
    V_obs = V_bar × √Σ
    """
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    
    W = coherence_window(R, R_d)
    h = h_function(g_bar)
    
    Sigma = 1.0 + A_galaxy * W * h
    return V_bar * np.sqrt(Sigma)


def predict_mond(R, V_bar):
    """
    MOND prediction using simple interpolation function.
    """
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    x = g_bar / a0_mond
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return np.sqrt(g_bar * nu * R_m) / 1000


# =============================================================================
# DATA LOADING
# =============================================================================

def find_sparc_data():
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def find_master_sheet():
    """Find SPARC master sheet with disk scale lengths."""
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG/MasterSheet_SPARC.mrt"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG/MasterSheet_SPARC.mrt"),
        Path("/Users/leonardspeiser/Projects/GravityCalculator/data/Rotmod_LTG/MasterSheet_SPARC.mrt"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def load_master_sheet_data():
    """Load all data from SPARC master sheet.
    
    Returns dict with galaxy properties including:
    - Hubble type, distance, inclination
    - Luminosity, effective radius
    - Disk scale length (Rd)
    - HI mass, HI radius
    - Vflat, quality flag
    """
    master_file = find_master_sheet()
    galaxy_data = {}
    
    if master_file is None:
        return galaxy_data
    
    # Hubble type mapping
    hubble_types = {
        0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
        6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'
    }
    
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    # Find the last '---' line which marks the start of data
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            data_start = i + 1
    
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        
        # Split by whitespace - much more robust than fixed-width parsing
        parts = line.split()
        
        # Need at least 15 columns for a valid data row
        if len(parts) < 15:
            continue
        
        # Skip lines that look like notes/references
        if parts[0].startswith('Note') or '=' in parts[0]:
            continue
        
        try:
            # Column mapping based on split:
            # 0: Galaxy name
            # 1: Hubble type (0-11)
            # 2: Distance (Mpc)
            # 3: e_D
            # 4: f_D (distance method)
            # 5: Inc
            # 6: e_Inc
            # 7: L[3.6] (10^9 Lsun)
            # 8: e_L[3.6]
            # 9: Reff
            # 10: SBeff
            # 11: Rdisk
            # 12: SBdisk
            # 13: MHI (10^9 Msun)
            # 14: RHI
            # 15: Vflat (if present)
            # 16: e_Vflat (if present)
            # 17: Q (if present)
            
            name = parts[0]
            hubble_num = int(parts[1]) if parts[1].isdigit() else -1
            distance = float(parts[2])
            inc = float(parts[5])
            luminosity = float(parts[7])
            r_eff = float(parts[9])
            r_disk = float(parts[11])
            m_hi = float(parts[13])
            r_hi = float(parts[14])
            
            # Vflat might be 0.0 if not measured
            v_flat = float(parts[15]) if len(parts) > 15 else 0
            quality = int(parts[17]) if len(parts) > 17 and parts[17].isdigit() else 0
            
            galaxy_data[name] = {
                'hubble_type': hubble_types.get(hubble_num, 'Unknown'),
                'hubble_num': hubble_num,
                'distance': distance,
                'inclination': inc,
                'luminosity': luminosity,  # 10^9 Lsun
                'R_eff': r_eff,
                'R_d': r_disk,
                'M_HI': m_hi,  # 10^9 Msun
                'R_HI': r_hi,
                'V_flat_catalog': v_flat,
                'quality': quality
            }
        except (ValueError, IndexError) as e:
            continue
    
    return galaxy_data


def load_galaxy(rotmod_file, master_data):
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    
    name = rotmod_file.stem.replace('_rotmod', '')
    
    with open(rotmod_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
                except:
                    continue
    
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    V_bar_sq = np.sign(V_gas) * V_gas**2 + np.sign(V_disk) * V_disk**2 + V_bulge**2
    if np.any(V_bar_sq < 0):
        return None
    V_bar = np.sqrt(V_bar_sq)
    
    # Get properties from master sheet
    props = master_data.get(name, {})
    
    # Get disk scale length from master sheet, or estimate
    if name in master_data and master_data[name]['R_d'] > 0:
        R_d = master_data[name]['R_d']
        R_d_source = 'catalog'
    else:
        R_d = np.max(R) / 4.0
        R_d_source = 'estimated'
    
    # Derived quantities
    n_outer = max(3, len(R) // 3)
    V_flat = np.mean(np.sort(V_obs)[-n_outer:])
    V_max = np.max(V_obs)
    R_max = np.max(R)
    R_half = R[len(R)//2]
    
    # Gas fraction
    V_gas_max = np.max(np.abs(V_gas))
    V_disk_max = np.max(np.abs(V_disk))
    gas_fraction = V_gas_max**2 / (V_gas_max**2 + V_disk_max**2 + 0.01)
    
    # Bulge fraction
    V_bulge_max = np.max(V_bulge)
    bulge_fraction = V_bulge_max**2 / (V_max**2 + 0.01)
    
    # Acceleration analysis
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    g_obs = (V_obs * 1000)**2 / R_m
    g_bar_mean = np.mean(g_bar)
    g_bar_min = np.min(g_bar)
    g_bar_max = np.max(g_bar)
    
    # How "deep MOND" - fraction of points below g†
    frac_deep_mond = np.mean(g_bar < g_dagger)
    
    # Rising vs flat rotation curve
    V_inner = np.mean(V_obs[:max(3, len(V_obs)//4)])
    V_outer = np.mean(V_obs[-max(3, len(V_obs)//4):])
    rise_ratio = V_outer / (V_inner + 1)
    
    return {
        'name': name,
        'R': R, 'V_obs': V_obs, 'V_err': V_err, 'V_bar': V_bar,
        'V_gas': V_gas, 'V_disk': V_disk, 'V_bulge': V_bulge,
        'V_flat': V_flat, 'V_max': V_max, 'R_max': R_max,
        'R_d': R_d, 'R_d_source': R_d_source,
        'gas_fraction': gas_fraction, 'bulge_fraction': bulge_fraction,
        'g_bar_mean': g_bar_mean, 'g_bar_min': g_bar_min, 'g_bar_max': g_bar_max,
        'frac_deep_mond': frac_deep_mond, 'rise_ratio': rise_ratio,
        'n_points': len(R),
        # From master sheet
        'hubble_type': props.get('hubble_type', 'Unknown'),
        'hubble_num': props.get('hubble_num', -1),
        'distance': props.get('distance', 0),
        'inclination': props.get('inclination', 0),
        'luminosity': props.get('luminosity', 0),
        'M_HI': props.get('M_HI', 0),
        'R_HI': props.get('R_HI', 0),
        'quality': props.get('quality', 0),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print("=" * 100)
print("COMPLETE GALAXY-BY-GALAXY BREAKDOWN: Σ-GRAVITY vs MOND")
print("=" * 100)
print(f"\nUsing Σ-Gravity formulas from README.md:")
print(f"  g† = cH₀/(4√π) = {g_dagger:.2e} m/s²")
print(f"  A = √3 = {A_galaxy:.3f}")
print(f"  h(gN) = √(g†/gN) × g†/(g† + gN)")
print(f"  W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)Rd")
print(f"  Σ = 1 + A × W(r) × h(gN)")

sparc_dir = find_sparc_data()
if sparc_dir is None:
    print("ERROR: SPARC data not found!")
    exit(1)

# Load master sheet data
master_data = load_master_sheet_data()
print(f"\nLoaded master sheet data for {len(master_data)} galaxies")

# Analyze all galaxies
results = []

for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
    data = load_galaxy(rotmod_file, master_data)
    if data is None:
        continue
    
    try:
        # Predictions
        V_sigma = predict_sigma_gravity(data['R'], data['V_bar'], data['R_d'])
        V_mond = predict_mond(data['R'], data['V_bar'])
        
        # RMS errors
        rms_sigma = np.sqrt(np.mean((data['V_obs'] - V_sigma)**2))
        rms_mond = np.sqrt(np.mean((data['V_obs'] - V_mond)**2))
        
        # Chi-squared (if errors available)
        if np.any(data['V_err'] > 0):
            chi2_sigma = np.sum(((data['V_obs'] - V_sigma) / np.maximum(data['V_err'], 1))**2) / max(len(data['V_obs']) - 1, 1)
            chi2_mond = np.sum(((data['V_obs'] - V_mond) / np.maximum(data['V_err'], 1))**2) / max(len(data['V_obs']) - 1, 1)
        else:
            chi2_sigma = chi2_mond = 0
        
        # Normalized RMS
        nrms_sigma = rms_sigma / data['V_flat']
        nrms_mond = rms_mond / data['V_flat']
        
        # Which model wins
        sigma_wins = rms_sigma < rms_mond
        
        # Improvement ratio
        improvement = (rms_mond - rms_sigma) / rms_mond if rms_mond > 0 else 0
        
        # Categorize acceleration regime
        if data['frac_deep_mond'] > 0.7:
            accel_regime = 'Deep MOND'
        elif data['frac_deep_mond'] < 0.3:
            accel_regime = 'Newtonian'
        else:
            accel_regime = 'Transition'
        
        # Categorize by V_flat
        if data['V_flat'] < 80:
            mass_type = 'Dwarf'
        elif data['V_flat'] < 150:
            mass_type = 'Normal'
        else:
            mass_type = 'Massive'
        
        results.append({
            'name': data['name'],
            'rms_sigma': rms_sigma,
            'rms_mond': rms_mond,
            'chi2_sigma': chi2_sigma,
            'chi2_mond': chi2_mond,
            'nrms_sigma': nrms_sigma,
            'nrms_mond': nrms_mond,
            'sigma_wins': sigma_wins,
            'improvement': improvement,
            'V_flat': data['V_flat'],
            'V_max': data['V_max'],
            'R_max': data['R_max'],
            'R_d': data['R_d'],
            'R_d_source': data['R_d_source'],
            'gas_fraction': data['gas_fraction'],
            'bulge_fraction': data['bulge_fraction'],
            'g_bar_mean': data['g_bar_mean'],
            'g_bar_min': data['g_bar_min'],
            'g_bar_max': data['g_bar_max'],
            'frac_deep_mond': data['frac_deep_mond'],
            'rise_ratio': data['rise_ratio'],
            'n_points': data['n_points'],
            'hubble_type': data['hubble_type'],
            'hubble_num': data['hubble_num'],
            'distance': data['distance'],
            'inclination': data['inclination'],
            'luminosity': data['luminosity'],
            'M_HI': data['M_HI'],
            'R_HI': data['R_HI'],
            'quality': data['quality'],
            'accel_regime': accel_regime,
            'mass_type': mass_type,
        })
    except Exception as e:
        print(f"Error processing {data['name']}: {e}")
        continue

print(f"\nAnalyzed {len(results)} galaxies")

# Sort by improvement (best Σ-Gravity performance first)
results_sorted = sorted(results, key=lambda x: x['improvement'], reverse=True)

# =============================================================================
# COMPLETE TABLE: ALL GALAXIES
# =============================================================================

print("\n" + "=" * 150)
print("COMPLETE GALAXY TABLE (sorted by Σ-Gravity improvement over MOND)")
print("=" * 150)
print(f"{'Galaxy':<12} {'Type':<6} {'Winner':<6} {'RMS_Σ':>8} {'RMS_M':>8} {'Δ%':>7} "
      f"{'V_flat':>7} {'R_d':>6} {'Rd_src':>7} {'Gas%':>5} {'Deep%':>5} "
      f"{'Regime':<10} {'Mass':<8} {'Inc':>4} {'Q':>2} {'Hubble':<5}")
print("-" * 150)

for r in results_sorted:
    winner = "Σ" if r['sigma_wins'] else "M"
    print(f"{r['name']:<12} {r['hubble_type']:<6} {winner:<6} "
          f"{r['rms_sigma']:>8.2f} {r['rms_mond']:>8.2f} {r['improvement']*100:>+6.1f}% "
          f"{r['V_flat']:>7.1f} {r['R_d']:>6.2f} {r['R_d_source']:<7} "
          f"{r['gas_fraction']*100:>5.0f} {r['frac_deep_mond']*100:>5.0f} "
          f"{r['accel_regime']:<10} {r['mass_type']:<8} {r['inclination']:>4.0f} {r['quality']:>2} {r['hubble_type']:<5}")

# =============================================================================
# GALAXIES WHERE MOND WINS (Deep Analysis)
# =============================================================================

mond_wins = [r for r in results if not r['sigma_wins']]
mond_wins_sorted = sorted(mond_wins, key=lambda x: x['improvement'])

print("\n\n" + "=" * 150)
print(f"GALAXIES WHERE MOND WINS ({len(mond_wins)} galaxies) - Sorted by MOND advantage")
print("=" * 150)
print(f"{'Galaxy':<12} {'Type':<6} {'RMS_Σ':>8} {'RMS_M':>8} {'MOND Δ%':>8} "
      f"{'V_flat':>7} {'R_d':>6} {'Rd_src':>7} {'Gas%':>5} {'Deep%':>5} "
      f"{'g_min':>10} {'g_max':>10} {'Regime':<10} {'Rise':>5}")
print("-" * 150)

for r in mond_wins_sorted:
    mond_advantage = -r['improvement'] * 100
    print(f"{r['name']:<12} {r['hubble_type']:<6} "
          f"{r['rms_sigma']:>8.2f} {r['rms_mond']:>8.2f} {mond_advantage:>+7.1f}% "
          f"{r['V_flat']:>7.1f} {r['R_d']:>6.2f} {r['R_d_source']:<7} "
          f"{r['gas_fraction']*100:>5.0f} {r['frac_deep_mond']*100:>5.0f} "
          f"{r['g_bar_min']:>10.2e} {r['g_bar_max']:>10.2e} "
          f"{r['accel_regime']:<10} {r['rise_ratio']:>5.2f}")

# =============================================================================
# DEEP MOND GALAXIES WHERE MOND WINS
# =============================================================================

deep_mond_mond_wins = [r for r in mond_wins if r['accel_regime'] == 'Deep MOND']
deep_mond_mond_wins_sorted = sorted(deep_mond_mond_wins, key=lambda x: x['improvement'])

print("\n\n" + "=" * 150)
print(f"DEEP MOND REGIME GALAXIES WHERE MOND WINS ({len(deep_mond_mond_wins)} galaxies)")
print("=" * 150)
print(f"{'Galaxy':<12} {'Type':<6} {'RMS_Σ':>8} {'RMS_M':>8} {'MOND Δ%':>8} "
      f"{'V_flat':>7} {'R_d':>6} {'Gas%':>5} {'Bulge%':>6} "
      f"{'g_min/g†':>8} {'R_max':>6} {'Rise':>5} {'Inc':>4} {'L_3.6':>8}")
print("-" * 150)

for r in deep_mond_mond_wins_sorted:
    mond_advantage = -r['improvement'] * 100
    g_min_ratio = r['g_bar_min'] / g_dagger
    print(f"{r['name']:<12} {r['hubble_type']:<6} "
          f"{r['rms_sigma']:>8.2f} {r['rms_mond']:>8.2f} {mond_advantage:>+7.1f}% "
          f"{r['V_flat']:>7.1f} {r['R_d']:>6.2f} "
          f"{r['gas_fraction']*100:>5.0f} {r['bulge_fraction']*100:>6.1f} "
          f"{g_min_ratio:>8.2f} {r['R_max']:>6.1f} {r['rise_ratio']:>5.2f} "
          f"{r['inclination']:>4.0f} {r['luminosity']:>8.3f}")

# =============================================================================
# ANALYSIS BY HUBBLE TYPE
# =============================================================================

print("\n\n" + "=" * 100)
print("BREAKDOWN BY HUBBLE TYPE")
print("=" * 100)

hubble_groups = {}
for r in results:
    ht = r['hubble_type']
    if ht not in hubble_groups:
        hubble_groups[ht] = []
    hubble_groups[ht].append(r)

# Sort by Hubble sequence
hubble_order = ['S0', 'Sa', 'Sab', 'Sb', 'Sbc', 'Sc', 'Scd', 'Sd', 'Sdm', 'Sm', 'Im', 'BCD', 'Unknown']
for ht in hubble_order:
    if ht not in hubble_groups:
        continue
    galaxies = hubble_groups[ht]
    n = len(galaxies)
    wins = sum(1 for g in galaxies if g['sigma_wins'])
    mean_rms_s = np.mean([g['rms_sigma'] for g in galaxies])
    mean_rms_m = np.mean([g['rms_mond'] for g in galaxies])
    mean_imp = np.mean([g['improvement'] for g in galaxies])
    mean_gas = np.mean([g['gas_fraction'] for g in galaxies])
    mean_deep = np.mean([g['frac_deep_mond'] for g in galaxies])
    
    print(f"\n{ht} (N={n}):")
    print(f"  Σ-Gravity wins: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Mean RMS: Σ={mean_rms_s:.1f}, MOND={mean_rms_m:.1f} km/s")
    print(f"  Mean improvement: {mean_imp*100:+.1f}%")
    print(f"  Mean gas fraction: {mean_gas*100:.0f}%")
    print(f"  Mean deep MOND fraction: {mean_deep*100:.0f}%")

# =============================================================================
# PATTERN ANALYSIS: What's special about MOND winners?
# =============================================================================

sigma_wins_list = [r for r in results if r['sigma_wins']]

print("\n\n" + "=" * 100)
print("PATTERN ANALYSIS: MOND WINNERS vs Σ-GRAVITY WINNERS")
print("=" * 100)

def compare_groups(group1, group2, name1, name2):
    metrics = [
        ('V_flat', 'km/s'),
        ('R_d', 'kpc'),
        ('gas_fraction', '%'),
        ('bulge_fraction', '%'),
        ('frac_deep_mond', '%'),
        ('rise_ratio', ''),
        ('inclination', 'deg'),
        ('R_max', 'kpc'),
        ('luminosity', '10^9 Lsun'),
        ('M_HI', '10^9 Msun'),
    ]
    
    print(f"\n{'Metric':<20} {name1:>15} {name2:>15} {'Difference':>15}")
    print("-" * 70)
    
    for metric, unit in metrics:
        vals1 = [g[metric] for g in group1 if g[metric] is not None and not np.isnan(g[metric])]
        vals2 = [g[metric] for g in group2 if g[metric] is not None and not np.isnan(g[metric])]
        
        if len(vals1) > 0 and len(vals2) > 0:
            mean1 = np.mean(vals1)
            mean2 = np.mean(vals2)
            
            if 'fraction' in metric:
                mean1 *= 100
                mean2 *= 100
            
            diff = mean2 - mean1
            print(f"{metric:<20} {mean1:>12.2f} {mean2:>12.2f} {diff:>+12.2f} {unit}")

compare_groups(mond_wins, sigma_wins_list, "MOND wins", "Σ-Gravity wins")

# =============================================================================
# SPECIFIC ANALYSIS: Deep MOND MOND-winners
# =============================================================================

print("\n\n" + "=" * 100)
print("DEEP ANALYSIS: CHARACTERISTICS OF DEEP MOND GALAXIES WHERE MOND WINS")
print("=" * 100)

deep_mond_all = [r for r in results if r['accel_regime'] == 'Deep MOND']
deep_mond_sigma_wins = [r for r in deep_mond_all if r['sigma_wins']]

if len(deep_mond_mond_wins) > 0 and len(deep_mond_sigma_wins) > 0:
    compare_groups(deep_mond_mond_wins, deep_mond_sigma_wins, 
                   "Deep MOND, MOND wins", "Deep MOND, Σ wins")

# =============================================================================
# HUBBLE TYPE DISTRIBUTION IN MOND WINNERS
# =============================================================================

print("\n\n" + "=" * 100)
print("HUBBLE TYPE DISTRIBUTION IN MOND WINNERS")
print("=" * 100)

mond_hubble_counts = {}
for r in mond_wins:
    ht = r['hubble_type']
    mond_hubble_counts[ht] = mond_hubble_counts.get(ht, 0) + 1

total_hubble_counts = {}
for r in results:
    ht = r['hubble_type']
    total_hubble_counts[ht] = total_hubble_counts.get(ht, 0) + 1

print(f"\n{'Hubble Type':<12} {'MOND wins':>10} {'Total':>10} {'% MOND wins':>12}")
print("-" * 50)

for ht in hubble_order:
    if ht in total_hubble_counts:
        mond_n = mond_hubble_counts.get(ht, 0)
        total_n = total_hubble_counts[ht]
        pct = 100 * mond_n / total_n if total_n > 0 else 0
        print(f"{ht:<12} {mond_n:>10} {total_n:>10} {pct:>11.1f}%")

# =============================================================================
# CSV OUTPUT FOR FURTHER ANALYSIS
# =============================================================================

csv_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/galaxy_breakdown_results.csv")
print(f"\n\nWriting CSV to: {csv_path}")

with open(csv_path, 'w') as f:
    # Header
    headers = ['Galaxy', 'Hubble_Type', 'Winner', 'RMS_Sigma', 'RMS_MOND', 'Improvement_Pct',
               'V_flat', 'R_d', 'R_d_source', 'Gas_Pct', 'Bulge_Pct', 'Deep_MOND_Pct',
               'g_bar_min', 'g_bar_max', 'g_bar_mean', 'Accel_Regime', 'Mass_Type',
               'Rise_Ratio', 'R_max', 'N_points', 'Inclination', 'Distance',
               'Luminosity', 'M_HI', 'R_HI', 'Quality']
    f.write(','.join(headers) + '\n')
    
    for r in results_sorted:
        winner = 'Sigma' if r['sigma_wins'] else 'MOND'
        row = [
            r['name'], r['hubble_type'], winner,
            f"{r['rms_sigma']:.3f}", f"{r['rms_mond']:.3f}", f"{r['improvement']*100:.2f}",
            f"{r['V_flat']:.2f}", f"{r['R_d']:.3f}", r['R_d_source'],
            f"{r['gas_fraction']*100:.1f}", f"{r['bulge_fraction']*100:.1f}", f"{r['frac_deep_mond']*100:.1f}",
            f"{r['g_bar_min']:.4e}", f"{r['g_bar_max']:.4e}", f"{r['g_bar_mean']:.4e}",
            r['accel_regime'], r['mass_type'],
            f"{r['rise_ratio']:.3f}", f"{r['R_max']:.2f}", str(r['n_points']),
            f"{r['inclination']:.1f}", f"{r['distance']:.2f}",
            f"{r['luminosity']:.4f}", f"{r['M_HI']:.4f}", f"{r['R_HI']:.2f}",
            str(r['quality'])
        ]
        f.write(','.join(row) + '\n')

print(f"CSV file written with {len(results)} galaxies")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

n_sigma_wins = sum(1 for r in results if r['sigma_wins'])
mean_rms_sigma = np.mean([r['rms_sigma'] for r in results])
mean_rms_mond = np.mean([r['rms_mond'] for r in results])
median_improvement = np.median([r['improvement'] for r in results])

print(f"""
Total galaxies analyzed: {len(results)}

OVERALL PERFORMANCE:
  Σ-Gravity wins: {n_sigma_wins}/{len(results)} ({100*n_sigma_wins/len(results):.1f}%)
  MOND wins: {len(mond_wins)}/{len(results)} ({100*len(mond_wins)/len(results):.1f}%)
  
  Mean RMS:
    Σ-Gravity: {mean_rms_sigma:.2f} km/s
    MOND: {mean_rms_mond:.2f} km/s
  
  Median improvement: {median_improvement*100:+.1f}%

MOND WINNERS SUMMARY:
  Number: {len(mond_wins)}
  Most common Hubble types: {', '.join(f"{k}({v})" for k,v in sorted(mond_hubble_counts.items(), key=lambda x: -x[1])[:5])}
  Mean gas fraction: {np.mean([r['gas_fraction'] for r in mond_wins])*100:.0f}%
  Mean deep MOND fraction: {np.mean([r['frac_deep_mond'] for r in mond_wins])*100:.0f}%
  Mean V_flat: {np.mean([r['V_flat'] for r in mond_wins]):.0f} km/s
""")

