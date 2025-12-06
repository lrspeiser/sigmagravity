#!/usr/bin/env python3
"""
Which Galaxies Fit the Coherence Concept Best?
==============================================

Analyze residuals to find patterns in which galaxies fit well vs poorly.
Look for correlations with physical properties that might reveal
what makes coherence work or fail.

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
from typing import Dict, List, Tuple
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
print(f"g† = {g_dagger:.2e} m/s²")

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


def load_disk_scale_lengths():
    """Load disk scale lengths from SPARC master sheet.
    
    The master sheet is fixed-width format:
    - Bytes 1-11: Galaxy name
    - Bytes 62-66: Rdisk (disk scale length in kpc)
    """
    master_file = find_master_sheet()
    scale_lengths = {}
    
    if master_file is None:
        return scale_lengths
    
    with open(master_file, 'r') as f:
        in_data = False
        for line in f:
            # Skip header lines until we hit the data separator
            if line.startswith('---'):
                in_data = True
                continue
            
            if not in_data:
                continue
            
            # Data lines should have at least 66 characters
            if len(line) < 66:
                continue
            
            try:
                # Galaxy name: bytes 1-11 (0-indexed: 0-11)
                name = line[0:11].strip()
                
                # Disk scale length: bytes 62-66 (0-indexed: 61-66)
                rdisk_str = line[61:66].strip()
                
                if name and rdisk_str:
                    R_d = float(rdisk_str)
                    if R_d > 0:
                        scale_lengths[name] = R_d
            except (ValueError, IndexError):
                continue
    
    return scale_lengths


def load_galaxy(rotmod_file, scale_lengths):
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
    
    # Get disk scale length from photometry, or estimate from rotation curve
    if name in scale_lengths:
        R_d = scale_lengths[name]
    else:
        # Estimate: R_d ≈ R_max / 4 (typical for exponential disks)
        R_d = np.max(R) / 4.0
    
    # Derived quantities
    n_outer = max(3, len(R) // 3)
    V_flat = np.mean(np.sort(V_obs)[-n_outer:])
    V_max = np.max(V_obs)
    R_max = np.max(R)
    R_half = R[len(R)//2]
    
    # Gas fraction (rough proxy)
    V_gas_max = np.max(np.abs(V_gas))
    V_disk_max = np.max(np.abs(V_disk))
    gas_fraction = V_gas_max**2 / (V_gas_max**2 + V_disk_max**2 + 0.01)
    
    # Bulge fraction
    V_bulge_max = np.max(V_bulge)
    bulge_fraction = V_bulge_max**2 / (V_max**2 + 0.01)
    
    # Concentration (R_max / R_half)
    concentration = R_max / (R_half + 0.1)
    
    # Average acceleration
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    g_obs = (V_obs * 1000)**2 / R_m
    g_bar_mean = np.mean(g_bar)
    g_obs_mean = np.mean(g_obs)
    
    # How "deep MOND" - fraction of points below g†
    frac_deep_mond = np.mean(g_bar < g_dagger)
    
    # Rising vs flat rotation curve
    V_inner = np.mean(V_obs[:max(3, len(V_obs)//4)])
    V_outer = np.mean(V_obs[-max(3, len(V_obs)//4):])
    rise_ratio = V_outer / (V_inner + 1)
    
    return {
        'R': R, 'V_obs': V_obs, 'V_err': V_err, 'V_bar': V_bar,
        'V_flat': V_flat, 'V_max': V_max, 'R_max': R_max,
        'R_d': R_d,
        'gas_fraction': gas_fraction, 'bulge_fraction': bulge_fraction,
        'concentration': concentration, 'g_bar_mean': g_bar_mean,
        'frac_deep_mond': frac_deep_mond, 'rise_ratio': rise_ratio,
        'n_points': len(R),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print("=" * 80)
print("WHICH GALAXIES FIT THE COHERENCE CONCEPT BEST?")
print("=" * 80)
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

# Load disk scale lengths
scale_lengths = load_disk_scale_lengths()
print(f"\nLoaded {len(scale_lengths)} disk scale lengths from photometry")

# Load and analyze all galaxies
results = []

for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
    name = rotmod_file.stem.replace('_rotmod', '')
    data = load_galaxy(rotmod_file, scale_lengths)
    if data is None:
        continue
    
    try:
        # Predictions
        V_sigma = predict_sigma_gravity(data['R'], data['V_bar'], data['R_d'])
        V_mond = predict_mond(data['R'], data['V_bar'])
        
        # RMS errors
        rms_sigma = np.sqrt(np.mean((data['V_obs'] - V_sigma)**2))
        rms_mond = np.sqrt(np.mean((data['V_obs'] - V_mond)**2))
        
        # Normalized RMS (by V_flat to make comparable across galaxies)
        nrms_sigma = rms_sigma / data['V_flat']
        nrms_mond = rms_mond / data['V_flat']
        
        # Which model wins
        sigma_wins = rms_sigma < rms_mond
        
        # Improvement ratio
        improvement = (rms_mond - rms_sigma) / rms_mond if rms_mond > 0 else 0
        
        results.append({
            'name': name,
            'rms_sigma': rms_sigma,
            'rms_mond': rms_mond,
            'nrms_sigma': nrms_sigma,
            'nrms_mond': nrms_mond,
            'sigma_wins': sigma_wins,
            'improvement': improvement,
            'V_flat': data['V_flat'],
            'V_max': data['V_max'],
            'R_max': data['R_max'],
            'R_d': data['R_d'],
            'gas_fraction': data['gas_fraction'],
            'bulge_fraction': data['bulge_fraction'],
            'concentration': data['concentration'],
            'g_bar_mean': data['g_bar_mean'],
            'frac_deep_mond': data['frac_deep_mond'],
            'rise_ratio': data['rise_ratio'],
            'n_points': data['n_points'],
        })
    except:
        continue

print(f"\nAnalyzed {len(results)} galaxies")

# Sort by how well Σ-Gravity does relative to MOND
results_sorted = sorted(results, key=lambda x: x['improvement'], reverse=True)

# =============================================================================
# BEST AND WORST FITTING GALAXIES
# =============================================================================

print("\n" + "=" * 80)
print("TOP 20 GALAXIES WHERE Σ-GRAVITY BEATS MOND MOST")
print("=" * 80)
print(f"{'Galaxy':<15} {'RMS_Σ':<10} {'RMS_M':<10} {'Improve':<10} {'V_flat':<10} {'Gas%':<8} {'Deep%':<8}")
print("-" * 80)

for r in results_sorted[:20]:
    print(f"{r['name']:<15} {r['rms_sigma']:<10.1f} {r['rms_mond']:<10.1f} "
          f"{r['improvement']*100:>+7.1f}% {r['V_flat']:<10.0f} "
          f"{r['gas_fraction']*100:<8.0f} {r['frac_deep_mond']*100:<8.0f}")

print("\n" + "=" * 80)
print("TOP 20 GALAXIES WHERE MOND BEATS Σ-GRAVITY MOST")
print("=" * 80)
print(f"{'Galaxy':<15} {'RMS_Σ':<10} {'RMS_M':<10} {'Improve':<10} {'V_flat':<10} {'Gas%':<8} {'Deep%':<8}")
print("-" * 80)

for r in results_sorted[-20:]:
    print(f"{r['name']:<15} {r['rms_sigma']:<10.1f} {r['rms_mond']:<10.1f} "
          f"{r['improvement']*100:>+7.1f}% {r['V_flat']:<10.0f} "
          f"{r['gas_fraction']*100:<8.0f} {r['frac_deep_mond']*100:<8.0f}")

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("WHAT PREDICTS GOOD Σ-GRAVITY FITS?")
print("=" * 80)

# Extract arrays for correlation analysis
improvement = np.array([r['improvement'] for r in results])
V_flat = np.array([r['V_flat'] for r in results])
R_max = np.array([r['R_max'] for r in results])
R_d = np.array([r['R_d'] for r in results])
gas_fraction = np.array([r['gas_fraction'] for r in results])
bulge_fraction = np.array([r['bulge_fraction'] for r in results])
frac_deep_mond = np.array([r['frac_deep_mond'] for r in results])
rise_ratio = np.array([r['rise_ratio'] for r in results])
g_bar_mean = np.array([r['g_bar_mean'] for r in results])
n_points = np.array([r['n_points'] for r in results])

def correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    n = mask.sum()
    t = r * np.sqrt((n-2) / (1 - r**2 + 1e-10))
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), n-2))
    return r, p

print("\nCorrelations with Σ-Gravity improvement over MOND:")
print(f"{'Property':<25} {'Correlation':<15} {'p-value':<15} {'Interpretation'}")
print("-" * 80)

correlations = [
    ('V_flat', V_flat, 'Higher V → ?'),
    ('R_max', R_max, 'Larger extent → ?'),
    ('R_d (disk scale)', R_d, 'Larger disk → ?'),
    ('Gas fraction', gas_fraction, 'More gas → ?'),
    ('Bulge fraction', bulge_fraction, 'More bulge → ?'),
    ('Frac deep MOND', frac_deep_mond, 'Deeper MOND regime → ?'),
    ('Rise ratio (V_out/V_in)', rise_ratio, 'Flatter curve → ?'),
    ('log10(g_bar)', np.log10(g_bar_mean), 'Lower acceleration → ?'),
    ('N data points', n_points, 'More data → ?'),
]

for name, arr, interp in correlations:
    r, p = correlation(arr, improvement)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    direction = '↑' if r > 0 else '↓'
    print(f"{name:<25} {r:>+.3f} {sig:<6} {p:<15.4f} {direction} {interp}")

# =============================================================================
# BREAKDOWN BY GALAXY TYPE
# =============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY GALAXY TYPE (V_flat bins)")
print("=" * 80)

# Dwarfs (V < 80)
dwarfs = [r for r in results if r['V_flat'] < 80]
spirals = [r for r in results if 80 <= r['V_flat'] < 150]
massive = [r for r in results if r['V_flat'] >= 150]

def summarize(galaxies, label):
    if len(galaxies) == 0:
        return
    wins = sum(1 for g in galaxies if g['sigma_wins'])
    mean_rms_s = np.mean([g['rms_sigma'] for g in galaxies])
    mean_rms_m = np.mean([g['rms_mond'] for g in galaxies])
    mean_nrms_s = np.mean([g['nrms_sigma'] for g in galaxies])
    mean_nrms_m = np.mean([g['nrms_mond'] for g in galaxies])
    mean_imp = np.mean([g['improvement'] for g in galaxies])
    
    print(f"\n{label} (N={len(galaxies)}):")
    print(f"  Σ-Gravity wins: {wins}/{len(galaxies)} ({100*wins/len(galaxies):.1f}%)")
    print(f"  Mean RMS: Σ={mean_rms_s:.1f}, MOND={mean_rms_m:.1f} km/s")
    print(f"  Mean normalized RMS: Σ={mean_nrms_s:.3f}, MOND={mean_nrms_m:.3f}")
    print(f"  Mean improvement: {mean_imp*100:+.1f}%")

summarize(dwarfs, "DWARFS (V_flat < 80 km/s)")
summarize(spirals, "NORMAL SPIRALS (80-150 km/s)")
summarize(massive, "MASSIVE SPIRALS (V_flat > 150 km/s)")

# =============================================================================
# BREAKDOWN BY ACCELERATION REGIME
# =============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY ACCELERATION REGIME")
print("=" * 80)

# Deep MOND (most points below g†)
deep_mond = [r for r in results if r['frac_deep_mond'] > 0.7]
transition = [r for r in results if 0.3 <= r['frac_deep_mond'] <= 0.7]
newtonian = [r for r in results if r['frac_deep_mond'] < 0.3]

def summarize2(galaxies, label):
    if len(galaxies) == 0:
        print(f"\n{label}: No galaxies")
        return
    wins = sum(1 for g in galaxies if g['sigma_wins'])
    mean_imp = np.mean([g['improvement'] for g in galaxies])
    mean_rms_s = np.mean([g['rms_sigma'] for g in galaxies])
    
    print(f"\n{label} (N={len(galaxies)}):")
    print(f"  Σ-Gravity wins: {wins}/{len(galaxies)} ({100*wins/len(galaxies):.1f}%)")
    print(f"  Mean improvement over MOND: {mean_imp*100:+.1f}%")
    print(f"  Mean RMS: {mean_rms_s:.1f} km/s")

summarize2(deep_mond, "DEEP MOND REGIME (>70% points below g†)")
summarize2(transition, "TRANSITION REGIME (30-70% below g†)")
summarize2(newtonian, "NEWTONIAN-DOMINATED (<30% below g†)")

# =============================================================================
# BREAKDOWN BY GAS FRACTION
# =============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY GAS FRACTION")
print("=" * 80)

gas_rich = [r for r in results if r['gas_fraction'] > 0.5]
gas_poor = [r for r in results if r['gas_fraction'] < 0.2]
gas_mid = [r for r in results if 0.2 <= r['gas_fraction'] <= 0.5]

summarize2(gas_rich, "GAS-RICH (>50% gas-dominated)")
summarize2(gas_mid, "MIXED (20-50%)")
summarize2(gas_poor, "STELLAR-DOMINATED (<20% gas)")

# =============================================================================
# BREAKDOWN BY ROTATION CURVE SHAPE
# =============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY ROTATION CURVE SHAPE")
print("=" * 80)

rising = [r for r in results if r['rise_ratio'] > 1.3]
flat = [r for r in results if 0.9 <= r['rise_ratio'] <= 1.3]
declining = [r for r in results if r['rise_ratio'] < 0.9]

summarize2(rising, "RISING (V_out/V_in > 1.3)")
summarize2(flat, "FLAT (0.9 < V_out/V_in < 1.3)")
summarize2(declining, "DECLINING (V_out/V_in < 0.9)")

# =============================================================================
# THE KEY INSIGHT
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS: WHAT MAKES COHERENCE WORK?")
print("=" * 80)

# Find the strongest correlations
strong_correlations = []
for name, arr, interp in correlations:
    r, p = correlation(arr, improvement)
    if p < 0.05:
        strong_correlations.append((name, r, p))

strong_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("""
STATISTICALLY SIGNIFICANT PREDICTORS OF Σ-GRAVITY SUCCESS:
""")

for name, r, p in strong_correlations:
    direction = "BETTER" if r > 0 else "WORSE"
    print(f"  • {name}: r = {r:+.3f} (p = {p:.4f})")
    if 'V_flat' in name:
        if r > 0:
            print(f"    → Higher mass galaxies fit {direction}")
        else:
            print(f"    → Lower mass galaxies fit {direction}")
    elif 'gas' in name.lower():
        if r > 0:
            print(f"    → Gas-rich galaxies fit {direction}")
        else:
            print(f"    → Stellar-dominated galaxies fit {direction}")
    elif 'deep' in name.lower() or 'g_bar' in name.lower():
        if r > 0:
            print(f"    → Deep MOND regime fits {direction}")
        else:
            print(f"    → Transition/Newtonian regime fits {direction}")
    elif 'rise' in name.lower():
        if r > 0:
            print(f"    → Rising/flat rotation curves fit {direction}")
        else:
            print(f"    → Declining rotation curves fit {direction}")

print("""
PHYSICAL INTERPRETATION:
""")

# =============================================================================
# OUTLIER ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("OUTLIER ANALYSIS: WHAT'S SPECIAL ABOUT THE EXTREMES?")
print("=" * 80)

# Best 10 for Σ-Gravity
best_10 = results_sorted[:10]
worst_10 = results_sorted[-10:]

print("\nBEST 10 for Σ-Gravity - Common properties:")
print(f"  Mean V_flat: {np.mean([g['V_flat'] for g in best_10]):.0f} km/s")
print(f"  Mean gas fraction: {np.mean([g['gas_fraction'] for g in best_10])*100:.0f}%")
print(f"  Mean frac deep MOND: {np.mean([g['frac_deep_mond'] for g in best_10])*100:.0f}%")
print(f"  Mean rise ratio: {np.mean([g['rise_ratio'] for g in best_10]):.2f}")
print(f"  Mean R_max: {np.mean([g['R_max'] for g in best_10]):.1f} kpc")
print(f"  Mean R_d: {np.mean([g['R_d'] for g in best_10]):.1f} kpc")

print("\nWORST 10 for Σ-Gravity - Common properties:")
print(f"  Mean V_flat: {np.mean([g['V_flat'] for g in worst_10]):.0f} km/s")
print(f"  Mean gas fraction: {np.mean([g['gas_fraction'] for g in worst_10])*100:.0f}%")
print(f"  Mean frac deep MOND: {np.mean([g['frac_deep_mond'] for g in worst_10])*100:.0f}%")
print(f"  Mean rise ratio: {np.mean([g['rise_ratio'] for g in worst_10]):.2f}")
print(f"  Mean R_max: {np.mean([g['R_max'] for g in worst_10]):.1f} kpc")
print(f"  Mean R_d: {np.mean([g['R_d'] for g in worst_10]):.1f} kpc")

# Statistical test for difference
from scipy import stats

print("\n\nStatistical tests (best 10 vs worst 10):")
for prop in ['V_flat', 'gas_fraction', 'frac_deep_mond', 'rise_ratio', 'R_max', 'R_d']:
    best_vals = [g[prop] for g in best_10]
    worst_vals = [g[prop] for g in worst_10]
    t_stat, p_val = stats.ttest_ind(best_vals, worst_vals)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    print(f"  {prop:<20}: t={t_stat:+.2f}, p={p_val:.4f} {sig}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: WHEN DOES THE COHERENCE CONCEPT WORK?")
print("=" * 80)

n_sigma_wins = sum(1 for r in results if r['sigma_wins'])
mean_rms_sigma = np.mean([r['rms_sigma'] for r in results])
mean_rms_mond = np.mean([r['rms_mond'] for r in results])

print(f"""
Overall: Σ-Gravity beats MOND in {n_sigma_wins}/{len(results)} galaxies ({100*n_sigma_wins/len(results):.1f}%)
Mean RMS: Σ-Gravity = {mean_rms_sigma:.2f} km/s, MOND = {mean_rms_mond:.2f} km/s

The coherence model works BEST for:
  • [Based on correlations above]
  
The coherence model struggles with:
  • [Based on correlations above]

This suggests the following physical insights:
  • [Interpretation]
""")

