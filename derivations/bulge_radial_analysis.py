#!/usr/bin/env python3
"""
BULGE GALAXY RADIAL ANALYSIS

Questions:
1. How do bulge galaxies behave at different radii?
2. Is there a difference between compact vs extended bulge galaxies?
3. Where exactly does the model fail - inner, middle, or outer regions?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

c = 3e8
H0 = 2.27e-18
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

data_dir = Path(__file__).parent.parent / "data"
rotmod_dir = data_dir / "Rotmod_LTG"

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return r / (xi + r)

def load_galaxies():
    galaxies = []
    for f in sorted(rotmod_dir.glob("*.dat")):
        try:
            lines = f.read_text().strip().split('\n')
            data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
            if len(data_lines) < 3:
                continue
            
            data = np.array([list(map(float, l.split())) for l in data_lines])
            
            R = data[:, 0]
            V_obs = data[:, 1]
            V_gas = data[:, 3] if data.shape[1] > 3 else np.zeros_like(R)
            V_disk = data[:, 4] if data.shape[1] > 4 else np.zeros_like(R)
            V_bulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
            
            V_disk_scaled = np.abs(V_disk) * np.sqrt(0.5)
            V_bulge_scaled = np.abs(V_bulge) * np.sqrt(0.7)
            
            V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk_scaled**2 + V_bulge_scaled**2
            if np.any(V_bar_sq <= 0):
                continue
            V_bar = np.sqrt(np.maximum(V_bar_sq, 1e-10))
            
            if np.sum(V_disk**2) > 0:
                cumsum = np.cumsum(V_disk**2 * R)
                half_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                R_d = R[min(half_idx, len(R) - 1)]
            else:
                R_d = R[-1] / 3
            R_d = max(R_d, 0.3)
            
            # Global bulge fraction
            f_bulge = np.sum(V_bulge_scaled**2) / max(np.sum(V_bar_sq), 1e-10)
            
            # Local bulge fraction at each radius
            f_bulge_local = V_bulge_scaled**2 / np.maximum(V_bar_sq, 1e-10)
            
            # Find bulge effective radius (where bulge contribution drops to half)
            if np.sum(V_bulge_scaled**2) > 0:
                bulge_cumsum = np.cumsum(V_bulge_scaled**2)
                half_bulge_idx = np.searchsorted(bulge_cumsum, bulge_cumsum[-1] / 2)
                R_bulge = R[min(half_bulge_idx, len(R) - 1)]
            else:
                R_bulge = 0
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'V_bulge': V_bulge_scaled,
                'V_disk': V_disk_scaled,
                'R_d': R_d,
                'R_max': R.max(),
                'R_bulge': R_bulge,
                'f_bulge': f_bulge,
                'f_bulge_local': f_bulge_local,
                'V_flat': np.median(V_obs[-5:]) if len(V_obs) >= 5 else V_obs[-1],
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()
bulge_galaxies = [g for g in galaxies if g['f_bulge'] > 0.1]

print("=" * 80)
print("BULGE GALAXY RADIAL ANALYSIS")
print("=" * 80)
print(f"\nLoaded {len(galaxies)} galaxies, {len(bulge_galaxies)} with significant bulge (f > 0.1)")

# =============================================================================
# ANALYZE RESIDUALS BY RADIUS
# =============================================================================
print("\n" + "=" * 80)
print("RESIDUALS BY RADIAL ZONE")
print("=" * 80)

A = np.exp(1 / (2 * np.pi))
xi_coeff = 1 / (2 * np.pi)

def compute_residuals(gal):
    R_m = gal['R'] * kpc_to_m
    g_bar = (gal['V_bar'] * 1000)**2 / R_m
    xi = xi_coeff * gal['R_d']
    W = W_coherence(gal['R'], xi)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    V_pred = gal['V_bar'] * np.sqrt(Sigma)
    return gal['V_obs'] - V_pred

# Split each galaxy into inner, middle, outer zones
zone_results = {'inner': [], 'middle': [], 'outer': []}

for gal in bulge_galaxies:
    residuals = compute_residuals(gal)
    R = gal['R']
    R_max = R.max()
    
    # Define zones
    inner_mask = R < R_max / 3
    middle_mask = (R >= R_max / 3) & (R < 2 * R_max / 3)
    outer_mask = R >= 2 * R_max / 3
    
    if inner_mask.sum() > 0:
        zone_results['inner'].append({
            'name': gal['name'],
            'rms': np.sqrt(np.mean(residuals[inner_mask]**2)),
            'mean_residual': np.mean(residuals[inner_mask]),
            'f_bulge': gal['f_bulge'],
            'R_bulge': gal['R_bulge'],
            'R_max': R_max,
        })
    if middle_mask.sum() > 0:
        zone_results['middle'].append({
            'name': gal['name'],
            'rms': np.sqrt(np.mean(residuals[middle_mask]**2)),
            'mean_residual': np.mean(residuals[middle_mask]),
            'f_bulge': gal['f_bulge'],
        })
    if outer_mask.sum() > 0:
        zone_results['outer'].append({
            'name': gal['name'],
            'rms': np.sqrt(np.mean(residuals[outer_mask]**2)),
            'mean_residual': np.mean(residuals[outer_mask]),
            'f_bulge': gal['f_bulge'],
        })

print("\nMean RMS by zone for bulge galaxies:")
print(f"\n  {'Zone':<12} {'Mean RMS':<12} {'Mean Residual':<15}")
print("  " + "-" * 40)
for zone in ['inner', 'middle', 'outer']:
    df = pd.DataFrame(zone_results[zone])
    print(f"  {zone:<12} {df['rms'].mean():<12.1f} {df['mean_residual'].mean():+.1f}")

# =============================================================================
# COMPACT vs EXTENDED BULGE GALAXIES
# =============================================================================
print("\n" + "=" * 80)
print("COMPACT vs EXTENDED BULGE GALAXIES")
print("=" * 80)

# Classify by R_bulge / R_max ratio (compactness)
for gal in bulge_galaxies:
    gal['compactness'] = gal['R_bulge'] / gal['R_max'] if gal['R_max'] > 0 else 0

# Split into compact (small R_bulge/R_max) and extended (large R_bulge/R_max)
median_compactness = np.median([g['compactness'] for g in bulge_galaxies])
compact = [g for g in bulge_galaxies if g['compactness'] < median_compactness]
extended = [g for g in bulge_galaxies if g['compactness'] >= median_compactness]

print(f"\nMedian compactness (R_bulge/R_max): {median_compactness:.3f}")
print(f"Compact bulges (< median): {len(compact)} galaxies")
print(f"Extended bulges (≥ median): {len(extended)} galaxies")

# Compute RMS for each group
def compute_rms(gal):
    residuals = compute_residuals(gal)
    return np.sqrt(np.mean(residuals**2))

compact_rms = [compute_rms(g) for g in compact]
extended_rms = [compute_rms(g) for g in extended]

print(f"\n  {'Group':<20} {'N':<6} {'Mean RMS':<12} {'Median RMS':<12}")
print("  " + "-" * 55)
print(f"  {'Compact bulge':<20} {len(compact):<6} {np.mean(compact_rms):<12.1f} {np.median(compact_rms):<12.1f}")
print(f"  {'Extended bulge':<20} {len(extended):<6} {np.mean(extended_rms):<12.1f} {np.median(extended_rms):<12.1f}")

# Statistical test
_, p_value = stats.mannwhitneyu(compact_rms, extended_rms)
print(f"\n  Mann-Whitney U test p-value: {p_value:.4f}")

# =============================================================================
# DETAILED RADIAL PROFILES
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED RADIAL PROFILES FOR BULGE GALAXIES")
print("=" * 80)

print("\nFor each bulge galaxy, showing residuals at different radii:")
print(f"\n  {'Galaxy':<20} {'f_bulge':<8} {'R_bulge':<10} {'Inner RMS':<12} {'Outer RMS':<12} {'Pattern':<15}")
print("  " + "-" * 85)

patterns = []
for gal in sorted(bulge_galaxies, key=lambda x: x['f_bulge'], reverse=True)[:20]:
    residuals = compute_residuals(gal)
    R = gal['R']
    R_max = R.max()
    
    inner_mask = R < R_max / 2
    outer_mask = R >= R_max / 2
    
    inner_rms = np.sqrt(np.mean(residuals[inner_mask]**2)) if inner_mask.sum() > 0 else 0
    outer_rms = np.sqrt(np.mean(residuals[outer_mask]**2)) if outer_mask.sum() > 0 else 0
    
    # Determine pattern
    if outer_rms > inner_rms * 1.5:
        pattern = "OUTER WORSE"
    elif inner_rms > outer_rms * 1.5:
        pattern = "INNER WORSE"
    else:
        pattern = "UNIFORM"
    
    patterns.append({
        'name': gal['name'],
        'f_bulge': gal['f_bulge'],
        'R_bulge': gal['R_bulge'],
        'inner_rms': inner_rms,
        'outer_rms': outer_rms,
        'pattern': pattern,
    })
    
    print(f"  {gal['name']:<20} {gal['f_bulge']:<8.2f} {gal['R_bulge']:<10.1f} {inner_rms:<12.1f} {outer_rms:<12.1f} {pattern:<15}")

patterns_df = pd.DataFrame(patterns)
print(f"\nPattern distribution:")
print(patterns_df['pattern'].value_counts())

# =============================================================================
# RESIDUAL SIGN BY RADIUS
# =============================================================================
print("\n" + "=" * 80)
print("RESIDUAL SIGN BY RADIUS (OVER/UNDER PREDICTION)")
print("=" * 80)

print("""
Positive residual = V_obs > V_pred (model UNDER-predicts)
Negative residual = V_obs < V_pred (model OVER-predicts)
""")

inner_residuals = []
outer_residuals = []

for gal in bulge_galaxies:
    residuals = compute_residuals(gal)
    R = gal['R']
    R_max = R.max()
    
    inner_mask = R < R_max / 2
    outer_mask = R >= R_max / 2
    
    if inner_mask.sum() > 0:
        inner_residuals.extend(residuals[inner_mask])
    if outer_mask.sum() > 0:
        outer_residuals.extend(residuals[outer_mask])

print(f"Inner regions (R < R_max/2):")
print(f"  Mean residual: {np.mean(inner_residuals):+.1f} km/s")
print(f"  Positive (under-predict): {(np.array(inner_residuals) > 0).sum()}/{len(inner_residuals)}")

print(f"\nOuter regions (R ≥ R_max/2):")
print(f"  Mean residual: {np.mean(outer_residuals):+.1f} km/s")
print(f"  Positive (under-predict): {(np.array(outer_residuals) > 0).sum()}/{len(outer_residuals)}")

# =============================================================================
# COMPARE BULGE vs NO-BULGE AT SAME RADII
# =============================================================================
print("\n" + "=" * 80)
print("BULGE vs NO-BULGE GALAXIES AT OUTER RADII")
print("=" * 80)

no_bulge = [g for g in galaxies if g['f_bulge'] < 0.01]

print(f"\nComparing outer region (R > R_max/2) performance:")
print(f"  Bulge galaxies: {len(bulge_galaxies)}")
print(f"  No-bulge galaxies: {len(no_bulge)}")

bulge_outer_rms = []
no_bulge_outer_rms = []

for gal in bulge_galaxies:
    residuals = compute_residuals(gal)
    R = gal['R']
    outer_mask = R >= gal['R_max'] / 2
    if outer_mask.sum() > 0:
        bulge_outer_rms.append(np.sqrt(np.mean(residuals[outer_mask]**2)))

for gal in no_bulge:
    residuals = compute_residuals(gal)
    R = gal['R']
    outer_mask = R >= gal['R_max'] / 2
    if outer_mask.sum() > 0:
        no_bulge_outer_rms.append(np.sqrt(np.mean(residuals[outer_mask]**2)))

print(f"\n  {'Group':<20} {'Mean Outer RMS':<15} {'Median':<12}")
print("  " + "-" * 50)
print(f"  {'Bulge galaxies':<20} {np.mean(bulge_outer_rms):<15.1f} {np.median(bulge_outer_rms):<12.1f}")
print(f"  {'No-bulge galaxies':<20} {np.mean(no_bulge_outer_rms):<15.1f} {np.median(no_bulge_outer_rms):<12.1f}")

# =============================================================================
# VELOCITY PROFILE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("VELOCITY PROFILE SHAPES")
print("=" * 80)

print("\nAnalyzing rotation curve shapes for bulge galaxies:")

for gal in sorted(bulge_galaxies, key=lambda x: x['f_bulge'], reverse=True)[:10]:
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    
    # Compute velocity gradients
    if len(R) > 5:
        inner_gradient = (V_obs[2] - V_obs[0]) / (R[2] - R[0])
        outer_gradient = (V_obs[-1] - V_obs[-3]) / (R[-1] - R[-3])
        
        # Is curve rising, flat, or declining?
        if outer_gradient > 5:
            shape = "RISING"
        elif outer_gradient < -5:
            shape = "DECLINING"
        else:
            shape = "FLAT"
        
        # Baryonic vs observed ratio at outer radii
        V_ratio = V_obs[-1] / V_bar[-1]
        
        print(f"  {gal['name']:<20}: {shape:<10} V_obs/V_bar = {V_ratio:.2f}, outer gradient = {outer_gradient:+.1f} km/s/kpc")

# =============================================================================
# LOCAL BULGE FRACTION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("LOCAL BULGE FRACTION vs RESIDUALS")
print("=" * 80)

all_f_bulge_local = []
all_residuals = []

for gal in bulge_galaxies:
    residuals = compute_residuals(gal)
    f_bulge_local = gal['f_bulge_local']
    
    all_f_bulge_local.extend(f_bulge_local)
    all_residuals.extend(residuals)

# Correlation
r, p = stats.pearsonr(all_f_bulge_local, all_residuals)
print(f"\nCorrelation between local f_bulge and residual:")
print(f"  r = {r:.3f}, p = {p:.4f}")

# Bin by local bulge fraction
bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
for i in range(len(bins) - 1):
    mask = (np.array(all_f_bulge_local) >= bins[i]) & (np.array(all_f_bulge_local) < bins[i+1])
    if mask.sum() > 0:
        res = np.array(all_residuals)[mask]
        print(f"  f_bulge [{bins[i]:.1f}-{bins[i+1]:.1f}]: mean residual = {np.mean(res):+.1f} km/s, N = {mask.sum()}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
KEY FINDINGS:

1. RADIAL ZONE PERFORMANCE:
   - Inner regions: RMS = {pd.DataFrame(zone_results['inner'])['rms'].mean():.1f} km/s
   - Middle regions: RMS = {pd.DataFrame(zone_results['middle'])['rms'].mean():.1f} km/s  
   - Outer regions: RMS = {pd.DataFrame(zone_results['outer'])['rms'].mean():.1f} km/s

2. COMPACT vs EXTENDED BULGES:
   - Compact bulges: {np.mean(compact_rms):.1f} km/s
   - Extended bulges: {np.mean(extended_rms):.1f} km/s
   - p-value: {p_value:.4f}

3. RESIDUAL PATTERNS:
   - Inner regions: Mean residual = {np.mean(inner_residuals):+.1f} km/s
   - Outer regions: Mean residual = {np.mean(outer_residuals):+.1f} km/s
   
   Positive = under-prediction, Negative = over-prediction

4. BULGE vs NO-BULGE AT OUTER RADII:
   - Bulge galaxies outer RMS: {np.mean(bulge_outer_rms):.1f} km/s
   - No-bulge galaxies outer RMS: {np.mean(no_bulge_outer_rms):.1f} km/s

5. LOCAL BULGE FRACTION CORRELATION:
   - r = {r:.3f} (correlation with residual)
   - Where bulge is dominant locally, residuals tend to be...
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

