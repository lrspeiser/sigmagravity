#!/usr/bin/env python3
"""
GEOMETRY PHYSICS DEEP DIVE

Key findings from root cause analysis:
1. 2D regions (disk/gas) have V_ratio = 1.88
2. 3D regions (bulge) have V_ratio = 1.14
3. After controlling for g, more z-axis → LOWER V_ratio

This is OPPOSITE of what we expected!
- We thought 3D (like clusters) needs MORE enhancement
- But data shows 3D regions have LESS enhancement

Let's understand why and what this means for the theory.
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

print("=" * 80)
print("GEOMETRY PHYSICS DEEP DIVE")
print("=" * 80)

# Load data (same as before)
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
            
            f_bulge_local = V_bulge_scaled**2 / np.maximum(V_bar_sq, 1e-10)
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R, 'V_obs': V_obs, 'V_bar': V_bar,
                'V_bulge': V_bulge_scaled, 'V_disk': V_disk_scaled, 'V_gas': V_gas,
                'R_d': R_d,
                'f_bulge_local': f_bulge_local,
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()

# Create point-by-point data
all_points = []
for gal in galaxies:
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    
    for i in range(len(R)):
        R_m = R[i] * kpc_to_m
        g_bar = (V_bar[i] * 1000)**2 / R_m
        V_ratio = V_obs[i] / V_bar[i]
        
        f_b = gal['f_bulge_local'][i]
        f_d = gal['V_disk'][i]**2 / max(V_bar[i]**2, 1e-10)
        f_g = np.abs(np.sign(gal['V_gas'][i]) * gal['V_gas'][i]**2) / max(V_bar[i]**2, 1e-10)
        
        all_points.append({
            'galaxy': gal['name'],
            'R': R[i],
            'V_obs': V_obs[i],
            'V_bar': V_bar[i],
            'V_ratio': V_ratio,
            'g_bar': g_bar,
            'g_norm': g_bar / g_dagger,
            'log_g_norm': np.log10(g_bar / g_dagger),
            'f_bulge': f_b,
            'f_disk': f_d,
            'f_gas': f_g,
            'R_d': gal['R_d'],
            'R_norm': R[i] / gal['R_d'],
        })

df = pd.DataFrame(all_points)
print(f"\nLoaded {len(df)} data points")

# =============================================================================
# THE KEY PUZZLE: WHY DO 3D REGIONS HAVE LOWER V_ratio?
# =============================================================================
print("\n" + "=" * 80)
print("THE KEY PUZZLE: 3D REGIONS HAVE LOWER V_ratio")
print("=" * 80)

print("""
OBSERVATION:
- 2D (disk/gas) regions: V_ratio = 1.88
- 3D (bulge) regions: V_ratio = 1.14

This seems backwards! Clusters (3D) need A = 8.0, but bulges (3D) need less?

POSSIBLE EXPLANATIONS:
1. Bulges have HIGH g → low h(g) → low enhancement (confounding)
2. Bulges are NOT like clusters (different physics)
3. The baryonic model for bulges is wrong (M/L issue)
4. Velocity dispersion in bulges affects the measurement
""")

# =============================================================================
# CONTROL FOR ACCELERATION
# =============================================================================
print("\n" + "=" * 80)
print("CONTROLLING FOR ACCELERATION")
print("=" * 80)

# Bin by log(g/g†) and compare 2D vs 3D within each bin
print("\nV_ratio by acceleration bin, split by geometry:")
print(f"\n  {'log(g/g†)':<12} {'2D V_ratio':<12} {'3D V_ratio':<12} {'Diff':<10} {'2D N':<8} {'3D N':<8}")
print("  " + "-" * 70)

g_bins = [(-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2)]

for g_min, g_max in g_bins:
    mask_g = (df['log_g_norm'] >= g_min) & (df['log_g_norm'] < g_max)
    mask_2d = df['f_bulge'] < 0.2
    mask_3d = df['f_bulge'] > 0.5
    
    subset_2d = df[mask_g & mask_2d]
    subset_3d = df[mask_g & mask_3d]
    
    if len(subset_2d) > 10 and len(subset_3d) > 5:
        v2d = subset_2d['V_ratio'].mean()
        v3d = subset_3d['V_ratio'].mean()
        diff = v2d - v3d
        print(f"  [{g_min:+.1f},{g_max:+.1f}]     {v2d:<12.3f} {v3d:<12.3f} {diff:+.3f}     {len(subset_2d):<8} {len(subset_3d):<8}")

# =============================================================================
# THE REAL PHYSICS: WHAT'S DIFFERENT ABOUT BULGES?
# =============================================================================
print("\n" + "=" * 80)
print("WHAT'S PHYSICALLY DIFFERENT ABOUT BULGE REGIONS?")
print("=" * 80)

bulge_points = df[df['f_bulge'] > 0.5]
disk_points = df[df['f_bulge'] < 0.2]

print(f"\nComparing bulge-dominated vs disk-dominated regions:")
print(f"\n  {'Property':<25} {'Bulge (f>0.5)':<15} {'Disk (f<0.2)':<15} {'Ratio':<10}")
print("  " + "-" * 70)

print(f"  {'Mean g/g†':<25} {bulge_points['g_norm'].mean():<15.2f} {disk_points['g_norm'].mean():<15.2f} {bulge_points['g_norm'].mean()/disk_points['g_norm'].mean():<10.1f}")
print(f"  {'Mean R/R_d':<25} {bulge_points['R_norm'].mean():<15.2f} {disk_points['R_norm'].mean():<15.2f} {bulge_points['R_norm'].mean()/disk_points['R_norm'].mean():<10.2f}")
print(f"  {'Mean V_bar (km/s)':<25} {bulge_points['V_bar'].mean():<15.0f} {disk_points['V_bar'].mean():<15.0f} {bulge_points['V_bar'].mean()/disk_points['V_bar'].mean():<10.2f}")
print(f"  {'Mean V_obs (km/s)':<25} {bulge_points['V_obs'].mean():<15.0f} {disk_points['V_obs'].mean():<15.0f} {bulge_points['V_obs'].mean()/disk_points['V_obs'].mean():<10.2f}")
print(f"  {'Mean V_ratio':<25} {bulge_points['V_ratio'].mean():<15.3f} {disk_points['V_ratio'].mean():<15.3f} {bulge_points['V_ratio'].mean()/disk_points['V_ratio'].mean():<10.2f}")

# =============================================================================
# HYPOTHESIS: BULGE M/L IS ALREADY CORRECT
# =============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS: BULGE BARYONIC MODEL IS ALREADY CORRECT")
print("=" * 80)

print("""
If V_obs ≈ V_bar in bulge regions (V_ratio ≈ 1.14), this suggests:
1. The baryonic mass in bulges is ALREADY sufficient to explain velocities
2. Little "dark matter" or enhancement is needed in bulge regions
3. This is OPPOSITE of disk regions where V_ratio ≈ 1.88

PHYSICAL INTERPRETATION:
- Bulges are dense, high-g regions where Newtonian gravity is nearly correct
- Disks are diffuse, low-g regions where enhancement is needed
- This is exactly what MOND and Σ-Gravity predict!

The "problem" is not that bulges need MORE enhancement,
but that our model OVER-enhances bulge regions.
""")

# Check: what's the required enhancement in each region?
print("\nRequired enhancement (Σ = V_ratio²) by region:")
print(f"  Bulge regions: Σ = {bulge_points['V_ratio'].mean()**2:.3f}")
print(f"  Disk regions: Σ = {disk_points['V_ratio'].mean()**2:.3f}")
print(f"  Ratio: {disk_points['V_ratio'].mean()**2 / bulge_points['V_ratio'].mean()**2:.2f}x more enhancement needed in disk")

# =============================================================================
# THE h(g) FUNCTION: DOES IT MATCH THE DATA?
# =============================================================================
print("\n" + "=" * 80)
print("COMPARING DATA TO h(g) PREDICTIONS")
print("=" * 80)

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

# The required h to explain the data
df['Sigma_required'] = df['V_ratio']**2
df['h_required'] = (df['Sigma_required'] - 1)  # Assuming A × W = 1

# The predicted h from our formula
df['h_predicted'] = h_function(df['g_bar'])

print("\nComparing required vs predicted h(g) by acceleration bin:")
print(f"\n  {'log(g/g†)':<12} {'h_required':<12} {'h_predicted':<12} {'Ratio':<10}")
print("  " + "-" * 50)

for g_min, g_max in g_bins:
    mask = (df['log_g_norm'] >= g_min) & (df['log_g_norm'] < g_max)
    subset = df[mask]
    if len(subset) > 50:
        h_req = subset['h_required'].mean()
        h_pred = subset['h_predicted'].mean()
        ratio = h_req / h_pred if h_pred > 0 else np.nan
        print(f"  [{g_min:+.1f},{g_max:+.1f}]     {h_req:<12.3f} {h_pred:<12.3f} {ratio:<10.2f}")

# =============================================================================
# GEOMETRY-DEPENDENT ENHANCEMENT
# =============================================================================
print("\n" + "=" * 80)
print("TESTING GEOMETRY-DEPENDENT ENHANCEMENT")
print("=" * 80)

print("""
Hypothesis: Enhancement depends on local geometry
  Σ = 1 + A(geometry) × h(g)
  
where A(geometry) varies from A_2D (disk) to A_3D (bulge)
""")

# For each point, what A would give the correct V_ratio?
# V_ratio² = 1 + A × h
# A = (V_ratio² - 1) / h

df['A_required'] = (df['V_ratio']**2 - 1) / np.maximum(df['h_predicted'], 0.01)

print("\nRequired A by geometry:")
print(f"\n  {'f_bulge bin':<15} {'Mean A_req':<12} {'Median A_req':<12} {'N':<8}")
print("  " + "-" * 50)

for f_min, f_max in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
    mask = (df['f_bulge'] >= f_min) & (df['f_bulge'] < f_max)
    subset = df[mask]
    if len(subset) > 20:
        # Filter out extreme values
        A_vals = subset['A_required'][(subset['A_required'] > 0) & (subset['A_required'] < 50)]
        if len(A_vals) > 10:
            print(f"  [{f_min:.1f}-{f_max:.1f}]       {A_vals.mean():<12.2f} {A_vals.median():<12.2f} {len(A_vals):<8}")

# =============================================================================
# THE TRANSITION REGION IN DETAIL
# =============================================================================
print("\n" + "=" * 80)
print("TRANSITION REGION DETAILED ANALYSIS")
print("=" * 80)

# Look at galaxies with significant bulges
bulge_galaxies = [g for g in galaxies if np.sum(g['f_bulge_local'] > 0.3) > 3]

print(f"\nAnalyzing {len(bulge_galaxies)} galaxies with significant bulges")
print("\nFor each galaxy, where is the transition and what happens there?")

transition_data = []
for gal in bulge_galaxies:
    R = gal['R']
    f_bulge = gal['f_bulge_local']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    
    # Find transition point (where f_bulge crosses 0.5)
    for i in range(1, len(R)):
        if f_bulge[i-1] > 0.5 and f_bulge[i] < 0.5:
            R_trans = R[i]
            
            # Get properties just inside and outside transition
            inside_mask = R < R_trans
            outside_mask = R > R_trans
            
            if inside_mask.sum() > 2 and outside_mask.sum() > 2:
                V_ratio_inside = np.mean(V_obs[inside_mask] / V_bar[inside_mask])
                V_ratio_outside = np.mean(V_obs[outside_mask] / V_bar[outside_mask])
                
                transition_data.append({
                    'galaxy': gal['name'],
                    'R_trans': R_trans,
                    'V_ratio_inside': V_ratio_inside,
                    'V_ratio_outside': V_ratio_outside,
                    'jump': V_ratio_outside - V_ratio_inside,
                })
            break

if transition_data:
    trans_df = pd.DataFrame(transition_data)
    print(f"\nFound {len(trans_df)} galaxies with clear bulge-disk transition")
    print(f"\n  Mean V_ratio inside bulge: {trans_df['V_ratio_inside'].mean():.3f}")
    print(f"  Mean V_ratio outside bulge: {trans_df['V_ratio_outside'].mean():.3f}")
    print(f"  Mean jump at transition: {trans_df['jump'].mean():+.3f}")
    
    print(f"\nGalaxies with largest jumps:")
    print(f"\n  {'Galaxy':<20} {'R_trans':<10} {'Inside':<10} {'Outside':<10} {'Jump':<10}")
    print("  " + "-" * 65)
    for _, row in trans_df.nlargest(10, 'jump').iterrows():
        print(f"  {row['galaxy']:<20} {row['R_trans']:<10.1f} {row['V_ratio_inside']:<10.3f} {row['V_ratio_outside']:<10.3f} {row['jump']:+.3f}")

# =============================================================================
# SUMMARY: THE PHYSICS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: THE PHYSICS OF GEOMETRY")
print("=" * 80)

print(f"""
KEY FINDINGS:

1. BULGE REGIONS NEED LESS ENHANCEMENT:
   - V_ratio in bulge: {bulge_points['V_ratio'].mean():.2f} → Σ = {bulge_points['V_ratio'].mean()**2:.2f}
   - V_ratio in disk: {disk_points['V_ratio'].mean():.2f} → Σ = {disk_points['V_ratio'].mean()**2:.2f}
   - Disk needs {disk_points['V_ratio'].mean()**2 / bulge_points['V_ratio'].mean()**2:.1f}x more enhancement

2. THIS IS CONSISTENT WITH HIGH g IN BULGES:
   - Bulge: g/g† = {bulge_points['g_norm'].mean():.1f}
   - Disk: g/g† = {disk_points['g_norm'].mean():.2f}
   - h(g) naturally gives less enhancement at high g

3. BUT THERE'S A GEOMETRY EFFECT TOO:
   - After controlling for g, 3D regions still have lower V_ratio
   - This suggests 3D geometry genuinely needs less enhancement
   
4. PHYSICAL INTERPRETATION:
   - In 3D (bulge): gravity is more "Newtonian" (less enhancement needed)
   - In 2D (disk): gravity needs more enhancement
   - This is OPPOSITE of what we assumed for clusters!

5. IMPLICATIONS FOR CLUSTERS:
   - Clusters are 3D, but they DO need enhancement (A = 8)
   - The difference: clusters are at r >> ξ (outer regions)
   - Bulges are at r ~ ξ (inner regions)
   - Maybe it's not 2D vs 3D, but INNER vs OUTER

6. NEW HYPOTHESIS:
   Enhancement depends on:
   - g/g† (acceleration regime) → h(g)
   - R/ξ (radial position) → W(r)
   - NOT directly on 2D vs 3D geometry
   
   The geometry effect is INDIRECT:
   - 3D bulges are in inner, high-g regions
   - 2D disks are in outer, low-g regions
   - The correlation with geometry is a proxy for position/acceleration
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)



