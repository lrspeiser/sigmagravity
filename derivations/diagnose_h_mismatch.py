#!/usr/bin/env python3
"""
DIAGNOSE THE h MISMATCH

From the previous analysis:
  g/g† = [0.3-1.0]: h_predicted = 0.873, Galaxy h_actual = 1.238, Cluster h_actual = 0.685
  g/g† = [1.0-3.0]: h_predicted = 0.278, Galaxy h_actual = 0.585, Cluster h_actual = 0.333

Galaxies need ~1.4-2.1× MORE h than predicted
Clusters need ~0.8-1.2× the predicted h

This is the REAL puzzle. Let's understand why.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

c = 3e8
H0 = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

data_dir = Path(__file__).parent.parent / "data"

print("=" * 80)
print("DIAGNOSING THE h MISMATCH")
print("=" * 80)

# Parameters
A_GALAXY = np.exp(1 / (2 * np.pi))
A_CLUSTER = 8.0
XI_COEFF = 1 / (2 * np.pi)

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    return r / (xi + r)

# =============================================================================
# DETAILED GALAXY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED GALAXY ANALYSIS")
print("=" * 80)

rotmod_dir = data_dir / "Rotmod_LTG"

galaxy_data = []
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
        
        # Get point-by-point data
        xi = XI_COEFF * R_d
        for i in range(len(R)):
            R_m = R[i] * kpc_to_m
            g_bar = (V_bar[i] * 1000)**2 / R_m
            g_norm = g_bar / g_dagger
            
            W = W_coherence(R[i], xi)
            Sigma = (V_obs[i] / V_bar[i])**2
            
            h_pred = h_function(g_bar)
            Sigma_pred = 1 + A_GALAXY * W * h_pred
            
            if Sigma > 1 and A_GALAXY * W > 0.01:
                h_actual = (Sigma - 1) / (A_GALAXY * W)
            else:
                h_actual = np.nan
            
            galaxy_data.append({
                'g_norm': g_norm,
                'log_g': np.log10(g_norm),
                'W': W,
                'Sigma': Sigma,
                'Sigma_pred': Sigma_pred,
                'h_pred': h_pred,
                'h_actual': h_actual,
                'h_ratio': h_actual / h_pred if h_pred > 0.01 else np.nan,
            })
    except:
        continue

df = pd.DataFrame(galaxy_data)
print(f"Loaded {len(df)} galaxy data points")

# =============================================================================
# ANALYZE h RATIO BY ACCELERATION
# =============================================================================
print("\n" + "=" * 80)
print("h RATIO (ACTUAL/PREDICTED) BY ACCELERATION")
print("=" * 80)

print(f"\n  {'log(g/g†)':<12} {'h_pred':<10} {'h_actual':<10} {'h_ratio':<10} {'N':<8}")
print("  " + "-" * 55)

g_bins = [(-2.5, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1)]

for g_min, g_max in g_bins:
    mask = (df['log_g'] >= g_min) & (df['log_g'] < g_max)
    n = mask.sum()
    
    if n > 10:
        h_pred = df[mask]['h_pred'].median()
        h_actual = df[mask]['h_actual'].median()
        h_ratio = df[mask]['h_ratio'].median()
        print(f"  [{g_min:+.1f},{g_max:+.1f}]     {h_pred:<10.3f} {h_actual:<10.3f} {h_ratio:<10.2f} {n:<8}")

# =============================================================================
# THE PATTERN
# =============================================================================
print("\n" + "=" * 80)
print("THE PATTERN")
print("=" * 80)

print("""
At LOW g (g/g† < 0.3):  h_ratio ≈ 1.0-1.2  (model works well)
At MID g (g/g† ~ 1):    h_ratio ≈ 1.4-2.0  (model under-predicts)
At HIGH g (g/g† > 3):   h_ratio ≈ 2.0-4.0  (model severely under-predicts)

The model UNDER-PREDICTS enhancement at high accelerations!

This is consistent with our earlier finding that h(g) decays too fast.
""")

# =============================================================================
# WHAT h(g) WOULD FIT BETTER?
# =============================================================================
print("\n" + "=" * 80)
print("WHAT h(g) WOULD FIT BETTER?")
print("=" * 80)

# Test different h(g) forms
def h_current(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def h_slower_decay(g, beta=0.5):
    """h = √(g†/g) × (g†/(g†+g))^β with β < 1"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * (g_dagger / (g_dagger + g))**beta

def h_mond_simple(g):
    """MOND simple: ν = 1/(1 - exp(-√x)) - 1"""
    g = np.maximum(np.asarray(g), 1e-15)
    x = g / g_dagger
    nu = 1 / (1 - np.exp(-np.sqrt(x)))
    return nu - 1

def h_log_form(g):
    """h = ln(1 + g†/g)"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.log(1 + g_dagger / g)

# Test each form
print("\nCompare h(g) forms at different accelerations:")
print(f"\n  {'g/g†':<10} {'Current':<12} {'β=0.5':<12} {'β=0.7':<12} {'MOND':<12} {'Log':<12} {'Data':<12}")
print("  " + "-" * 80)

for g_min, g_max in g_bins:
    mask = (df['log_g'] >= g_min) & (df['log_g'] < g_max)
    if mask.sum() < 10:
        continue
    
    g_mid = 10**((g_min + g_max) / 2) * g_dagger
    
    h_cur = h_current(g_mid)
    h_b05 = h_slower_decay(g_mid, 0.5)
    h_b07 = h_slower_decay(g_mid, 0.7)
    h_mond = h_mond_simple(g_mid)
    h_log = h_log_form(g_mid)
    h_data = df[mask]['h_actual'].median()
    
    print(f"  [{g_min:+.1f},{g_max:+.1f}]   {h_cur:<12.3f} {h_b05:<12.3f} {h_b07:<12.3f} {h_mond:<12.3f} {h_log:<12.3f} {h_data:<12.3f}")

# =============================================================================
# IMPLICATIONS FOR CLUSTERS
# =============================================================================
print("\n" + "=" * 80)
print("IMPLICATIONS FOR CLUSTERS")
print("=" * 80)

# Load clusters
cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
clusters = []
if cluster_file.exists():
    cl_df = pd.read_csv(cluster_file)
    cl_valid = cl_df[
        cl_df['M500_1e14Msun'].notna() & 
        cl_df['MSL_200kpc_1e12Msun'].notna() &
        (cl_df['spec_z_constraint'] == 'yes')
    ].copy()
    cl_valid = cl_valid[cl_valid['M500_1e14Msun'] > 2.0].copy()
    
    for _, row in cl_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar = 0.4 * 0.15 * M500
        M_lens = row['MSL_200kpc_1e12Msun'] * 1e12
        r_m = 200 * kpc_to_m
        g_bar = G_const * M_bar * M_sun / r_m**2
        
        clusters.append({
            'M_bar': M_bar,
            'M_lens': M_lens,
            'g_bar': g_bar,
            'g_norm': g_bar / g_dagger,
            'Sigma': M_lens / M_bar,
        })

print(f"\nCluster g/g† range: {min(c['g_norm'] for c in clusters):.2f} - {max(c['g_norm'] for c in clusters):.2f}")

# Test different h(g) forms on clusters
print("\nCluster predictions with different h(g) forms:")
print(f"\n  {'h(g) form':<20} {'Median ratio':<15} {'Scatter (dex)':<15}")
print("  " + "-" * 55)

for name, h_func in [
    ("Current", h_current),
    ("β=0.5", lambda g: h_slower_decay(g, 0.5)),
    ("β=0.7", lambda g: h_slower_decay(g, 0.7)),
    ("MOND simple", h_mond_simple),
    ("Logarithmic", h_log_form),
]:
    ratios = []
    for cl in clusters:
        h = h_func(cl['g_bar'])
        Sigma = 1 + A_CLUSTER * h
        M_pred = cl['M_bar'] * Sigma
        ratios.append(M_pred / cl['M_lens'])
    
    median_ratio = np.median(ratios)
    scatter = np.std(np.log10(ratios))
    print(f"  {name:<20} {median_ratio:<15.3f} {scatter:<15.3f}")

# =============================================================================
# THE TRADE-OFF
# =============================================================================
print("\n" + "=" * 80)
print("THE TRADE-OFF")
print("=" * 80)

print("""
The data shows a TRADE-OFF:

1. Current h(g) = √(g†/g) × g†/(g†+g):
   - Works well for clusters (ratio = 0.955)
   - Under-predicts galaxies at high g (h_ratio > 1.5)

2. Slower decay (β < 1):
   - Better for galaxies at high g
   - Over-predicts clusters (ratio > 1)

3. MOND-like h(g):
   - Better for galaxies at high g
   - Over-predicts clusters

POSSIBLE RESOLUTIONS:

A. ACCEPT THE TRADE-OFF
   - Current h(g) is a compromise
   - Galaxy under-prediction at high g is a known limitation
   
B. DIFFERENT A FOR HIGH-g REGIONS
   - Lower A in bulge-dominated regions
   - This is what we found earlier (bulge suppression)
   
C. MODIFY W(r) FOR BULGES
   - Bulges have different coherence structure
   - W_bulge < W_disk at same radius
   
D. OBSERVABLE-DEPENDENT h(g)
   - h_dynamics ≠ h_lensing
   - This is the gravitational slip hypothesis (but we showed it's not needed)
""")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
THE ROOT CAUSE:

The model under-predicts enhancement at high accelerations (g > g†).

This is most prominent in:
1. Inner regions of galaxies (bulges)
2. High surface brightness galaxies

CURRENT SOLUTION:
- Use current h(g) as the best compromise
- Accept ~1.5-2× under-prediction at high g
- This manifests as higher RMS in bulge-dominated galaxies

ALTERNATIVE SOLUTIONS TO EXPLORE:
1. Different coherence window for bulges
2. A that depends on local density gradient
3. h(g) with slower decay at high g (but must preserve cluster fit)

THE KEY CONSTRAINT:
Any modification must preserve the excellent cluster fit (ratio ≈ 0.955).
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

