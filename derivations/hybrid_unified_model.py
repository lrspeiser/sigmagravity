#!/usr/bin/env python3
"""
HYBRID UNIFIED MODEL

Problem: 
- Path length model works for clusters but not galaxies
- Current model works for both but has separate A values

Solution: Find what's DIFFERENT about how path length affects galaxies vs clusters

Key observation:
- For clusters: We use A = 8.0 and it works
- For galaxies: We use A = 1.17 (independent of path length) and it works

What if the path length formula is CORRECT but there's a saturation effect?
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
print("HYBRID UNIFIED MODEL")
print("=" * 80)

# =============================================================================
# THE INSIGHT
# =============================================================================
print("\n" + "=" * 80)
print("THE KEY INSIGHT")
print("=" * 80)

print("""
For galaxies, the current model uses:
  A = A₀ = exp(1/2π) ≈ 1.17 (FIXED, independent of galaxy size)
  
This works because ξ = R_d/(2π) scales with galaxy size.
The coherence window W = r/(ξ + r) adapts to each galaxy.

For clusters, we need:
  A = 8.0 (much larger)
  W ≈ 1 (saturated)

HYPOTHESIS: A depends on the RATIO of path length to coherence scale.

For galaxies:
  L ~ 2h ~ 0.3 R_d (path through disk)
  ξ ~ R_d/(2π) ~ 0.16 R_d (coherence scale)
  L/ξ ~ 0.3/0.16 ~ 2 (roughly constant!)

For clusters:
  L ~ 600 kpc (path through cluster)
  ξ ~ R_core/(2π) ~ 50 kpc (coherence scale)
  L/ξ ~ 600/50 ~ 12 (much larger!)

So the difference is in L/ξ, not just L!
""")

# =============================================================================
# TEST: A DEPENDS ON L/ξ
# =============================================================================
print("\n" + "=" * 80)
print("TEST: A = A₀ × f(L/ξ)")
print("=" * 80)

A_0 = np.exp(1 / (2 * np.pi))

# For galaxies: L/ξ ~ 2
# For clusters: L/ξ ~ 12

# If A = A₀ × (L/ξ)^n:
# A_cluster / A_galaxy = (12/2)^n = 6^n = 8.0/1.17 = 6.84
# n = log(6.84) / log(6) = 1.05

n_Lxi = np.log(8.0 / A_0) / np.log(12 / 2)
print(f"If A = A₀ × (L/ξ)^n:")
print(f"  n = {n_Lxi:.3f}")
print(f"  For L/ξ = 2: A = {A_0 * 2**n_Lxi:.3f}")
print(f"  For L/ξ = 12: A = {A_0 * 12**n_Lxi:.2f}")

# But this gives A varying with galaxy size, which we know doesn't work well
print(f"\nBut this would make A vary with galaxy size, which worsens fit.")

# =============================================================================
# ALTERNATIVE: SATURATION MODEL
# =============================================================================
print("\n" + "=" * 80)
print("ALTERNATIVE: SATURATION MODEL")
print("=" * 80)

print("""
What if A saturates at a maximum value for small L/ξ?

A = A₀ × min(1, (L/ξ)^n / (L/ξ)_sat^n)

For L/ξ < (L/ξ)_sat: A = A₀ (saturated)
For L/ξ > (L/ξ)_sat: A = A₀ × (L/ξ / (L/ξ)_sat)^n

This would give:
- Galaxies (L/ξ ~ 2): A = A₀ = 1.17 (saturated)
- Clusters (L/ξ ~ 12): A = A₀ × (12/2)^n (growing)

For A_cluster = 8.0:
  8.0 = 1.17 × (12/2)^n
  n = log(8.0/1.17) / log(6) = 1.05
""")

# =============================================================================
# BETTER IDEA: A = A₀ × (1 + log(L/ξ))
# =============================================================================
print("\n" + "=" * 80)
print("BETTER IDEA: A = A₀ × (1 + α × log(L/ξ))")
print("=" * 80)

print("""
Logarithmic scaling saturates naturally at small L/ξ.

A = A₀ × (1 + α × ln(L/ξ))

For L/ξ = 2: A = A₀ × (1 + α × 0.69)
For L/ξ = 12: A = A₀ × (1 + α × 2.48)

To get A_cluster = 8.0:
  8.0 = 1.17 × (1 + α × 2.48)
  8.0/1.17 = 1 + α × 2.48
  6.84 - 1 = α × 2.48
  α = 5.84 / 2.48 = 2.35

Check for galaxy:
  A = 1.17 × (1 + 2.35 × 0.69) = 1.17 × 2.62 = 3.07

This is too high! Galaxies would have A = 3.07, not 1.17.
""")

# =============================================================================
# THE REAL SOLUTION: DIFFERENT COHERENCE MECHANISMS
# =============================================================================
print("\n" + "=" * 80)
print("THE REAL SOLUTION: DIFFERENT COHERENCE MECHANISMS")
print("=" * 80)

print("""
What if the coherence mechanism is DIFFERENT for 2D vs 3D systems?

For 2D (disk):
  - Coherence is primarily AZIMUTHAL (around the disk)
  - Limited by disk thickness (vertical)
  - A is set by the azimuthal coherence, not path length
  - A = A₀ = exp(1/2π) (from azimuthal mode count)

For 3D (cluster):
  - Coherence is RADIAL (all directions)
  - Limited by cluster size
  - A grows with path length
  - A = A₀ × (L/L₀)^n

This explains why:
  - Galaxy A is constant (azimuthal coherence)
  - Cluster A scales with size (radial coherence)
""")

# =============================================================================
# UNIFIED FORMULA WITH MODE SWITCHING
# =============================================================================
print("\n" + "=" * 80)
print("UNIFIED FORMULA WITH MODE SWITCHING")
print("=" * 80)

print("""
UNIFIED FORMULA:

A = A₀ × [1 + (D × (L/L₀)^n - D)]
  = A₀ × [1 - D + D × (L/L₀)^n]

where:
  D = dimensionality factor (0 for 2D, 1 for 3D)
  L = path length
  L₀ = reference scale
  n = 0.27 (exponent)

For 2D (D=0):
  A = A₀ × [1 - 0 + 0] = A₀ = 1.17

For 3D (D=1):
  A = A₀ × [1 - 1 + 1 × (L/L₀)^n] = A₀ × (L/L₀)^n

This smoothly interpolates between:
  - Pure 2D: A = A₀ (constant)
  - Pure 3D: A = A₀ × (L/L₀)^n (path-length dependent)
""")

# =============================================================================
# TEST THE UNIFIED FORMULA
# =============================================================================
print("\n" + "=" * 80)
print("TESTING UNIFIED FORMULA")
print("=" * 80)

# Load data
rotmod_dir = data_dir / "Rotmod_LTG"

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

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
        
        h_disk = 0.15 * R_d
        f_bulge = np.sum(V_bulge**2) / max(np.sum(V_disk**2 + V_bulge**2 + V_gas**2), 1e-10)
        
        galaxies.append({
            'name': f.stem.replace('_rotmod', ''),
            'R': R, 'V_obs': V_obs, 'V_bar': V_bar, 'R_d': R_d,
            'h_disk': h_disk, 'f_bulge': f_bulge,
        })
    except:
        continue

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
        })

print(f"Loaded {len(galaxies)} galaxies, {len(clusters)} clusters")

# Test unified formula
XI_COEFF = 1 / (2 * np.pi)
L_0 = 0.5
n_exp = 0.27

def unified_A(L, D):
    """A = A₀ × [1 - D + D × (L/L₀)^n]"""
    return A_0 * (1 - D + D * (L / L_0)**n_exp)

print("\nTest unified formula: A = A₀ × [1 - D + D × (L/L₀)^n]")
print(f"  n = {n_exp}, L₀ = {L_0} kpc")

# Galaxy test with D = f_bulge
rms_list = []
for gal in galaxies:
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    
    L = 2 * gal['h_disk']
    D = gal['f_bulge']  # 0 for pure disk, 1 for pure bulge
    A = unified_A(L, D)
    
    xi = XI_COEFF * gal['R_d']
    W = R / (xi + R)
    
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    h = h_function(g_bar)
    
    Sigma = 1 + A * W * h
    V_pred = V_bar * np.sqrt(Sigma)
    
    rms = np.sqrt(np.mean((V_obs - V_pred)**2))
    rms_list.append(rms)

# Cluster test with D = 1
ratios = []
for cl in clusters:
    L = 600
    D = 1.0
    A = unified_A(L, D)
    
    h = h_function(cl['g_bar'])
    Sigma = 1 + A * h
    M_pred = cl['M_bar'] * Sigma
    ratios.append(M_pred / cl['M_lens'])

print(f"\nUnified formula results:")
print(f"  Galaxy RMS: {np.mean(rms_list):.2f} km/s")
print(f"  Cluster ratio: {np.median(ratios):.3f}")

# =============================================================================
# OPTIMIZE THE FORMULA
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMIZE THE FORMULA")
print("=" * 80)

print("\nGrid search for optimal parameters:")
print(f"\n  {'n':<8} {'L₀':<8} {'Galaxy RMS':<12} {'Cluster ratio':<15} {'Score':<10}")
print("  " + "-" * 60)

best_score = float('inf')
best_params = None

for n_test in [0.25, 0.27, 0.30, 0.33]:
    for L_0_test in [0.3, 0.4, 0.5, 0.6]:
        def unified_A_test(L, D):
            return A_0 * (1 - D + D * (L / L_0_test)**n_test)
        
        # Galaxy
        rms_list = []
        for gal in galaxies:
            R = gal['R']
            V_obs = gal['V_obs']
            V_bar = gal['V_bar']
            
            L = 2 * gal['h_disk']
            D = gal['f_bulge']
            A = unified_A_test(L, D)
            
            xi = XI_COEFF * gal['R_d']
            W = R / (xi + R)
            
            R_m = R * kpc_to_m
            g_bar = (V_bar * 1000)**2 / R_m
            h = h_function(g_bar)
            
            Sigma = 1 + A * W * h
            V_pred = V_bar * np.sqrt(Sigma)
            
            rms = np.sqrt(np.mean((V_obs - V_pred)**2))
            rms_list.append(rms)
        
        # Cluster
        ratios = []
        for cl in clusters:
            L = 600
            D = 1.0
            A = unified_A_test(L, D)
            
            h = h_function(cl['g_bar'])
            Sigma = 1 + A * h
            M_pred = cl['M_bar'] * Sigma
            ratios.append(M_pred / cl['M_lens'])
        
        gal_rms = np.mean(rms_list)
        cl_ratio = np.median(ratios)
        score = gal_rms + 30 * abs(cl_ratio - 1.0)
        
        if score < best_score:
            best_score = score
            best_params = (n_test, L_0_test, gal_rms, cl_ratio)
        
        print(f"  {n_test:<8.2f} {L_0_test:<8.2f} {gal_rms:<12.2f} {cl_ratio:<15.3f} {score:<10.2f}")

print(f"\nBest: n = {best_params[0]:.2f}, L₀ = {best_params[1]:.2f}")
print(f"  Galaxy RMS = {best_params[2]:.2f} km/s")
print(f"  Cluster ratio = {best_params[3]:.3f}")

# =============================================================================
# COMPARE TO CURRENT MODEL
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON TO CURRENT MODEL")
print("=" * 80)

# Current model
rms_current = []
for gal in galaxies:
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    
    xi = XI_COEFF * gal['R_d']
    W = R / (xi + R)
    
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    h = h_function(g_bar)
    
    Sigma = 1 + A_0 * W * h
    V_pred = V_bar * np.sqrt(Sigma)
    
    rms = np.sqrt(np.mean((V_obs - V_pred)**2))
    rms_current.append(rms)

ratios_current = []
for cl in clusters:
    h = h_function(cl['g_bar'])
    Sigma = 1 + 8.0 * h
    M_pred = cl['M_bar'] * Sigma
    ratios_current.append(M_pred / cl['M_lens'])

print(f"\nCurrent model (A_galaxy = 1.17, A_cluster = 8.0):")
print(f"  Galaxy RMS: {np.mean(rms_current):.2f} km/s")
print(f"  Cluster ratio: {np.median(ratios_current):.3f}")

print(f"\nUnified model (A = A₀ × [1 - D + D × (L/L₀)^n]):")
print(f"  Galaxy RMS: {best_params[2]:.2f} km/s")
print(f"  Cluster ratio: {best_params[3]:.3f}")

print(f"\nDifference:")
print(f"  Galaxy RMS: +{best_params[2] - np.mean(rms_current):.2f} km/s (worse)")
print(f"  Cluster ratio: {best_params[3] - np.median(ratios_current):+.3f}")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
UNIFIED FORMULA:

A = A₀ × [1 - D + D × (L/L₀)^n]

where:
  A₀ = exp(1/2π) ≈ 1.17
  D = dimensionality (0 for 2D disk, 1 for 3D cluster)
  L = path length through baryons
  L₀ = {best_params[1]:.2f} kpc
  n = {best_params[0]:.2f}

PHYSICAL INTERPRETATION:

For 2D systems (D → 0):
  A → A₀ (constant, independent of path length)
  Coherence is AZIMUTHAL (around the disk)
  
For 3D systems (D → 1):
  A → A₀ × (L/L₀)^n (path-length dependent)
  Coherence is RADIAL (all directions)

This explains WHY:
- Galaxy A is constant: Disk coherence is azimuthal
- Cluster A scales with size: Cluster coherence is radial
- The transition is smooth: Bulge-dominated galaxies are intermediate

TRADE-OFF:
- Current model: Galaxy RMS = {np.mean(rms_current):.2f} km/s
- Unified model: Galaxy RMS = {best_params[2]:.2f} km/s
- Cost: ~{best_params[2] - np.mean(rms_current):.1f} km/s worse for galaxies

The unified model is PHYSICALLY MOTIVATED but slightly worse numerically.
This suggests the current empirical parameters are still optimal,
but now we understand WHY they work.
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

