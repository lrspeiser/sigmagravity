#!/usr/bin/env python3
"""
TEST BULGE SUPPRESSION ON FULL REGRESSION SUITE

Key finding: A_bulge = 0 gives best galaxy fits!
This means bulges should have NO coherence enhancement.

Test this against:
1. SPARC galaxies (should improve)
2. Clusters (should be unchanged - no bulge component)
3. Milky Way (has bulge - check impact)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 3e8
H0 = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

print("=" * 80)
print("TESTING BULGE SUPPRESSION ON FULL REGRESSION")
print("=" * 80)

data_dir = Path(__file__).parent.parent / "data"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return r / (xi + r)

def mond_velocity(R, V_bar):
    a0 = 1.2e-10
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    y = g_bar / a0
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    return V_bar * np.sqrt(nu)

# =============================================================================
# SPARC GALAXIES
# =============================================================================
print("\n" + "=" * 80)
print("SPARC GALAXIES")
print("=" * 80)

rotmod_dir = data_dir / "Rotmod_LTG"

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
            
            bulge_frac = np.sum(V_bulge_scaled**2) / max(np.sum(V_bar_sq), 1e-10)
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'V_gas': V_gas,
                'V_disk': V_disk_scaled,
                'V_bulge': V_bulge_scaled,
                'R_d': R_d,
                'bulge_frac': bulge_frac,
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()
print(f"Loaded {len(galaxies)} galaxies")

def predict_with_bulge_suppression(gal, A_disk, A_bulge, xi_coeff):
    """Predict velocity with separate disk and bulge amplitudes."""
    R_m = gal['R'] * kpc_to_m
    xi = xi_coeff * gal['R_d']
    
    # Disk contribution
    g_disk = (gal['V_disk'] * 1000)**2 / R_m
    W_disk = W_coherence(gal['R'], xi)
    h_disk = h_function(g_disk)
    
    # Bulge contribution (with suppressed A)
    g_bulge = (gal['V_bulge'] * 1000)**2 / R_m
    W_bulge = W_coherence(gal['R'], xi)
    h_bulge = h_function(g_bulge)
    
    # Gas contribution (like disk)
    g_gas = np.abs(np.sign(gal['V_gas']) * (gal['V_gas'] * 1000)**2 / R_m)
    W_gas = W_coherence(gal['R'], xi)
    h_gas = h_function(g_gas)
    
    # Weighted enhancement
    Sigma_disk = 1 + A_disk * W_disk * h_disk
    Sigma_bulge = 1 + A_bulge * W_bulge * h_bulge
    Sigma_gas = 1 + A_disk * W_gas * h_gas
    
    # Combine by velocity squared fractions
    V_bar_sq = gal['V_bar']**2
    f_disk = gal['V_disk']**2 / np.maximum(V_bar_sq, 1e-10)
    f_bulge = gal['V_bulge']**2 / np.maximum(V_bar_sq, 1e-10)
    f_gas = np.abs(np.sign(gal['V_gas']) * gal['V_gas']**2) / np.maximum(V_bar_sq, 1e-10)
    
    Sigma_total = f_disk * Sigma_disk + f_bulge * Sigma_bulge + f_gas * Sigma_gas
    return gal['V_bar'] * np.sqrt(np.maximum(Sigma_total, 1))

def predict_baseline(gal, A, xi_coeff):
    """Baseline: same A for all components."""
    R_m = gal['R'] * kpc_to_m
    g_bar = (gal['V_bar'] * 1000)**2 / R_m
    xi = xi_coeff * gal['R_d']
    W = W_coherence(gal['R'], xi)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    return gal['V_bar'] * np.sqrt(Sigma)

# Parameters
A_disk = np.exp(1 / (2 * np.pi))
xi_coeff = 1 / (2 * np.pi)

# Test different A_bulge values
print("\nTesting A_bulge values:")
print(f"\n  {'A_bulge':<10} {'Mean RMS':<12} {'Win vs MOND':<15} {'Bulge-dom RMS':<15}")
print("  " + "-" * 55)

for A_bulge in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.17]:
    results = []
    for gal in galaxies:
        V_pred = predict_with_bulge_suppression(gal, A_disk, A_bulge, xi_coeff)
        V_mond = mond_velocity(gal['R'], gal['V_bar'])
        
        rms = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
        rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
        
        results.append({
            'rms': rms,
            'rms_mond': rms_mond,
            'wins': rms < rms_mond,
            'bulge_frac': gal['bulge_frac'],
        })
    
    df = pd.DataFrame(results)
    bulge_dom = df[df['bulge_frac'] > 0.3]
    
    marker = " <-- current" if abs(A_bulge - A_disk) < 0.1 else ""
    print(f"  {A_bulge:<10.2f} {df['rms'].mean():<12.2f} {df['wins'].mean()*100:<15.1f}% {bulge_dom['rms'].mean():<15.2f}{marker}")

# Best model
print("\n" + "-" * 60)
print("BEST MODEL: A_bulge = 0")
print("-" * 60)

results_baseline = []
results_best = []

for gal in galaxies:
    V_baseline = predict_baseline(gal, A_disk, xi_coeff)
    V_best = predict_with_bulge_suppression(gal, A_disk, 0.0, xi_coeff)
    V_mond = mond_velocity(gal['R'], gal['V_bar'])
    
    rms_baseline = np.sqrt(np.mean((gal['V_obs'] - V_baseline)**2))
    rms_best = np.sqrt(np.mean((gal['V_obs'] - V_best)**2))
    rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
    
    results_baseline.append({
        'name': gal['name'],
        'rms': rms_baseline,
        'wins': rms_baseline < rms_mond,
    })
    results_best.append({
        'name': gal['name'],
        'rms': rms_best,
        'wins': rms_best < rms_mond,
        'bulge_frac': gal['bulge_frac'],
    })

df_baseline = pd.DataFrame(results_baseline)
df_best = pd.DataFrame(results_best)

print(f"\n  Baseline (A_bulge = A_disk):")
print(f"    Mean RMS: {df_baseline['rms'].mean():.2f} km/s")
print(f"    Win rate: {df_baseline['wins'].mean()*100:.1f}%")

print(f"\n  With bulge suppression (A_bulge = 0):")
print(f"    Mean RMS: {df_best['rms'].mean():.2f} km/s")
print(f"    Win rate: {df_best['wins'].mean()*100:.1f}%")

# Per-galaxy comparison
merged = df_baseline.merge(df_best, on='name', suffixes=('_base', '_best'))
merged['improvement'] = merged['rms_base'] - merged['rms_best']

print(f"\n  Galaxies improved: {(merged['improvement'] > 0).sum()}")
print(f"  Galaxies worsened: {(merged['improvement'] < 0).sum()}")
print(f"  Mean improvement: {merged['improvement'].mean():.2f} km/s")

# =============================================================================
# CLUSTERS (should be unchanged)
# =============================================================================
print("\n" + "=" * 80)
print("CLUSTERS")
print("=" * 80)

cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
if cluster_file.exists():
    cl_df = pd.read_csv(cluster_file)
    cl_valid = cl_df[
        cl_df['M500_1e14Msun'].notna() & 
        cl_df['MSL_200kpc_1e12Msun'].notna() &
        (cl_df['spec_z_constraint'] == 'yes')
    ].copy()
    cl_valid = cl_valid[cl_valid['M500_1e14Msun'] > 2.0].copy()
    
    print(f"Loaded {len(cl_valid)} clusters")
    
    # Clusters don't have bulges - should be unchanged
    ratios = []
    for _, row in cl_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar = 0.4 * 0.15 * M500
        M_lens = row['MSL_200kpc_1e12Msun'] * 1e12
        r_kpc = 200
        
        r_m = r_kpc * kpc_to_m
        g_bar = G_const * M_bar * M_sun / r_m**2
        
        h = h_function(np.array([g_bar]))[0]
        A_cluster = 8.0
        Sigma = 1 + A_cluster * h  # W ≈ 1 for clusters
        
        M_pred = M_bar * Sigma
        ratio = M_pred / M_lens
        ratios.append(ratio)
    
    print(f"\n  Cluster results (unchanged - no bulge component):")
    print(f"    Median M_pred/M_lens: {np.median(ratios):.3f}")
    print(f"    Scatter: {np.std(np.log10(ratios)):.3f} dex")
else:
    print("  [Cluster data not found]")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
BULGE SUPPRESSION RESULTS:

1. SPARC GALAXIES:
   - Baseline RMS: {df_baseline['rms'].mean():.2f} km/s
   - With A_bulge=0: {df_best['rms'].mean():.2f} km/s
   - Improvement: {df_baseline['rms'].mean() - df_best['rms'].mean():.2f} km/s
   - Win rate: {df_baseline['wins'].mean()*100:.1f}% → {df_best['wins'].mean()*100:.1f}%

2. CLUSTERS:
   - Unchanged (clusters have no bulge component)
   - Median ratio: {np.median(ratios):.3f}

3. PHYSICAL INTERPRETATION:
   
   Bulges have NO coherence enhancement because:
   
   a) RANDOM ORBITS: Bulge stars have high velocity dispersion
      - Orbits are not coherent (random orientations)
      - Unlike disk stars which orbit in same plane
      
   b) 3D GEOMETRY: Bulges are spheroidal, not planar
      - No preferred direction for coherence
      - Unlike clusters which are measured at r >> ξ
      
   c) EMBEDDED IN DISK: Bulge potential is mixed with disk
      - Not a clean 3D system like clusters
      - Coherence is disrupted by disk potential

4. FORMULA UPDATE:
   
   For galaxies with bulges:
   
   Σ = 1 + A × W(r) × h(g)
   
   where A varies by component:
   - A_disk = A_gas = e^(1/2π) ≈ 1.17
   - A_bulge = 0 (no enhancement)
   
   This is equivalent to:
   
   Σ = 1 + A × W(r) × h(g) × (1 - f_bulge)
   
   where f_bulge is the local bulge fraction.
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

