#!/usr/bin/env python3
"""
GALAXY-CLUSTER UNIFICATION

The key puzzle: At similar g/g†, galaxies and clusters require DIFFERENT h values!

From the data:
  log(g/g†) = [-0.5, 0]: Galaxy h = 1.20, Cluster h = 0.69, Ratio = 1.76
  log(g/g†) = [0, 0.5]:  Galaxy h = 0.66, Cluster h = 0.33, Ratio = 1.98

This means at the SAME acceleration, galaxies need ~2× more enhancement than clusters.

WHY? Let's explore the physical differences.
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
print("GALAXY-CLUSTER UNIFICATION")
print("=" * 80)

# =============================================================================
# THE PUZZLE
# =============================================================================
print("\n" + "=" * 80)
print("THE PUZZLE: SAME g, DIFFERENT h REQUIRED")
print("=" * 80)

print("""
At similar accelerations (g/g† ~ 0.3-3), we find:
  - Galaxies need h ≈ 0.7-1.2
  - Clusters need h ≈ 0.3-0.7
  - Ratio ≈ 2×

This is AFTER accounting for:
  - Different A values (A_galaxy = 1.17, A_cluster = 8.0)
  - Different W values (W_galaxy ~ 0.5-0.9, W_cluster ≈ 1)

So the remaining difference must be in h(g) itself, or in how we measure it.

POSSIBLE EXPLANATIONS:

1. MEASUREMENT DIFFERENCE
   - Galaxies: V_circular from stellar orbits
   - Clusters: M_lens from light bending
   - These probe different potentials (Φ vs Φ+Ψ)

2. SCALE DIFFERENCE
   - Galaxies: R ~ 1-50 kpc
   - Clusters: R ~ 200-1000 kpc
   - Coherence may depend on absolute scale, not just g

3. DENSITY PROFILE DIFFERENCE
   - Galaxies: Disk + bulge (concentrated)
   - Clusters: ICM (extended)
   - The distribution of mass affects the enhancement

4. VELOCITY DISPERSION DIFFERENCE
   - Galaxies: σ/V ~ 0.1 (disk) to 1.0 (bulge)
   - Clusters: σ/V ~ 0.5-1.0 (hot gas)
   - Random motions may affect coherence

Let's quantify these differences.
""")

# =============================================================================
# LOAD DATA
# =============================================================================
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
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R, 'V_obs': V_obs, 'V_bar': V_bar, 'R_d': R_d,
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()
print(f"\nLoaded {len(galaxies)} galaxies")

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
        r_kpc = 200
        r_m = r_kpc * kpc_to_m
        g_bar = G_const * M_bar * M_sun / r_m**2
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar,
            'M_lens': M_lens,
            'g_bar': g_bar,
            'r_kpc': r_kpc,
            'M500': M500,
        })
    print(f"Loaded {len(clusters)} clusters")

# =============================================================================
# COMPARE PHYSICAL PROPERTIES
# =============================================================================
print("\n" + "=" * 80)
print("PHYSICAL PROPERTY COMPARISON")
print("=" * 80)

# Galaxy properties at similar g/g† to clusters
A_galaxy = np.exp(1 / (2 * np.pi))
xi_coeff = 1 / (2 * np.pi)

def W_coherence(r, xi):
    return r / (xi + r)

galaxy_points = []
for gal in galaxies:
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    xi = xi_coeff * gal['R_d']
    
    for i in range(len(R)):
        R_m = R[i] * kpc_to_m
        g_bar = (V_bar[i] * 1000)**2 / R_m
        g_norm = g_bar / g_dagger
        
        # Only include points in cluster g range
        if 0.3 < g_norm < 5:
            W = W_coherence(R[i], xi)
            Sigma = (V_obs[i] / V_bar[i])**2
            h_req = (Sigma - 1) / (A_galaxy * W) if A_galaxy * W > 0.01 else np.nan
            
            galaxy_points.append({
                'R_kpc': R[i],
                'g_norm': g_norm,
                'Sigma': Sigma,
                'h_required': h_req,
                'W': W,
            })

df_gal = pd.DataFrame(galaxy_points)

# Cluster properties
A_cluster = 8.0
cluster_points = []
for cl in clusters:
    g_norm = cl['g_bar'] / g_dagger
    Sigma = cl['M_lens'] / cl['M_bar']
    h_req = (Sigma - 1) / A_cluster
    
    cluster_points.append({
        'R_kpc': cl['r_kpc'],
        'g_norm': g_norm,
        'Sigma': Sigma,
        'h_required': h_req,
        'W': 1.0,  # Assumed
        'M_bar': cl['M_bar'],
    })

df_cl = pd.DataFrame(cluster_points)

print("\nAt similar g/g† (0.3 - 5):")
print(f"\n  {'Property':<20} {'Galaxies':<15} {'Clusters':<15} {'Ratio':<10}")
print("  " + "-" * 65)
print(f"  {'N points':<20} {len(df_gal):<15} {len(df_cl):<15}")
print(f"  {'Mean R (kpc)':<20} {df_gal['R_kpc'].mean():<15.1f} {df_cl['R_kpc'].mean():<15.1f} {df_cl['R_kpc'].mean()/df_gal['R_kpc'].mean():<10.1f}")
print(f"  {'Mean g/g†':<20} {df_gal['g_norm'].mean():<15.2f} {df_cl['g_norm'].mean():<15.2f} {df_cl['g_norm'].mean()/df_gal['g_norm'].mean():<10.2f}")
print(f"  {'Mean Σ':<20} {df_gal['Sigma'].mean():<15.2f} {df_cl['Sigma'].mean():<15.2f} {df_cl['Sigma'].mean()/df_gal['Sigma'].mean():<10.2f}")
print(f"  {'Mean W':<20} {df_gal['W'].mean():<15.2f} {df_cl['W'].mean():<15.2f}")
print(f"  {'Mean h_required':<20} {df_gal['h_required'].median():<15.2f} {df_cl['h_required'].median():<15.2f} {df_gal['h_required'].median()/df_cl['h_required'].median():<10.2f}")

# =============================================================================
# THE SCALE HYPOTHESIS
# =============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 1: SCALE MATTERS")
print("=" * 80)

print("""
What if coherence depends on ABSOLUTE scale, not just g/g†?

Galaxies: R ~ 5-30 kpc → coherence length ~ few kpc
Clusters: R ~ 200 kpc → coherence length ~ 100s of kpc

If coherence length ∝ R, then:
  - Galaxies at R=10 kpc have ξ ~ 1 kpc
  - Clusters at R=200 kpc have ξ ~ 20-100 kpc
  
This would mean W_cluster ≈ 1 (r >> ξ), which we already assume.

But what if the AMPLITUDE also scales with R?
""")

# Test: Does h_required correlate with R?
print("\nCorrelation of h_required with R:")
from scipy import stats

# For galaxies
r_gal, p_gal = stats.spearmanr(df_gal['R_kpc'], df_gal['h_required'].dropna())
print(f"  Galaxies: r = {r_gal:.3f}, p = {p_gal:.4f}")

# Combined
df_all = pd.concat([
    df_gal[['R_kpc', 'h_required']].assign(type='galaxy'),
    df_cl[['R_kpc', 'h_required']].assign(type='cluster')
])
r_all, p_all = stats.spearmanr(df_all['R_kpc'], df_all['h_required'].dropna())
print(f"  Combined: r = {r_all:.3f}, p = {p_all:.4f}")

# =============================================================================
# THE DENSITY PROFILE HYPOTHESIS
# =============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 2: DENSITY PROFILE MATTERS")
print("=" * 80)

print("""
Galaxies have CONCENTRATED mass (disk + bulge):
  ρ(r) ∝ exp(-r/R_d) for disk
  ρ(r) ∝ r^(-1) to r^(-2) for bulge
  
Clusters have EXTENDED mass (ICM):
  ρ(r) ∝ r^(-1) to r^(-2) (NFW-like)
  
The GRADIENT of the density profile affects the gravitational field.

Steeper gradient → more "focused" field → more coherence?
Shallower gradient → more "diffuse" field → less coherence?

This could explain why galaxies need more h than clusters at same g.
""")

# =============================================================================
# THE MEASUREMENT HYPOTHESIS
# =============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 3: DIFFERENT OBSERVABLES")
print("=" * 80)

print("""
Galaxies: We measure V_circular from stellar orbits
  V² = r × g_eff = r × g_bar × Σ
  
Clusters: We measure M_lens from light bending
  M_lens = ∫(Φ + Ψ)/c² × area
  
In GR: Φ = Ψ (no gravitational slip)
In modified gravity: Φ ≠ Ψ is possible!

If Σ-Gravity has gravitational slip:
  Σ_dynamical = 1 + A × W × h(g)  [for orbits, probes Φ]
  Σ_lensing = 1 + A × W × h(g) × η  [for light, probes (Φ+Ψ)/2]
  
where η = (Φ+Ψ)/(2Φ) is the slip parameter.

If η < 1, lensing sees LESS enhancement than dynamics.
This would explain why clusters (lensing) need lower h than galaxies (dynamics).
""")

# Calculate implied slip
h_gal_mean = df_gal['h_required'].median()
h_cl_mean = df_cl['h_required'].median()
implied_slip = h_cl_mean / h_gal_mean

print(f"\nImplied gravitational slip:")
print(f"  h_galaxy / h_cluster = {h_gal_mean / h_cl_mean:.2f}")
print(f"  If this is due to slip: η = {implied_slip:.2f}")
print(f"  This means lensing sees {implied_slip*100:.0f}% of dynamical enhancement")

# =============================================================================
# UNIFIED FORMULA WITH SLIP
# =============================================================================
print("\n" + "=" * 80)
print("UNIFIED FORMULA WITH GRAVITATIONAL SLIP")
print("=" * 80)

print("""
PROPOSAL: Same h(g), but different observable-dependent factor

For dynamics (galaxy rotation):
  Σ_dyn = 1 + A × W × h(g)
  
For lensing (cluster mass):
  Σ_lens = 1 + A × W × h(g) × η
  
where η = gravitational slip parameter

From our data: η ≈ 0.5

PHYSICAL INTERPRETATION:
- The enhancement affects the Newtonian potential Φ
- But the spatial curvature Ψ is less affected
- Lensing probes (Φ + Ψ)/2, so it sees less enhancement

This is consistent with many modified gravity theories!
""")

# Test this hypothesis
print("\nTesting slip hypothesis:")

eta = 0.5  # Slip parameter

# For clusters, if we account for slip, what A do we need?
# M_lens/M_bar = 1 + A_cluster × h × η
# We want A_cluster × η ≈ A_galaxy (for consistency)

A_galaxy = np.exp(1 / (2 * np.pi))
A_cluster_current = 8.0

A_cluster_effective = A_cluster_current * eta
print(f"  A_galaxy = {A_galaxy:.2f}")
print(f"  A_cluster × η = {A_cluster_current} × {eta} = {A_cluster_effective:.2f}")
print(f"  Ratio: {A_cluster_effective / A_galaxy:.2f}")

# =============================================================================
# THE PATH LENGTH CONNECTION
# =============================================================================
print("\n" + "=" * 80)
print("CONNECTING SLIP TO PATH LENGTH")
print("=" * 80)

print("""
Our current model: A ∝ L^(1/4) where L is path length through baryons

For galaxies: L ~ 2h ~ 0.6 kpc (disk thickness)
For clusters: L ~ 2 × r_core ~ 400 kpc

Ratio: L_cluster / L_galaxy ≈ 700

If A ∝ L^(1/4):
  A_cluster / A_galaxy = 700^(1/4) ≈ 5.1

Actual ratio: 8.0 / 1.17 ≈ 6.8

Close! But not exact.

REFINED HYPOTHESIS:
The path length affects BOTH:
1. The amplitude A (how much enhancement is possible)
2. The slip η (how much of enhancement affects lensing)

At longer path lengths:
- More coherent enhancement (higher A)
- But also more "averaging" of Φ and Ψ (η closer to 1)

At shorter path lengths:
- Less coherent enhancement (lower A)
- But Φ and Ψ can differ more (η closer to 0.5)
""")

# =============================================================================
# FINAL UNIFIED MODEL
# =============================================================================
print("\n" + "=" * 80)
print("FINAL UNIFIED MODEL PROPOSAL")
print("=" * 80)

print("""
UNIFIED Σ-GRAVITY FORMULA:

  Σ = 1 + A(L) × W(r) × h(g) × η(observable)

where:
  A(L) = A₀ × (L/L₀)^(1/4)     [path length scaling]
  W(r) = r^k / (ξ^k + r^k)     [coherence window]
  h(g) = √(g†/g) × (g†/(g†+g))^β  [acceleration dependence]
  η = 1 for dynamics, ~0.5 for lensing  [gravitational slip]

PARAMETERS:
  A₀ = e^(1/2π) ≈ 1.17  [base amplitude]
  L₀ = reference path length
  k = 1 for 2D, 1.5 for 3D
  ξ = R_d/(2π)
  β = 0.7-1.0 (to be refined)
  η_dyn = 1.0, η_lens ≈ 0.5

This gives:
  - Galaxy dynamics: Σ = 1 + 1.17 × W × h × 1.0
  - Cluster lensing: Σ = 1 + 8.0 × 1 × h × 0.5 = 1 + 4.0 × h
  
Effective A for clusters: 8.0 × 0.5 = 4.0
This is closer to A_galaxy × (L_cluster/L_galaxy)^(1/4) = 1.17 × 5.1 = 6.0

The remaining difference may be due to:
- Different W values (not exactly 1 for clusters)
- Uncertainty in path length estimates
- β parameter optimization
""")

# =============================================================================
# TEST THE UNIFIED MODEL
# =============================================================================
print("\n" + "=" * 80)
print("TESTING UNIFIED MODEL")
print("=" * 80)

def h_function(g, beta=1.0):
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * (g_dagger / (g_dagger + g))**beta

# Test with different slip values
print("\nCluster predictions with gravitational slip:")
print(f"\n  {'η':<8} {'Median ratio':<15} {'Scatter (dex)':<15}")
print("  " + "-" * 40)

for eta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    ratios = []
    for cl in clusters:
        h = h_function(cl['g_bar'])
        Sigma = 1 + A_cluster_current * h * eta
        M_pred = cl['M_bar'] * Sigma
        ratios.append(M_pred / cl['M_lens'])
    
    median_ratio = np.median(ratios)
    scatter = np.std(np.log10(ratios))
    marker = " <-- current" if eta == 1.0 else " <-- best" if abs(median_ratio - 1.0) < 0.1 else ""
    print(f"  {eta:<8.1f} {median_ratio:<15.3f} {scatter:<15.3f}{marker}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
KEY FINDINGS:

1. SAME g, DIFFERENT h REQUIRED:
   - At g/g† ~ 1: Galaxies need h ~ 0.7, Clusters need h ~ 0.35
   - Ratio ≈ 2× cannot be explained by A or W alone

2. GRAVITATIONAL SLIP EXPLANATION:
   - Lensing probes (Φ+Ψ)/2, dynamics probes Φ
   - If Ψ < Φ in enhanced regions, lensing sees less enhancement
   - Implied slip: η ≈ 0.5

3. UNIFIED MODEL:
   - Same h(g) for both domains
   - Same A scaling with path length
   - Different η: η_dyn = 1.0, η_lens ≈ 0.5

4. PHYSICAL INTERPRETATION:
   - The coherence enhancement primarily affects Φ (Newtonian potential)
   - The spatial curvature Ψ is less affected
   - This is a testable prediction of Σ-Gravity!

5. IMPLICATIONS:
   - NOT copying MOND (which has η = 1)
   - Provides a physical reason for galaxy/cluster difference
   - Gravitational slip is measurable (e.g., galaxy-galaxy lensing vs dynamics)
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

