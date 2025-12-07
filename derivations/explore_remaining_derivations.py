#!/usr/bin/env python3
"""
EXPLORE REMAINING DERIVATIONS

This script explores:
1. The 4√π factor in g† - can we derive it?
2. The L^(1/4) exponent - why 1/4?
3. Per-galaxy/cluster analysis to find patterns
4. What physics distinguishes good vs poor fits?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 3e8
H0 = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19

print("=" * 80)
print("EXPLORING REMAINING DERIVATIONS")
print("=" * 80)

# =============================================================================
# PART I: THE 4√π FACTOR
# =============================================================================
print("\n" + "=" * 80)
print("PART I: DERIVING THE 4√π FACTOR IN g†")
print("=" * 80)

print("""
Current: g† = cH₀/(4√π) ≈ 9.6 × 10⁻¹¹ m/s²

The factor 4√π ≈ 7.09 needs derivation. Let's explore possibilities:
""")

# Test different factors
factors = [
    ("2π", 2 * np.pi, "Circumference of unit circle"),
    ("4", 4, "Number of spacetime dimensions"),
    ("√(4π)", np.sqrt(4 * np.pi), "Surface area factor (4πr²)"),
    ("4√π", 4 * np.sqrt(np.pi), "Current value"),
    ("2e", 2 * np.e, "Original value (deprecated)"),
    ("π²/√2", np.pi**2 / np.sqrt(2), "π² geometry"),
    ("8/√e", 8 / np.sqrt(np.e), "8 dimensions / √e"),
    ("2π√e", 2 * np.pi * np.sqrt(np.e), "2π × √e"),
]

# MOND's a0 for comparison
a0_mond = 1.2e-10

print(f"  Factor         Value      g† (m/s²)      Ratio to a₀")
print("  " + "-" * 60)

for name, factor, desc in factors:
    g_dagger = c * H0 / factor
    ratio = g_dagger / a0_mond
    marker = " <-- CURRENT" if name == "4√π" else ""
    print(f"  {name:<14} {factor:<10.4f} {g_dagger:.3e}    {ratio:.3f}{marker}")

print("""
HYPOTHESIS 1: GAUSSIAN COHERENCE KERNEL
───────────────────────────────────────
If the coherence field has a 3D Gaussian profile:
    C(r) = exp(-r²/2σ²)

The normalization factor is (2πσ²)^(3/2).
For σ = R_H/(4√π), this gives the 4√π factor.

HYPOTHESIS 2: SPHERICAL AVERAGING
─────────────────────────────────
If we average over a sphere of radius R:
    ⟨g⟩ = ∫ g(r) × (4πr²) dr / (4πR³/3)
    
The 4π from surface area and the factor from volume give 4√π.

HYPOTHESIS 3: DIMENSIONAL REDUCTION
───────────────────────────────────
4 = spacetime dimensions
√π = Gaussian normalization in 1D
4√π = product of dimensional and statistical factors
""")

# =============================================================================
# PART II: THE L^(1/4) EXPONENT
# =============================================================================
print("\n" + "=" * 80)
print("PART II: THE L^(1/4) EXPONENT")
print("=" * 80)

print("""
Current: A = A₀ × L^(1/4) where L is path length through baryons

Why 1/4? Possibilities:

1. SPACETIME DIMENSION: 1/d for d = 4
   - In 4D spacetime, field strength scales as 1/r²
   - Integrated effect over path L scales as L^(1/d) = L^(1/4)

2. NESTED AVERAGING:
   - If we average over 4 independent dimensions
   - Each contributes L^(1/4)

3. PROPAGATOR SCALING:
   - In QFT, propagators in d dimensions scale as p^(2-d)
   - For d = 4: p^(-2), which gives L^(1/4) after integration

Let's test if 1/4 is actually optimal:
""")

# Load data
data_dir = Path(__file__).parent.parent / "data"

def load_sparc():
    galaxies = []
    rotmod_dir = data_dir / "Rotmod_LTG"
    if not rotmod_dir.exists():
        return []
    
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
            
            # Estimate path length (2 × scale height)
            L = 2 * 0.3 * R_d  # Typical h/R_d ≈ 0.3
            
            galaxies.append({
                'name': f.stem,
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'R_d': R_d,
                'L': L,
                'V_flat': np.median(V_obs[-5:]) if len(V_obs) >= 5 else V_obs[-1],
            })
        except:
            continue
    return galaxies

sparc = load_sparc()
print(f"Loaded {len(sparc)} SPARC galaxies")

# =============================================================================
# PART III: PER-GALAXY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("PART III: PER-GALAXY FIT QUALITY ANALYSIS")
print("=" * 80)

g_dagger = c * H0 / (4 * np.sqrt(np.pi))

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return r / (xi + r)

def predict_velocity(R, V_bar, R_d, A, xi_coeff):
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    xi = xi_coeff * R_d
    W = W_coherence(R, xi)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)

def mond_velocity(R, V_bar):
    a0 = 1.2e-10
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    y = g_bar / a0
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    return V_bar * np.sqrt(nu)

# Compute per-galaxy metrics
A_galaxy = np.exp(1 / (2 * np.pi))
xi_coeff = 1 / (2 * np.pi)

galaxy_results = []
for gal in sparc:
    V_pred = predict_velocity(gal['R'], gal['V_bar'], gal['R_d'], A_galaxy, xi_coeff)
    V_mond = mond_velocity(gal['R'], gal['V_bar'])
    
    rms_sigma = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
    rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
    
    # Compute characteristic acceleration
    R_m = gal['R'] * kpc_to_m
    g_bar = (gal['V_bar'] * 1000)**2 / R_m
    g_mean = np.mean(g_bar)
    g_outer = np.mean(g_bar[-3:]) if len(g_bar) >= 3 else g_bar[-1]
    
    galaxy_results.append({
        'name': gal['name'],
        'R_d': gal['R_d'],
        'V_flat': gal['V_flat'],
        'L': gal['L'],
        'rms_sigma': rms_sigma,
        'rms_mond': rms_mond,
        'wins': rms_sigma < rms_mond,
        'g_mean': g_mean,
        'g_outer': g_outer,
        'n_points': len(gal['R']),
        'R_max': gal['R'].max(),
    })

df = pd.DataFrame(galaxy_results)

print(f"\nPer-galaxy statistics:")
print(f"  Total galaxies: {len(df)}")
print(f"  Σ-Gravity wins: {df['wins'].sum()} ({df['wins'].mean()*100:.1f}%)")
print(f"  Mean RMS (Σ-Gravity): {df['rms_sigma'].mean():.2f} km/s")
print(f"  Mean RMS (MOND): {df['rms_mond'].mean():.2f} km/s")

# Analyze what distinguishes good vs poor fits
print("\n" + "-" * 60)
print("WHAT DISTINGUISHES GOOD VS POOR FITS?")
print("-" * 60)

# Split into quartiles by Σ-Gravity RMS
df['rms_quartile'] = pd.qcut(df['rms_sigma'], 4, labels=['Best', 'Good', 'Fair', 'Poor'])

print("\nBy fit quality quartile:")
print(f"\n  Quartile   N    RMS (km/s)   R_d (kpc)   V_flat (km/s)   g_outer/g†")
print("  " + "-" * 70)

for q in ['Best', 'Good', 'Fair', 'Poor']:
    subset = df[df['rms_quartile'] == q]
    print(f"  {q:<10} {len(subset):<4} {subset['rms_sigma'].mean():<12.1f} {subset['R_d'].mean():<11.2f} {subset['V_flat'].mean():<15.0f} {subset['g_outer'].mean()/g_dagger:.2f}")

# Correlation analysis
print("\nCorrelations with Σ-Gravity RMS:")
print(f"  R_d:       r = {df['R_d'].corr(df['rms_sigma']):.3f}")
print(f"  V_flat:    r = {df['V_flat'].corr(df['rms_sigma']):.3f}")
print(f"  g_outer:   r = {df['g_outer'].corr(df['rms_sigma']):.3f}")
print(f"  R_max:     r = {df['R_max'].corr(df['rms_sigma']):.3f}")
print(f"  n_points:  r = {df['n_points'].corr(df['rms_sigma']):.3f}")

# Where does Σ-Gravity beat MOND?
print("\n" + "-" * 60)
print("WHERE DOES Σ-GRAVITY BEAT MOND?")
print("-" * 60)

wins = df[df['wins']]
losses = df[~df['wins']]

print(f"\n  Property       Σ-Gravity Wins    MOND Wins")
print("  " + "-" * 50)
print(f"  Mean R_d:      {wins['R_d'].mean():.2f} kpc          {losses['R_d'].mean():.2f} kpc")
print(f"  Mean V_flat:   {wins['V_flat'].mean():.0f} km/s         {losses['V_flat'].mean():.0f} km/s")
print(f"  Mean g_outer:  {wins['g_outer'].mean()/g_dagger:.2f} g†            {losses['g_outer'].mean()/g_dagger:.2f} g†")
print(f"  Mean R_max:    {wins['R_max'].mean():.1f} kpc          {losses['R_max'].mean():.1f} kpc")

# Statistical tests
_, p_Rd = stats.mannwhitneyu(wins['R_d'], losses['R_d'])
_, p_Vflat = stats.mannwhitneyu(wins['V_flat'], losses['V_flat'])
_, p_g = stats.mannwhitneyu(wins['g_outer'], losses['g_outer'])

print(f"\n  Statistical significance (Mann-Whitney U):")
print(f"    R_d difference: p = {p_Rd:.4f}")
print(f"    V_flat difference: p = {p_Vflat:.4f}")
print(f"    g_outer difference: p = {p_g:.4f}")

# =============================================================================
# PART IV: TESTING ALTERNATIVE g† FACTORS
# =============================================================================
print("\n" + "=" * 80)
print("PART IV: TESTING ALTERNATIVE g† FACTORS")
print("=" * 80)

def evaluate_g_factor(factor):
    """Evaluate performance with different g† factor."""
    g_dagger_test = c * H0 / factor
    
    def h_test(g):
        g = np.maximum(np.asarray(g), 1e-15)
        return np.sqrt(g_dagger_test / g) * g_dagger_test / (g_dagger_test + g)
    
    rms_list = []
    for gal in sparc:
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_test(g_bar)
        Sigma = 1 + A_galaxy * W * h
        V_pred = gal['V_bar'] * np.sqrt(Sigma)
        rms = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
        rms_list.append(rms)
    
    return np.mean(rms_list)

print("\nTesting different g† factors:")
print(f"\n  Factor         g† (m/s²)      Mean RMS (km/s)")
print("  " + "-" * 50)

best_factor = None
best_rms = float('inf')

for name, factor, desc in factors:
    rms = evaluate_g_factor(factor)
    if rms < best_rms:
        best_rms = rms
        best_factor = (name, factor)
    marker = " <-- CURRENT" if name == "4√π" else ""
    g_val = c * H0 / factor
    print(f"  {name:<14} {g_val:.3e}    {rms:.2f}{marker}")

print(f"\n  Best factor: {best_factor[0]} = {best_factor[1]:.4f}")
print(f"  Current (4√π): {4 * np.sqrt(np.pi):.4f}")

# =============================================================================
# PART V: TESTING ALTERNATIVE L EXPONENTS
# =============================================================================
print("\n" + "=" * 80)
print("PART V: TESTING ALTERNATIVE L EXPONENTS")
print("=" * 80)

print("""
Testing if 1/4 is optimal for the path length exponent.
We'll fit A = A₀ × L^α and find optimal α.
""")

# For this we need path length estimates
# Using L ≈ 2h where h ≈ 0.3 R_d (typical disk scale height)

def evaluate_L_exponent(alpha):
    """Evaluate performance with different L exponent."""
    A0 = np.exp(1 / (2 * np.pi))  # Base amplitude
    L_ref = 1.0  # Reference path length (kpc)
    
    rms_list = []
    for gal in sparc:
        L = gal['L']  # Path length estimate
        A = A0 * (L / L_ref)**alpha
        
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_function(g_bar)
        Sigma = 1 + A * W * h
        V_pred = gal['V_bar'] * np.sqrt(Sigma)
        rms = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
        rms_list.append(rms)
    
    return np.mean(rms_list)

print(f"\n  α (exponent)   Mean RMS (km/s)")
print("  " + "-" * 35)

best_alpha = None
best_rms_alpha = float('inf')

for alpha in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    rms = evaluate_L_exponent(alpha)
    if rms < best_rms_alpha:
        best_rms_alpha = rms
        best_alpha = alpha
    marker = " <-- 1/4" if alpha == 0.25 else ""
    print(f"  {alpha:<14.2f} {rms:.2f}{marker}")

print(f"\n  Best α = {best_alpha:.2f}")
print(f"  1/4 = 0.25 (theoretical)")

# =============================================================================
# PART VI: WORST-FIT GALAXY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("PART VI: WORST-FIT GALAXIES - WHAT'S DIFFERENT?")
print("=" * 80)

worst_10 = df.nlargest(10, 'rms_sigma')
best_10 = df.nsmallest(10, 'rms_sigma')

print("\nWorst 10 fits:")
print(f"\n  Galaxy              RMS (km/s)   R_d    V_flat   g_outer/g†")
print("  " + "-" * 65)
for _, row in worst_10.iterrows():
    print(f"  {row['name']:<20} {row['rms_sigma']:<12.1f} {row['R_d']:<6.2f} {row['V_flat']:<8.0f} {row['g_outer']/g_dagger:.2f}")

print("\nBest 10 fits:")
print(f"\n  Galaxy              RMS (km/s)   R_d    V_flat   g_outer/g†")
print("  " + "-" * 65)
for _, row in best_10.iterrows():
    print(f"  {row['name']:<20} {row['rms_sigma']:<12.1f} {row['R_d']:<6.2f} {row['V_flat']:<8.0f} {row['g_outer']/g_dagger:.2f}")

print("""
PATTERNS TO INVESTIGATE:
1. Do worst-fit galaxies have unusual morphology?
2. Are they edge-on (extinction issues)?
3. Do they have bars or warps?
4. Are they interacting?
5. Do they have unusual gas fractions?
""")

# =============================================================================
# PART VII: CLUSTER ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("PART VII: CLUSTER ANALYSIS")
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
    
    print(f"\nAnalyzing {len(cl_valid)} clusters")
    
    # Compute per-cluster predictions
    cluster_results = []
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
        
        cluster_results.append({
            'name': row['cluster'],
            'M500': M500,
            'M_bar': M_bar,
            'M_lens': M_lens,
            'M_pred': M_pred,
            'ratio': ratio,
            'g_bar': g_bar,
            'z': row['spec_z'] if 'spec_z' in row else np.nan,
        })
    
    cl_results = pd.DataFrame(cluster_results)
    
    print(f"\nCluster statistics:")
    print(f"  Median ratio: {cl_results['ratio'].median():.3f}")
    print(f"  Scatter: {np.std(np.log10(cl_results['ratio'])):.3f} dex")
    
    # Best and worst clusters
    print("\nWorst 5 clusters (ratio furthest from 1):")
    cl_results['abs_log_ratio'] = np.abs(np.log10(cl_results['ratio']))
    worst_cl = cl_results.nlargest(5, 'abs_log_ratio')
    print(f"\n  Cluster              Ratio    M500 (1e14)   g_bar/g†")
    print("  " + "-" * 55)
    for _, row in worst_cl.iterrows():
        print(f"  {row['name']:<20} {row['ratio']:<8.3f} {row['M500']/1e14:<13.1f} {row['g_bar']/g_dagger:.2f}")
    
    # Correlation with properties
    print("\nCorrelations with log(ratio):")
    log_ratio = np.log10(cl_results['ratio'])
    print(f"  M500: r = {cl_results['M500'].corr(log_ratio):.3f}")
    print(f"  g_bar: r = {cl_results['g_bar'].corr(log_ratio):.3f}")
else:
    print("  [Cluster data not found]")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: PATHS TO FURTHER DERIVATION")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ REMAINING DERIVATIONS AND APPROACHES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. THE 4√π FACTOR                                                           │
│    - Current: g† = cH₀/(4√π)                                                │
│    - Best alternative found: [see above]                                    │
│    - Derivation path: Gaussian coherence kernel in 3D+1                     │
│                                                                             │
│ 2. THE L^(1/4) EXPONENT                                                     │
│    - Current: A = A₀ × L^(1/4)                                              │
│    - Best α found: [see above]                                              │
│    - Derivation path: 1/d for d=4 spacetime dimensions                      │
│                                                                             │
│ 3. GALAXY FIT QUALITY PATTERNS                                              │
│    - Σ-Gravity wins on: [larger/smaller] R_d, [higher/lower] V_flat         │
│    - MOND wins on: [opposite pattern]                                       │
│    - Investigate: morphology, gas fraction, environment                     │
│                                                                             │
│ 4. CLUSTER FIT QUALITY PATTERNS                                             │
│    - Outliers: [see worst clusters above]                                   │
│    - Investigate: dynamical state, merging history, BCG properties          │
│                                                                             │
│ 5. NEXT STEPS                                                               │
│    a. Look up worst-fit galaxies in literature for unusual properties       │
│    b. Test if 4√π can be derived from Gaussian kernel normalization         │
│    c. Test if 1/4 can be derived from 4D propagator scaling                 │
│    d. Explore per-galaxy amplitude fitting to find L-dependence             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("END OF EXPLORATION")
print("=" * 80)

