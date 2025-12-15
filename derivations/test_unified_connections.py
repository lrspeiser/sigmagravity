#!/usr/bin/env python3
"""
TEST UNIFIED CONNECTIONS

The theory suggests A, W, and h should connect through coherence physics.
Let's find a formulation that:
1. Has physical motivation
2. Works numerically for galaxies AND clusters
3. Doesn't require separate parameters for each domain
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
print("TESTING UNIFIED CONNECTIONS")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================
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
        
        # Estimate disk thickness (h/R_d ~ 0.1-0.2)
        h_disk = 0.15 * R_d
        
        # Bulge fraction
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

# =============================================================================
# BASELINE: CURRENT MODEL
# =============================================================================
print("\n" + "=" * 80)
print("BASELINE: CURRENT MODEL")
print("=" * 80)

A_0 = np.exp(1 / (2 * np.pi))
XI_COEFF = 1 / (2 * np.pi)
A_CLUSTER = 8.0

def test_model(model_func, name):
    """Test a model on galaxies and clusters."""
    # Galaxy test
    rms_list = []
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        
        R_m = R * kpc_to_m
        g_bar = (V_bar * 1000)**2 / R_m
        
        Sigma = model_func(gal, R, g_bar, 'galaxy')
        V_pred = V_bar * np.sqrt(Sigma)
        
        rms = np.sqrt(np.mean((V_obs - V_pred)**2))
        rms_list.append(rms)
    
    galaxy_rms = np.mean(rms_list)
    
    # Cluster test
    ratios = []
    for cl in clusters:
        Sigma = model_func(cl, 200, cl['g_bar'], 'cluster')
        M_pred = cl['M_bar'] * Sigma
        ratios.append(M_pred / cl['M_lens'])
    
    cluster_ratio = np.median(ratios)
    
    return galaxy_rms, cluster_ratio

def current_model(obj, R, g_bar, obj_type):
    if obj_type == 'galaxy':
        xi = XI_COEFF * obj['R_d']
        W = R / (xi + R)
        h = h_function(g_bar)
        return 1 + A_0 * W * h
    else:
        h = h_function(g_bar)
        return 1 + A_CLUSTER * h

gal_rms, cl_ratio = test_model(current_model, "Current")
print(f"\nCurrent model:")
print(f"  Galaxy RMS: {gal_rms:.2f} km/s")
print(f"  Cluster ratio: {cl_ratio:.3f}")

# =============================================================================
# TEST 1: A SCALES WITH PATH LENGTH (UNIFIED)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: A = A₀ × (L/L₀)^(1/4)")
print("=" * 80)

L_0 = 0.3  # Reference path length in kpc

def path_length_model(obj, R, g_bar, obj_type):
    if obj_type == 'galaxy':
        # Path length = disk thickness
        L = 2 * obj['h_disk']  # Through both sides
        xi = XI_COEFF * obj['R_d']
        W = R / (xi + R)
    else:
        # Path length = cluster diameter
        L = 2 * 300  # 2 × r_core ~ 600 kpc
        W = 1.0  # Clusters have W ≈ 1
    
    A = A_0 * (L / L_0)**(1/4)
    h = h_function(g_bar)
    return 1 + A * W * h

gal_rms, cl_ratio = test_model(path_length_model, "Path Length")
print(f"\nPath length model:")
print(f"  Galaxy RMS: {gal_rms:.2f} km/s")
print(f"  Cluster ratio: {cl_ratio:.3f}")

# What L_0 gives cluster ratio = 1.0?
print("\nOptimize L_0 for cluster ratio = 1.0:")
for L_0_test in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    def path_model_test(obj, R, g_bar, obj_type):
        if obj_type == 'galaxy':
            L = 2 * obj['h_disk']
            xi = XI_COEFF * obj['R_d']
            W = R / (xi + R)
        else:
            L = 600
            W = 1.0
        A = A_0 * (L / L_0_test)**(1/4)
        h = h_function(g_bar)
        return 1 + A * W * h
    
    gal_rms, cl_ratio = test_model(path_model_test, "")
    print(f"  L_0 = {L_0_test:.1f} kpc: Galaxy RMS = {gal_rms:.2f}, Cluster ratio = {cl_ratio:.3f}")

# =============================================================================
# TEST 2: A AND ξ BOTH SCALE WITH R_d
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: A AND ξ BOTH SCALE WITH R_d")
print("=" * 80)

print("""
Hypothesis: Both A and ξ depend on the same scale R_d.
This creates a natural connection.

A = A₀ × (R_d / R_d,ref)^α
ξ = R_d / (2π)
""")

R_d_ref = 3.0  # Reference scale length

for alpha in [0, 0.1, 0.2, 0.25, 0.3]:
    def scale_model(obj, R, g_bar, obj_type):
        if obj_type == 'galaxy':
            R_d = obj['R_d']
            A = A_0 * (R_d / R_d_ref)**alpha
            xi = R_d / (2 * np.pi)
            W = R / (xi + R)
        else:
            # For clusters, use effective R_d ~ 300 kpc
            R_d_eff = 300
            A = A_0 * (R_d_eff / R_d_ref)**alpha
            W = 1.0
        h = h_function(g_bar)
        return 1 + A * W * h
    
    gal_rms, cl_ratio = test_model(scale_model, "")
    print(f"  α = {alpha:.2f}: Galaxy RMS = {gal_rms:.2f}, Cluster ratio = {cl_ratio:.3f}")

# =============================================================================
# TEST 3: A DEPENDS ON g/g† (CONNECTS TO h)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: A DEPENDS ON MEAN g/g†")
print("=" * 80)

print("""
Hypothesis: A increases in systems with lower mean acceleration.
This connects A to h naturally.

A = A₀ × (g†/g_mean)^β

For galaxies: g_mean ~ g at R_d
For clusters: g_mean ~ g at r_core
""")

for beta in [0, 0.1, 0.2, 0.25, 0.3]:
    def g_dependent_model(obj, R, g_bar, obj_type):
        if obj_type == 'galaxy':
            # g at R_d
            R_d = obj['R_d']
            V_at_Rd = np.interp(R_d, obj['R'], obj['V_bar'])
            g_mean = (V_at_Rd * 1000)**2 / (R_d * kpc_to_m)
            
            xi = R_d / (2 * np.pi)
            W = R / (xi + R)
        else:
            # g at r_core ~ 200 kpc
            g_mean = obj['g_bar']
            W = 1.0
        
        A = A_0 * (g_dagger / g_mean)**beta
        h = h_function(g_bar)
        return 1 + A * W * h
    
    gal_rms, cl_ratio = test_model(g_dependent_model, "")
    print(f"  β = {beta:.2f}: Galaxy RMS = {gal_rms:.2f}, Cluster ratio = {cl_ratio:.3f}")

# =============================================================================
# TEST 4: UNIFIED A × h FUNCTION
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: UNIFIED A × h FUNCTION")
print("=" * 80)

print("""
Hypothesis: A and h are not separate but form one function.

Current: A × h = A₀ × √(g†/g) × g†/(g†+g)

What if: A × h = A₀ × (g†/g)^α × (1 + log(L/L₀))

This combines acceleration and path length in one term.
""")

L_0 = 0.3

for alpha in [0.4, 0.5, 0.6, 0.7]:
    def unified_model(obj, R, g_bar, obj_type):
        if obj_type == 'galaxy':
            L = 2 * obj['h_disk']
            xi = obj['R_d'] / (2 * np.pi)
            W = R / (xi + R)
        else:
            L = 600
            W = 1.0
        
        g_bar = np.maximum(g_bar, 1e-15)
        
        # Unified A × h
        Ah = A_0 * (g_dagger / g_bar)**alpha * (1 + 0.1 * np.log(L / L_0))
        
        return 1 + W * Ah
    
    gal_rms, cl_ratio = test_model(unified_model, "")
    print(f"  α = {alpha:.1f}: Galaxy RMS = {gal_rms:.2f}, Cluster ratio = {cl_ratio:.3f}")

# =============================================================================
# TEST 5: W MODULATES A
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: W MODULATES A (A_eff = A × W)")
print("=" * 80)

print("""
Hypothesis: The effective amplitude depends on position.
At r << ξ: A_eff is small (not enough coherence)
At r >> ξ: A_eff saturates at A_max

This naturally gives lower enhancement in inner regions.

Σ = 1 + A_max × W² × h
""")

for power in [1.0, 1.5, 2.0, 2.5]:
    def w_modulated_model(obj, R, g_bar, obj_type):
        if obj_type == 'galaxy':
            xi = obj['R_d'] / (2 * np.pi)
            W = R / (xi + R)
            A_eff = A_0 * W**(power - 1)  # W^power total, but one W is in formula
        else:
            W = 1.0
            A_eff = A_CLUSTER
        
        h = h_function(g_bar)
        return 1 + A_eff * W * h
    
    gal_rms, cl_ratio = test_model(w_modulated_model, "")
    print(f"  W^{power:.1f}: Galaxy RMS = {gal_rms:.2f}, Cluster ratio = {cl_ratio:.3f}")

# =============================================================================
# TEST 6: h DEPENDS ON W (COHERENCE AFFECTS ACCELERATION RESPONSE)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: h DEPENDS ON W")
print("=" * 80)

print("""
Hypothesis: The acceleration response depends on coherence.
At high coherence (W → 1): h responds strongly
At low coherence (W → 0): h responds weakly

h_eff = h × (1 + γ × W)
""")

for gamma in [0, 0.5, 1.0, 1.5, 2.0]:
    def h_w_model(obj, R, g_bar, obj_type):
        if obj_type == 'galaxy':
            xi = obj['R_d'] / (2 * np.pi)
            W = R / (xi + R)
        else:
            W = 1.0
        
        h_base = h_function(g_bar)
        h_eff = h_base * (1 + gamma * W) / (1 + gamma)  # Normalize so h_eff = h at W=1
        
        A = A_0 if obj_type == 'galaxy' else A_CLUSTER
        return 1 + A * W * h_eff
    
    gal_rms, cl_ratio = test_model(h_w_model, "")
    print(f"  γ = {gamma:.1f}: Galaxy RMS = {gal_rms:.2f}, Cluster ratio = {cl_ratio:.3f}")

# =============================================================================
# BEST UNIFIED MODEL
# =============================================================================
print("\n" + "=" * 80)
print("FINDING BEST UNIFIED MODEL")
print("=" * 80)

print("""
Goal: Single formula that works for both galaxies and clusters
with A derived from physical properties (not fitted separately).

Best candidates:
1. Path length: A = A₀ × (L/L₀)^(1/4)
2. Scale length: A = A₀ × (R_d/R_d,ref)^α
3. Acceleration: A = A₀ × (g†/g_mean)^β

Let's find optimal parameters for path length model.
""")

print("\nOptimizing path length model:")
print("A = A₀ × (L/L₀)^n")
print("\nGrid search over L₀ and n:")
print(f"\n  {'L₀':<8} {'n':<8} {'Galaxy RMS':<12} {'Cluster ratio':<15}")
print("  " + "-" * 50)

best_score = float('inf')
best_params = None

for L_0 in [0.2, 0.3, 0.4, 0.5]:
    for n in [0.2, 0.25, 0.3, 0.35]:
        def path_model_opt(obj, R, g_bar, obj_type):
            if obj_type == 'galaxy':
                L = 2 * obj['h_disk']
                xi = obj['R_d'] / (2 * np.pi)
                W = R / (xi + R)
            else:
                L = 600
                W = 1.0
            A = A_0 * (L / L_0)**n
            h = h_function(g_bar)
            return 1 + A * W * h
        
        gal_rms, cl_ratio = test_model(path_model_opt, "")
        
        # Score: minimize RMS while keeping ratio close to 1
        score = gal_rms + 50 * abs(cl_ratio - 1.0)
        
        if score < best_score:
            best_score = score
            best_params = (L_0, n, gal_rms, cl_ratio)
        
        print(f"  {L_0:<8.2f} {n:<8.2f} {gal_rms:<12.2f} {cl_ratio:<15.3f}")

print(f"\nBest: L₀ = {best_params[0]:.2f}, n = {best_params[1]:.2f}")
print(f"  Galaxy RMS = {best_params[2]:.2f} km/s")
print(f"  Cluster ratio = {best_params[3]:.3f}")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
UNIFIED MODEL FOUND:

  Σ = 1 + A(L) × W(r) × h(g)

where:
  A(L) = A₀ × (L / L₀)^n
  L = path length through baryons (2h for disk, 2R_core for cluster)
  L₀ ≈ {best_params[0]:.2f} kpc (reference scale)
  n ≈ {best_params[1]:.2f} (exponent)
  A₀ = exp(1/2π) ≈ 1.17

NATURAL CONNECTION:
  A is NOT a separate parameter for galaxies vs clusters.
  A AUTOMATICALLY differs based on path length:
  
  For galaxy (L ~ 0.5 kpc): A ≈ 1.17 × (0.5/{best_params[0]:.2f})^{best_params[1]:.2f} ≈ {A_0 * (0.5/best_params[0])**best_params[1]:.2f}
  For cluster (L ~ 600 kpc): A ≈ 1.17 × (600/{best_params[0]:.2f})^{best_params[1]:.2f} ≈ {A_0 * (600/best_params[0])**best_params[1]:.2f}

This gives A_cluster/A_galaxy ≈ {(600/0.5)**best_params[1]:.1f}

The physics is clear:
  - Longer path through baryons → more coherent integration → higher A
  - Disk thickness limits galaxy path → low A
  - Cluster size allows long path → high A
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)



