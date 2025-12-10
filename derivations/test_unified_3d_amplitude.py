#!/usr/bin/env python3
"""
Test Unified 3D Amplitude Formula (No D Switch)

The idea: Use A = A₀ × (L/L₀)^n for ALL systems, where L is the
effective path length through baryons. For thin disks, L is naturally
small (~ disk scale height), so A ≈ A₀. For clusters, L is large
(~ 600 kpc), so A >> A₀.

This eliminates the discrete D=0/1 switch by letting geometry do the work.

Key question: What is L for a disk galaxy?
- Option 1: L = scale height h_z (typically 0.3-0.5 kpc)
- Option 2: L = effective path at measurement radius
- Option 3: L = some function of R_d and inclination

Author: Leonard Speiser
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173

# Current canonical parameters
L_0_current = 0.40  # kpc
n_current = 0.27


def h_function(g):
    """Acceleration function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r, xi):
    """Coherence window W(r) = r/(ξ+r)"""
    return r / (xi + r)


def unified_amplitude(L, L0, n):
    """
    Unified amplitude: A = A₀ × (L/L₀)^n
    
    No D switch! Path length L determines everything.
    """
    return A_0 * (L / L0) ** n


def predict_galaxy_velocity(R, V_bar, R_d, L_eff, L0, n):
    """
    Predict rotation velocity using unified 3D amplitude.
    
    L_eff = effective path length for this galaxy
    """
    xi = R_d / (2 * np.pi)
    
    # Acceleration
    g_N = (V_bar * 1000)**2 / (R * kpc_to_m)
    
    # Components
    A = unified_amplitude(L_eff, L0, n)
    W = W_coherence(R, xi)
    h = h_function(g_N)
    
    # Enhancement
    Sigma = 1 + A * W * h
    
    return V_bar * np.sqrt(Sigma)


def compute_disk_path_length(R_d, h_z=0.3):
    """
    Compute effective path length for a disk galaxy.
    
    Options to test:
    1. L = h_z (scale height) - typically 0.3-0.5 kpc
    2. L = 2*h_z (diameter through disk)
    3. L = sqrt(h_z² + R_d²) (geometric mean)
    4. L = h_z × (1 + R_d/h_z)^α (hybrid)
    """
    return h_z  # Start with simplest: scale height


def load_sparc_data():
    """Load SPARC galaxy data."""
    data_dir = Path(__file__).parent.parent / "data" / "Rotmod_LTG"
    
    galaxies = []
    for f in sorted(data_dir.glob("*.dat")):
        try:
            df = pd.read_csv(f, sep=r'\s+', comment='#',
                           names=['R', 'V_obs', 'V_err', 'V_gas', 'V_disk', 'V_bul'])
            
            if len(df) < 5:
                continue
            
            R = df['R'].values
            V_obs = df['V_obs'].values
            V_err = df['V_err'].values
            V_gas = df['V_gas'].values
            V_disk = df['V_disk'].values
            V_bul = df['V_bul'].values if 'V_bul' in df else np.zeros_like(R)
            
            # Baryonic velocity with M/L = 0.5/0.7
            V_bar = np.sqrt(V_gas**2 + 0.5 * V_disk**2 + 0.7 * V_bul**2)
            
            # Estimate R_d from disk velocity peak
            if len(V_disk) > 3:
                peak_idx = np.argmax(np.abs(V_disk))
                R_d = R[min(peak_idx, len(R)-1)] / 2.2  # Peak at ~2.2 R_d
            else:
                R_d = R[len(R)//3]
            
            galaxies.append({
                'name': f.stem,
                'R': R,
                'V_obs': V_obs,
                'V_err': V_err,
                'V_bar': V_bar,
                'R_d': max(R_d, 0.5),
            })
        except:
            continue
    
    return galaxies


def load_cluster_data():
    """Load cluster data."""
    data_dir = Path(__file__).parent.parent / "data"
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    
    if not cluster_file.exists():
        return []
    
    df = pd.read_csv(cluster_file)
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes') &
        (df['M500_1e14Msun'] > 2.0)
    ].copy()
    
    clusters = []
    f_baryon = 0.15
    
    for _, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar = 0.4 * f_baryon * M500
        M_lens = row['MSL_200kpc_1e12Msun'] * 1e12
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar,
            'M_lens': M_lens,
            'r_kpc': 200,
            'L_eff': 600,  # Effective path length for clusters
        })
    
    return clusters


def predict_cluster_mass(M_bar, r_kpc, L_eff, L0, n):
    """Predict cluster mass with unified amplitude."""
    r_m = r_kpc * kpc_to_m
    g_N = G_const * M_bar * M_sun / r_m**2
    
    A = unified_amplitude(L_eff, L0, n)
    W = 1.0  # W ≈ 1 for clusters at r >> ξ
    h = h_function(g_N)
    
    Sigma = 1 + A * W * h
    return M_bar * Sigma


def evaluate_unified_model(galaxies, clusters, L0, n, h_z_disk=0.3):
    """
    Evaluate the unified model with no D switch.
    
    L_eff for galaxies = h_z (scale height)
    L_eff for clusters = 600 kpc (measured path length)
    """
    # Galaxy RMS - weighted by errors
    galaxy_rms_values = []
    for gal in galaxies:
        L_eff = h_z_disk  # Disk scale height
        V_pred = predict_galaxy_velocity(
            gal['R'], gal['V_bar'], gal['R_d'], L_eff, L0, n
        )
        # RMS for this galaxy
        residuals = gal['V_obs'] - V_pred
        rms = np.sqrt(np.mean(residuals**2))
        galaxy_rms_values.append(rms)
    
    galaxy_rms = np.mean(galaxy_rms_values)
    
    # Cluster ratio
    ratios = []
    for cl in clusters:
        M_pred = predict_cluster_mass(cl['M_bar'], cl['r_kpc'], cl['L_eff'], L0, n)
        ratios.append(M_pred / cl['M_lens'])
    
    cluster_median = np.median(ratios) if ratios else 0
    cluster_scatter = np.std(np.log10(ratios)) if ratios else 0
    
    return galaxy_rms, cluster_median, cluster_scatter


def optimize_unified_parameters(galaxies, clusters):
    """
    Find optimal L0 and n for the unified model.
    
    Constraint: Galaxy amplitude A_gal = A₀ × (h_z/L₀)^n should give good SPARC fits
    """
    print("\n" + "="*70)
    print("OPTIMIZING UNIFIED 3D AMPLITUDE (NO D SWITCH)")
    print("="*70)
    
    # Test different disk scale heights
    for h_z in [0.2, 0.3, 0.4, 0.5]:
        print(f"\n--- Testing h_z = {h_z} kpc (disk scale height) ---")
        
        def objective(params):
            L0, n = params
            if L0 <= 0 or n <= 0 or n > 1:
                return 1e10
            
            gal_rms, cl_median, cl_scatter = evaluate_unified_model(
                galaxies, clusters, L0, n, h_z
            )
            
            # Want: gal_rms low, cl_median near 1.0
            cluster_penalty = 100 * (cl_median - 1.0)**2
            return gal_rms + cluster_penalty
        
        result = minimize(objective, [0.1, 0.3], method='Nelder-Mead',
                         options={'maxiter': 500})
        
        L0_opt, n_opt = result.x
        gal_rms, cl_median, cl_scatter = evaluate_unified_model(
            galaxies, clusters, L0_opt, n_opt, h_z
        )
        
        A_galaxy = unified_amplitude(h_z, L0_opt, n_opt)
        A_cluster = unified_amplitude(600, L0_opt, n_opt)
        
        print(f"  L₀ = {L0_opt:.3f} kpc, n = {n_opt:.3f}")
        print(f"  A_galaxy (L={h_z}) = {A_galaxy:.3f}")
        print(f"  A_cluster (L=600) = {A_cluster:.2f}")
        print(f"  Galaxy RMS = {gal_rms:.2f} km/s")
        print(f"  Cluster median = {cl_median:.3f}")


def test_path_length_formulations(galaxies, clusters):
    """
    Test different ways to compute L_eff for disk galaxies.
    """
    print("\n" + "="*70)
    print("TESTING PATH LENGTH FORMULATIONS FOR DISKS")
    print("="*70)
    
    # Current model (with D switch) for comparison
    print("\n--- Current Model (D=0/1 switch) ---")
    # For D=0: A = A₀ = 1.173
    # For D=1: A = A₀ × (L/L₀)^n = 8.45
    gal_rms_current, cl_median_current, _ = evaluate_unified_model(
        galaxies, clusters, L_0_current, n_current, h_z_disk=L_0_current
    )
    # Note: When h_z = L₀, then A_galaxy = A₀ × (L₀/L₀)^n = A₀
    print(f"  Galaxy RMS = {gal_rms_current:.2f} km/s")
    print(f"  Cluster median = {cl_median_current:.3f}")
    
    # Test: What if L_eff for galaxies = L₀?
    # This makes A_galaxy = A₀ × 1 = A₀, matching current behavior
    print("\n--- Unified: L_galaxy = L₀ (recovers current behavior) ---")
    gal_rms, cl_median, _ = evaluate_unified_model(
        galaxies, clusters, L_0_current, n_current, h_z_disk=L_0_current
    )
    print(f"  Galaxy RMS = {gal_rms:.2f} km/s")
    print(f"  Cluster median = {cl_median:.3f}")
    
    # Test: What if we use actual disk scale heights?
    print("\n--- Unified: L_galaxy = 0.3 kpc (typical scale height) ---")
    gal_rms, cl_median, _ = evaluate_unified_model(
        galaxies, clusters, L_0_current, n_current, h_z_disk=0.3
    )
    A_gal = unified_amplitude(0.3, L_0_current, n_current)
    A_cl = unified_amplitude(600, L_0_current, n_current)
    print(f"  A_galaxy = {A_gal:.3f}, A_cluster = {A_cl:.2f}")
    print(f"  Galaxy RMS = {gal_rms:.2f} km/s")
    print(f"  Cluster median = {cl_median:.3f}")


def analytical_check():
    """
    Check the math: Can A = A₀ × (L/L₀)^n work for both galaxies and clusters?
    """
    print("\n" + "="*70)
    print("ANALYTICAL CHECK: IS UNIFIED FORMULA POSSIBLE?")
    print("="*70)
    
    # Target values
    A_galaxy_target = A_0  # ≈ 1.173
    A_cluster_target = 8.45
    L_galaxy = 0.3  # kpc (disk scale height)
    L_cluster = 600  # kpc
    
    # For A = A₀ × (L/L₀)^n with L₀ = L_galaxy:
    # A(L_galaxy) = A₀ × 1 = A₀ ✓
    # A(L_cluster) = A₀ × (L_cluster/L_galaxy)^n = 8.45
    # So (600/0.3)^n = 8.45/1.173 = 7.2
    # 2000^n = 7.2
    # n = log(7.2)/log(2000) = 0.259
    
    ratio = A_cluster_target / A_galaxy_target
    L_ratio = L_cluster / L_galaxy
    n_required = np.log(ratio) / np.log(L_ratio)
    
    print(f"\nTarget: A_galaxy = {A_galaxy_target:.3f}, A_cluster = {A_cluster_target:.2f}")
    print(f"Path lengths: L_galaxy = {L_galaxy} kpc, L_cluster = {L_cluster} kpc")
    print(f"\nFor A = A₀ × (L/L₀)^n with L₀ = L_galaxy:")
    print(f"  Ratio A_cluster/A_galaxy = {ratio:.2f}")
    print(f"  Ratio L_cluster/L_galaxy = {L_ratio:.0f}")
    print(f"  Required n = log({ratio:.2f})/log({L_ratio:.0f}) = {n_required:.3f}")
    
    # Check: this should match our current n = 0.27!
    print(f"\n  Current n = 0.27, Required n = {n_required:.3f}")
    print(f"  Match: {'YES!' if abs(n_required - 0.27) < 0.02 else 'No'}")
    
    # The key insight
    print(f"""
KEY INSIGHT:
-----------
The unified formula A = A₀ × (L/L₀)^n DOES work if we set:
  - L₀ = L_galaxy ≈ 0.3 kpc (disk scale height)
  - n ≈ 0.26 (close to our current 0.27)

This means L₀ = 0.4 kpc in our current formula is approximately
the disk scale height, not an arbitrary calibration constant!

The "D switch" is just a shortcut for:
  - D=0: L = L₀ → A = A₀
  - D=1: L = 600 kpc → A = A₀ × (600/L₀)^n ≈ 8.45

We can eliminate the switch by using L directly!
""")
    
    # Test different L_galaxy values
    print("\nSensitivity to L_galaxy:")
    for L_gal in [0.2, 0.3, 0.4, 0.5, 1.0]:
        n_req = np.log(ratio) / np.log(L_cluster / L_gal)
        A_cl = A_0 * (L_cluster / L_gal) ** n_req
        print(f"  L_galaxy = {L_gal} kpc → n = {n_req:.3f}, A_cluster = {A_cl:.2f}")


def main():
    print("="*70)
    print("UNIFIED 3D AMPLITUDE TEST (ELIMINATING D SWITCH)")
    print("="*70)
    print("""
The goal: Replace the discrete D=0/1 switch with a unified formula
where the path length L naturally handles the galaxy/cluster difference.

For a thin disk: L ≈ scale height h_z ≈ 0.3-0.5 kpc
For a cluster: L ≈ 600 kpc

If A = A₀ × (L/L₀)^n works for both, we eliminate the switch!
""")
    
    # First, check the math
    analytical_check()
    
    # Load data
    print("Loading data...")
    galaxies = load_sparc_data()
    clusters = load_cluster_data()
    print(f"  Galaxies: {len(galaxies)}")
    print(f"  Clusters: {len(clusters)}")
    
    if not galaxies or not clusters:
        print("Error: Could not load data")
        return
    
    # Test formulations
    test_path_length_formulations(galaxies, clusters)
    
    # Optimize
    optimize_unified_parameters(galaxies, clusters)
    
    # Summary
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
Key insight: If we set L_galaxy = L₀, then A_galaxy = A₀ × (L₀/L₀)^n = A₀,
which recovers the current behavior WITHOUT a D switch!

The unified formula becomes:
    A(L) = A₀ × (L/L₀)^n

where:
    - L = L₀ ≈ 0.4 kpc for disk galaxies (sets the reference)
    - L ≈ 600 kpc for clusters (path through ICM)

This is mathematically equivalent to the current model but conceptually
cleaner: there's no "switch", just different path lengths for different
geometries.

For intermediate systems (ellipticals, S0s), L would be between these
extremes, providing a natural continuous transition.
""")


if __name__ == "__main__":
    main()

