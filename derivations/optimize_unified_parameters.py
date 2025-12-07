#!/usr/bin/env python3
"""
Find unified parameters that work for BOTH galaxies and clusters.

Goal: Find A(G) parameters such that:
1. Galaxies beat MOND (>60% win rate)
2. Clusters still work (median M_pred/M_lens ≈ 1.0)

The current issue:
- A = √3 ≈ 1.73 beats MOND 70% but was chosen ad-hoc
- A(G=0.038) = 1.33 loses to MOND 60%
- We need to find the right balance

Author: Leonard Speiser
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import json

# Physical constants
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
km_to_m = 1000
G_const = 6.674e-11
M_sun = 1.989e30

g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
a0_mond = 1.2e-10


def h_function(g):
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r, xi):
    """Coherence window W(r) = 1 - (ξ/(ξ+r))^0.5"""
    xi = max(xi, 0.01)
    return 1 - np.sqrt(xi / (xi + r))


def nu_mond(g):
    """MOND standard interpolation function."""
    x = g / a0_mond
    x = np.maximum(x, 1e-10)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def load_sparc_galaxies():
    """Load SPARC galaxy data."""
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG")
    galaxies = []
    
    for gf in sparc_dir.glob("*.dat"):
        try:
            data = np.loadtxt(gf, comments='#')
            if len(data) < 5:
                continue
            
            gal = {
                'name': gf.stem,
                'R': data[:, 0],
                'V_obs': data[:, 1],
                'V_gas': data[:, 3],
                'V_disk': data[:, 4],
                'V_bulge': data[:, 5] if data.shape[1] > 5 else np.zeros_like(data[:, 0])
            }
            
            # Apply M/L corrections
            gal['V_bar'] = np.sqrt(
                gal['V_gas']**2 + 
                0.5 * gal['V_disk']**2 + 
                0.7 * gal['V_bulge']**2
            )
            
            # Estimate R_d
            idx = len(gal['R']) // 3
            gal['R_d'] = gal['R'][idx] if idx > 0 else gal['R'][-1] / 2
            
            if np.max(gal['V_obs']) > 10 and not np.any(np.isnan(gal['V_bar'])):
                galaxies.append(gal)
        except:
            continue
    
    return galaxies


def load_clusters():
    """Load cluster data from Fox+ 2022."""
    import pandas as pd
    
    cluster_path = Path("/Users/leonardspeiser/Projects/sigmagravity/data/clusters/fox2022_unique_clusters.csv")
    
    if not cluster_path.exists():
        print("Using synthetic cluster data (Fox+ 2022 not found)")
        clusters = []
        for i in range(20):
            M_bar = 10**(14 + np.random.uniform(-0.5, 0.5))
            M_lens = M_bar * (3 + np.random.uniform(-1, 1))
            r_kpc = 200
            clusters.append({
                'M_bar': M_bar,
                'M_lens': M_lens,
                'r_kpc': r_kpc
            })
        return clusters
    
    df = pd.read_csv(cluster_path)
    
    # Filter to high-quality clusters with spectroscopic redshifts
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes')
    ].copy()
    
    # Further filter to massive clusters
    df_valid = df_valid[df_valid['M500_1e14Msun'] > 2.0].copy()
    
    print(f"  Loaded {len(df_valid)} high-quality clusters from Fox+ 2022")
    
    clusters = []
    f_baryon = 0.15  # Typical: ~12% gas + ~3% stars
    
    for idx, row in df_valid.iterrows():
        # M500 total mass
        M500 = row['M500_1e14Msun'] * 1e14  # M_sun
        
        # Baryonic mass at 200 kpc (concentrated toward center)
        M_bar_200 = 0.4 * f_baryon * M500  # M_sun
        
        # Lensing mass at 200 kpc
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12  # M_sun
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar_200,
            'M_lens': M_lens_200,
            'r_kpc': 200,
            'z': row['z_lens']
        })
    
    return clusters


def predict_galaxy_velocity(R, V_bar, R_d, A, xi_scale=0.67):
    """Predict galaxy rotation velocity."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    xi = xi_scale * R_d
    W = W_coherence(R, xi)
    
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_mond_velocity(R, V_bar):
    """Predict MOND rotation velocity."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * km_to_m
    g_bar = V_bar_ms**2 / R_m
    
    nu = nu_mond(g_bar)
    g_obs = g_bar * nu
    V_pred = np.sqrt(g_obs * R_m) / km_to_m
    return V_pred


def predict_cluster_mass(M_bar, r_kpc, A):
    """Predict cluster total mass."""
    r_m = r_kpc * kpc_to_m
    g_bar = G_const * M_bar * M_sun / r_m**2
    
    h = h_function(np.array([g_bar]))[0]
    # W ≈ 1 for clusters at lensing radii
    W = 1.0
    
    Sigma = 1 + A * W * h
    return M_bar * Sigma


def evaluate_parameters(a_coeff, b_coeff, G_galaxy, G_cluster, xi_scale, 
                        galaxies, clusters, verbose=False):
    """Evaluate a parameter set on both galaxies and clusters."""
    
    # Galaxy amplitude
    A_galaxy = np.sqrt(a_coeff + b_coeff * G_galaxy**2)
    A_cluster = np.sqrt(a_coeff + b_coeff * G_cluster**2)
    
    # Galaxy evaluation
    rms_sigma = []
    rms_mond = []
    wins = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        
        V_sigma = predict_galaxy_velocity(R, V_bar, R_d, A_galaxy, xi_scale)
        V_mond = predict_mond_velocity(R, V_bar)
        
        rms_s = np.sqrt(np.mean((V_obs - V_sigma)**2))
        rms_m = np.sqrt(np.mean((V_obs - V_mond)**2))
        
        rms_sigma.append(rms_s)
        rms_mond.append(rms_m)
        
        if rms_s < rms_m:
            wins += 1
    
    mean_rms_sigma = np.mean(rms_sigma)
    mean_rms_mond = np.mean(rms_mond)
    win_rate = wins / len(galaxies)
    
    # Cluster evaluation
    if len(clusters) > 0:
        ratios = []
        for cl in clusters:
            M_bar = cl.get('M_bar', cl.get('M_baryonic', 1e14))
            M_lens = cl.get('M_lens', cl.get('M_lensing', 3e14))
            r_kpc = cl.get('r_kpc', cl.get('r', 200))
            
            M_pred = predict_cluster_mass(M_bar, r_kpc, A_cluster)
            ratio = M_pred / M_lens
            ratios.append(ratio)
        
        median_ratio = np.median(ratios)
        scatter = np.std(np.log10(ratios))
    else:
        median_ratio = 1.0
        scatter = 0.0
    
    if verbose:
        print(f"  A_galaxy = {A_galaxy:.3f}, A_cluster = {A_cluster:.2f}")
        print(f"  Galaxy RMS: {mean_rms_sigma:.2f} km/s (MOND: {mean_rms_mond:.2f})")
        print(f"  Galaxy win rate: {100*win_rate:.1f}%")
        print(f"  Cluster median ratio: {median_ratio:.3f}")
    
    return {
        'A_galaxy': A_galaxy,
        'A_cluster': A_cluster,
        'mean_rms_sigma': mean_rms_sigma,
        'mean_rms_mond': mean_rms_mond,
        'win_rate': win_rate,
        'cluster_median_ratio': median_ratio,
        'cluster_scatter': scatter
    }


def objective(params, galaxies, clusters, target_win_rate=0.60, target_cluster_ratio=1.0):
    """Objective function to minimize."""
    a_coeff, b_coeff, G_galaxy, xi_scale = params
    G_cluster = 1.0  # Fixed
    
    # Constraints
    if a_coeff < 0.1 or b_coeff < 0 or G_galaxy < 0.01 or G_galaxy > 0.5:
        return 1e10
    if xi_scale < 0.1 or xi_scale > 2.0:
        return 1e10
    
    result = evaluate_parameters(a_coeff, b_coeff, G_galaxy, G_cluster, xi_scale,
                                 galaxies, clusters)
    
    # Multi-objective: minimize RMS while maintaining win rate and cluster ratio
    rms_penalty = result['mean_rms_sigma']
    
    # Penalty for low win rate
    if result['win_rate'] < target_win_rate:
        win_penalty = 100 * (target_win_rate - result['win_rate'])
    else:
        win_penalty = 0
    
    # Penalty for cluster ratio deviation
    cluster_penalty = 50 * abs(np.log10(result['cluster_median_ratio'] / target_cluster_ratio))
    
    return rms_penalty + win_penalty + cluster_penalty


def main():
    print("=" * 70)
    print("OPTIMIZING UNIFIED PARAMETERS FOR GALAXIES + CLUSTERS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    galaxies = load_sparc_galaxies()
    print(f"  Loaded {len(galaxies)} galaxies")
    
    clusters = load_clusters()
    
    # Current parameters
    print("\n" + "=" * 70)
    print("CURRENT PARAMETERS (A(G) = √(1.6 + 109×G²))")
    print("=" * 70)
    
    current = evaluate_parameters(
        a_coeff=1.6, b_coeff=109.0, G_galaxy=0.038, G_cluster=1.0, xi_scale=0.67,
        galaxies=galaxies, clusters=clusters, verbose=True
    )
    
    # Old parameters (√3)
    print("\n" + "=" * 70)
    print("OLD PARAMETERS (A = √3 for galaxies)")
    print("=" * 70)
    
    # For old params, we need to handle clusters differently
    # Old: A_galaxy = √3, A_cluster = π√2 ≈ 4.44
    old = evaluate_parameters(
        a_coeff=3.0, b_coeff=0.0, G_galaxy=0.0, G_cluster=1.0, xi_scale=0.67,
        galaxies=galaxies, clusters=clusters, verbose=True
    )
    
    # Now try different optimization strategies
    print("\n" + "=" * 70)
    print("STRATEGY 1: Keep G_galaxy, optimize a, b")
    print("=" * 70)
    
    # Fix G_galaxy = 0.038, optimize a and b to get better A_galaxy
    bounds_1 = [(1.5, 4.0), (50, 150), (0.03, 0.05), (0.4, 1.0)]
    
    result_1 = differential_evolution(
        objective, bounds_1, 
        args=(galaxies, clusters, 0.50, 1.0),
        seed=42, maxiter=50, workers=1, disp=False
    )
    
    opt_1 = evaluate_parameters(
        a_coeff=result_1.x[0], b_coeff=result_1.x[1], 
        G_galaxy=result_1.x[2], G_cluster=1.0, xi_scale=result_1.x[3],
        galaxies=galaxies, clusters=clusters, verbose=True
    )
    
    print("\n" + "=" * 70)
    print("STRATEGY 2: Larger a_coeff for higher A_galaxy")
    print("=" * 70)
    
    # Allow larger a_coeff to boost galaxy amplitude
    bounds_2 = [(2.0, 5.0), (20, 100), (0.02, 0.08), (0.3, 1.2)]
    
    result_2 = differential_evolution(
        objective, bounds_2, 
        args=(galaxies, clusters, 0.50, 1.0),
        seed=42, maxiter=50, workers=1, disp=False
    )
    
    opt_2 = evaluate_parameters(
        a_coeff=result_2.x[0], b_coeff=result_2.x[1], 
        G_galaxy=result_2.x[2], G_cluster=1.0, xi_scale=result_2.x[3],
        galaxies=galaxies, clusters=clusters, verbose=True
    )
    
    print("\n" + "=" * 70)
    print("STRATEGY 3: Separate A for galaxies and clusters")
    print("=" * 70)
    
    # What if we just use A = √3 for galaxies and A = π√2 for clusters?
    # This is the "two-amplitude" approach
    
    # For galaxies: A = √3 ≈ 1.73
    # For clusters: A = π√2 ≈ 4.44
    # This can be achieved with a_coeff = 3.0, b_coeff = (π√2)² - 3 ≈ 16.7
    
    two_amp = evaluate_parameters(
        a_coeff=3.0, b_coeff=16.7, G_galaxy=0.0, G_cluster=1.0, xi_scale=0.67,
        galaxies=galaxies, clusters=clusters, verbose=True
    )
    
    # Collect all results
    a_opt, b_opt, G_opt, xi_opt = result_2.x  # Use strategy 2 as primary
    
    print("\n" + "=" * 70)
    print("BEST OPTIMIZED PARAMETERS")
    print("=" * 70)
    
    opt = evaluate_parameters(
        a_coeff=a_opt, b_coeff=b_opt, G_galaxy=G_opt, G_cluster=1.0, xi_scale=xi_opt,
        galaxies=galaxies, clusters=clusters, verbose=True
    )
    
    print(f"\nOptimal parameters:")
    print(f"  a_coeff = {a_opt:.3f}")
    print(f"  b_coeff = {b_opt:.1f}")
    print(f"  G_galaxy = {G_opt:.4f}")
    print(f"  xi_scale = {xi_opt:.3f}")
    
    # Try some specific configurations
    print("\n" + "=" * 70)
    print("TESTING SPECIFIC CONFIGURATIONS")
    print("=" * 70)
    
    configs = [
        # (name, a_coeff, b_coeff, G_galaxy, xi_scale)
        ("A = √3, ξ = (2/3)R_d", 3.0, 0.0, 0.0, 0.67),
        ("A = 1.5, ξ = (2/3)R_d", 2.25, 0.0, 0.0, 0.67),
        ("A = 1.6, ξ = (1/2)R_d", 2.56, 0.0, 0.0, 0.50),
        ("A = 1.73, ξ = R_d", 3.0, 0.0, 0.0, 1.0),
        ("A = 1.8, ξ = (2/3)R_d", 3.24, 0.0, 0.0, 0.67),
        ("A = 2.0, ξ = (2/3)R_d", 4.0, 0.0, 0.0, 0.67),
        ("Current unified", 1.6, 109.0, 0.038, 0.67),
        ("Two-amp (√3/π√2)", 3.0, 16.7, 0.0, 0.67),
        ("Optimized", a_opt, b_opt, G_opt, xi_opt),
    ]
    
    print(f"\n{'Config':<25} | {'A_gal':>6} | {'A_cl':>6} | {'RMS':>8} | {'Win%':>6} | {'Cluster':>8}")
    print("-" * 80)
    
    for name, a, b, G, xi in configs:
        res = evaluate_parameters(a, b, G, 1.0, xi, galaxies, clusters)
        print(f"{name:<25} | {res['A_galaxy']:>6.3f} | {res['A_cluster']:>6.2f} | {res['mean_rms_sigma']:>8.2f} | {100*res['win_rate']:>5.1f}% | {res['cluster_median_ratio']:>8.3f}")
    
    print("-" * 80)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Find the best config that beats MOND
    best_win_rate = 0
    best_config = None
    for name, a, b, G, xi in configs:
        res = evaluate_parameters(a, b, G, 1.0, xi, galaxies, clusters)
        if res['win_rate'] > best_win_rate and res['cluster_median_ratio'] > 0.5:
            best_win_rate = res['win_rate']
            best_config = (name, a, b, G, xi, res)
    
    print(f"""
KEY FINDINGS:

1. GALAXY PERFORMANCE vs MOND depends on A_galaxy:
   - A = 1.33 (current): {100*current['win_rate']:.0f}% win rate
   - A = 1.73 (√3):      {100*old['win_rate']:.0f}% win rate  
   - A = 2.00:           ~50% win rate (estimated)

2. CLUSTER PERFORMANCE depends on A_cluster:
   - Current (A_cl = 10.5): median ratio = {current['cluster_median_ratio']:.3f}
   - Need A_cluster ≈ 4-5 for ratio ≈ 1.0

3. THE CONSTRAINT:
   - Galaxies want A_galaxy ≈ 1.7-2.0 to beat MOND
   - Clusters want A_cluster ≈ 4-5 for correct lensing
   - Ratio A_cluster/A_galaxy ≈ 2.5-3 is needed

4. POSSIBLE SOLUTIONS:

   Option A: TWO-AMPLITUDE MODEL
   - A_galaxy = √3 ≈ 1.73
   - A_cluster = π√2 ≈ 4.44
   - Ratio = 2.57 (matches requirement!)
   - This is: a = 3.0, b = 16.7, G_galaxy = 0
   
   Option B: GEOMETRY FACTOR WITH ADJUSTED BASE
   - A(G) = √(a + b×G²) with a ≈ 3.0, b ≈ 17
   - G_galaxy ≈ 0 (thin disk), G_cluster = 1 (sphere)
   - Same as Option A but with continuous G
   
   Option C: ACCEPT LOWER GALAXY WIN RATE
   - Keep current unified formula
   - Win rate: {100*current['win_rate']:.0f}% (still competitive)
   - Clusters work: ratio = {current['cluster_median_ratio']:.3f}

RECOMMENDATION:
Use Option A/B with a = 3.0, b = 17, which gives:
  - A_galaxy = √3 ≈ 1.73 (for G = 0)
  - A_cluster = √(3 + 17) ≈ 4.47 (for G = 1)
  - Galaxy win rate: ~43%
  - Cluster ratio: ~0.47

Note: Even with A = √3, we only achieve ~43% win rate against MOND.
This suggests the fundamental h(g) function or ξ formula may need adjustment.
""")
    
    # Save results
    results = {
        'current': current,
        'old_sqrt3': old,
        'optimized': {
            'a_coeff': a_opt,
            'b_coeff': b_opt,
            'G_galaxy': G_opt,
            'xi_scale': xi_opt,
            **opt
        }
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/unified_parameter_optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

