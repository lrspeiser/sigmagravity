#!/usr/bin/env python3
"""
Theory Discovery Runner for Σ-Gravity

Runs symbolic regression on real astronomical data to discover:
1. Baryonic Tully-Fisher relation (v^4 ∝ M)
2. Radial Acceleration Relation (RAR)
3. Enhancement factor K(R) - the core Σ-Gravity prediction
4. Cluster density profiles

Usage:
    python run_discovery.py                    # Run all discoveries
    python run_discovery.py --target btf       # Baryonic Tully-Fisher only
    python run_discovery.py --target rar       # RAR only  
    python run_discovery.py --target K         # Enhancement factor K
    python run_discovery.py --target cluster   # Cluster profiles
"""

import sys
import argparse
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from data_loaders import (
    load_sparc_galaxies,
    load_sparc_rar,
    load_gaia_rar,
    load_gaia_enhancement_factor,
    load_cluster_profiles,
)
from theory_discovery_v3 import GPUTheoryDiscoveryEngine, GPU_AVAILABLE


def discover_baryonic_tully_fisher(seed: int = 42):
    """
    Discover the Baryonic Tully-Fisher Relation: v^4 ∝ M_baryon
    
    Using log scale: log(v) ∝ 0.25 * log(M_baryon) = 0.25 * M
    (since M in our data is already log10(M_baryon))
    """
    print("\n" + "="*70)
    print("  DISCOVERY: BARYONIC TULLY-FISHER RELATION")
    print("="*70)
    print("  Expected: log(v) ∝ 0.25 × M  (BTF: v⁴ ∝ M_baryon)")
    print()
    
    # Load all galaxies
    X, y, meta = load_sparc_galaxies(n_galaxies=None, seed=seed)
    
    # Target: log(v_flat) - should be ∝ 0.25 * M
    y_target = np.log10(y)
    
    engine = GPUTheoryDiscoveryEngine(
        variables=list(X.keys()),
        population_size=2000,
        max_generations=80,
        max_depth=4,
        complexity_penalty=0.008,
        seed=seed
    )
    
    best_prog, mse, r2 = engine.evolve(X, y_target, verbose=True)
    
    expr = best_prog.to_string(list(X.keys()))
    
    print()
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCOVERED: log(v) = {expr[:43]:<43} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  R² = {r2:.6f}                                              ║")
    print(f"  ║  Expected: log(v) ≈ 0.25 × M  (where M = log10(M_baryon))   ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    
    return {'target': 'BTF', 'discovered': expr, 'r_squared': r2}


def discover_rar_gaia(n_stars: int = 50000, seed: int = 42):
    """
    Discover the Radial Acceleration Relation from Gaia MW stars.
    
    Target: g_obs = f(g_bar, R, z, ...)
    """
    print("\n" + "="*70)
    print("  DISCOVERY: RADIAL ACCELERATION RELATION (GAIA MW)")
    print("="*70)
    print("  Expected: g_obs = g_bar × ν(g_bar/a₀)  or  g_obs = g_bar × [1+K(R)]")
    print()
    
    X, y, meta = load_gaia_rar(max_stars=n_stars, seed=seed)
    
    # Use only g_bar and R for cleaner discovery
    X_simple = {
        'g': X['g_bar'],  # log10(g_bar)
        'R': X['R'],
    }
    
    engine = GPUTheoryDiscoveryEngine(
        variables=list(X_simple.keys()),
        population_size=2000,
        max_generations=80,
        max_depth=5,
        complexity_penalty=0.008,
        seed=seed
    )
    
    best_prog, mse, r2 = engine.evolve(X_simple, y, verbose=True)
    
    expr = best_prog.to_string(list(X_simple.keys()))
    
    print()
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCOVERED: log(g_obs) = {expr[:40]:<40} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  R² = {r2:.6f}                                              ║")
    print(f"  ║  Variables: g = log10(g_bar), R = radius (kpc)              ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    
    return {'target': 'RAR', 'discovered': expr, 'r_squared': r2}


def discover_enhancement_factor(n_stars: int = 50000, seed: int = 42):
    """
    Discover the gravitational enhancement factor K(R).
    
    This is the CORE Σ-Gravity prediction:
    K = g_obs/g_bar - 1 = f(R, g_bar, ...)
    
    The theory predicts K follows a coherence window (Burr-XII distribution).
    """
    print("\n" + "="*70)
    print("  DISCOVERY: ENHANCEMENT FACTOR K(R) - Σ-GRAVITY CORE")
    print("="*70)
    print("  Target: K = g_obs/g_bar - 1")
    print("  Σ-Gravity predicts: K ~ Burr-XII coherence window")
    print()
    
    X, K, meta = load_gaia_enhancement_factor(max_stars=n_stars, seed=seed)
    
    # Focus on radius dependence
    X_simple = {
        'R': X['R'],
        'g': X['g_bar'],
    }
    
    engine = GPUTheoryDiscoveryEngine(
        variables=list(X_simple.keys()),
        population_size=2500,
        max_generations=100,
        max_depth=5,
        complexity_penalty=0.006,
        seed=seed
    )
    
    best_prog, mse, r2 = engine.evolve(X_simple, K, verbose=True)
    
    expr = best_prog.to_string(list(X_simple.keys()))
    
    # Compute mean K for reference
    K_mean = np.mean(K)
    K_std = np.std(K)
    
    print()
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCOVERED: K = {expr[:48]:<48} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  R² = {r2:.6f}                                              ║")
    print(f"  ║  K_mean = {K_mean:.3f} ± {K_std:.3f}                                     ║")
    print(f"  ║  Variables: R = radius (kpc), g = log10(g_bar)              ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    
    # Physical interpretation
    print()
    print("  PHYSICAL INTERPRETATION:")
    print("  ─────────────────────────")
    print(f"  • K > 0 means observed gravity exceeds baryonic prediction")
    print(f"  • Mean enhancement: {K_mean:.1f}× (gravity is {K_mean+1:.1f}× stronger than expected)")
    print(f"  • This is the 'missing mass' that Σ-Gravity explains via coherence")
    
    return {'target': 'K', 'discovered': expr, 'r_squared': r2, 'K_mean': K_mean}


def discover_cluster_density(seed: int = 42):
    """
    Discover cluster density profile: n_e(R)
    
    Expected: Beta-model n_e ∝ (1 + (r/r_c)²)^(-3β/2)
    """
    print("\n" + "="*70)
    print("  DISCOVERY: CLUSTER DENSITY PROFILE")
    print("="*70)
    print("  Expected: β-model: n_e ∝ (1 + (r/r_c)²)^(-3β/2)")
    print()
    
    X, y, meta = load_cluster_profiles()
    
    if len(y) == 0:
        print("  No cluster data available")
        return None
    
    engine = GPUTheoryDiscoveryEngine(
        variables=list(X.keys()),
        population_size=1500,
        max_generations=60,
        max_depth=5,
        complexity_penalty=0.01,
        seed=seed
    )
    
    best_prog, mse, r2 = engine.evolve(X, y, verbose=True)
    
    expr = best_prog.to_string(list(X.keys()))
    
    print()
    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  DISCOVERED: log(n_e) = {expr[:42]:<42} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  R² = {r2:.6f}                                              ║")
    print(f"  ║  Clusters: {str(meta.get('clusters', []))[:45]:<45} ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    
    return {'target': 'cluster', 'discovered': expr, 'r_squared': r2}


def run_all_discoveries(seed: int = 42):
    """Run all discovery targets and summarize results."""
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       Σ-GRAVITY THEORY DISCOVERY ENGINE                            ║")
    print("║       Symbolic Regression on Real Astronomical Data                ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Hardware: {'GPU-Accelerated' if GPU_AVAILABLE else 'CPU only'}")
    print(f"  Seed: {seed}")
    
    total_start = time.time()
    results = []
    
    # 1. Baryonic Tully-Fisher
    try:
        results.append(discover_baryonic_tully_fisher(seed=seed))
    except Exception as e:
        print(f"  BTF discovery failed: {e}")
    
    # 2. RAR from Gaia
    try:
        results.append(discover_rar_gaia(n_stars=30000, seed=seed))
    except Exception as e:
        print(f"  RAR discovery failed: {e}")
    
    # 3. Enhancement factor K (MOST IMPORTANT for Σ-Gravity)
    try:
        results.append(discover_enhancement_factor(n_stars=50000, seed=seed))
    except Exception as e:
        print(f"  K discovery failed: {e}")
    
    # 4. Cluster profiles
    try:
        result = discover_cluster_density(seed=seed)
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Cluster discovery failed: {e}")
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("  DISCOVERY SUMMARY")
    print("="*70)
    
    for r in results:
        if r:
            print(f"  • {r['target']:12s} | R² = {r['r_squared']:.4f} | {r['discovered'][:40]}")
    
    print()
    print(f"  Total time: {total_time:.1f}s")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Σ-Gravity Theory Discovery")
    parser.add_argument('--target', choices=['btf', 'rar', 'K', 'cluster', 'all'],
                       default='all', help='Discovery target')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--stars', type=int, default=30000, 
                       help='Number of Gaia stars to use')
    
    args = parser.parse_args()
    
    if args.target == 'all':
        run_all_discoveries(seed=args.seed)
    elif args.target == 'btf':
        discover_baryonic_tully_fisher(seed=args.seed)
    elif args.target == 'rar':
        discover_rar_gaia(n_stars=args.stars, seed=args.seed)
    elif args.target == 'K':
        discover_enhancement_factor(n_stars=args.stars, seed=args.seed)
    elif args.target == 'cluster':
        discover_cluster_density(seed=args.seed)


if __name__ == "__main__":
    main()
