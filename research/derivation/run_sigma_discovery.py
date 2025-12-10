#!/usr/bin/env python3
"""
Σ-GRAVITY FORMULA DISCOVERY - Main Runner
==========================================

Run the symbolic regression engine to discover Σ-Gravity formulas
from real astronomical data.

Data Sources:
- SPARC: 166 galaxy rotation curves
- Gaia: 1.8M Milky Way stars
- Clusters: 3 galaxy clusters with gas profiles

Targets:
- K: Enhancement factor K(R) = g_obs/g_bar - 1
- rar: Radial Acceleration Relation g_obs = f(g_bar)
- btf: Baryonic Tully-Fisher v = f(M)

Usage:
    python run_sigma_discovery.py --target K --stars 50000
    python run_sigma_discovery.py --target rar --exhaustive
    python run_sigma_discovery.py --target btf
    python run_sigma_discovery.py --all

Author: Σ-Gravity Research Team
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Import discovery engine
from sigma_discovery_evolution import (
    SigmaGravityDiscoveryEngine,
    DiscoveryResult,
)

# Import data loaders
from data_loaders import (
    load_sparc_galaxies,
    load_sparc_rar,
    load_gaia_rar,
    load_gaia_enhancement_factor,
    load_cluster_profiles,
    load_multi_scale_rar,
)


# =============================================================================
# DISCOVERY TARGETS
# =============================================================================

def discover_enhancement_factor(
    max_stars: int = 50000,
    max_generations: int = 150,
    population_size: int = 3000,
    exhaustive: bool = False,
    seed: int = None,
) -> dict:
    """
    Discover the enhancement factor K(R).
    
    Target: K = g_obs/g_bar - 1
    
    This is the core Σ-Gravity relationship to discover.
    The theory predicts K follows a coherence window function.
    """
    print("\n" + "="*70)
    print("DISCOVERY TARGET: Enhancement Factor K(R)")
    print("="*70)
    
    # Load Gaia data
    print("\n[1/3] Loading Gaia MW data...")
    try:
        X, y, meta = load_gaia_enhancement_factor(max_stars=max_stars, seed=seed or 42)
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        print("  Falling back to SPARC RAR data...")
        X_rar, y_rar, meta = load_sparc_rar(max_points=5000, seed=seed or 42)
        
        # Compute K from RAR
        log_ratio = y_rar - X_rar['g_bar']
        K = 10**log_ratio - 1
        valid = (K > -0.5) & (K < 100) & np.isfinite(K)
        
        X = {'R': X_rar['R'][valid], 'g_bar': X_rar['g_bar'][valid]}
        y = K[valid]
        meta['source'] = 'SPARC RAR → K'
        meta['n_stars'] = len(y)
    
    print(f"  N = {len(y):,} data points")
    print(f"  R range: [{X['R'].min():.2f}, {X['R'].max():.2f}] kpc")
    print(f"  K range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  K median: {np.median(y):.3f}")
    
    # Variable order: R first, then g_bar
    var_names = ['R', 'g_bar']
    X_ordered = {'R': X['R'], 'g_bar': X['g_bar']}
    
    # Initialize engine
    print("\n[2/3] Initializing discovery engine...")
    engine = SigmaGravityDiscoveryEngine(
        var_names=var_names,
        population_size=population_size,
        max_generations=max_generations,
        num_islands=6,
        physics_mode='exhaustive' if exhaustive else 'gravity',
        use_templates=True,
        exhaustive_depth=3,
        seed=seed,
    )
    
    # Run discovery
    print("\n[3/3] Running evolutionary search...")
    best_expr, mse, r2 = engine.evolve(X_ordered, y, target='K', verbose=True)
    
    # Results
    print("\n" + "-"*70)
    print("RESULTS: Enhancement Factor K(R)")
    print("-"*70)
    
    formula = best_expr.to_string(var_names)
    print(f"\n  BEST FORMULA: {formula}")
    print(f"  R² = {r2:.6f}")
    print(f"  MSE = {mse:.6e}")
    print(f"  Complexity = {best_expr.complexity()}")
    
    # Show Pareto front
    print("\n  Pareto Front (top 10):")
    for i, result in enumerate(engine.get_pareto_front()[:10]):
        print(f"    {i+1}. R²={result.r_squared:.5f} | C={result.complexity:2d} | {result.formula_string[:40]}")
    
    # Export results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"K_discovery_{timestamp}.json"
    engine.export_results(str(output_file))
    
    return {
        'target': 'K',
        'best_formula': formula,
        'r_squared': r2,
        'mse': mse,
        'n_data': len(y),
        'pareto_front': [r.to_dict() for r in engine.get_pareto_front()[:20]],
    }


def discover_rar(
    max_points: int = 10000,
    max_generations: int = 100,
    population_size: int = 2000,
    exhaustive: bool = False,
    seed: int = None,
) -> dict:
    """
    Discover the Radial Acceleration Relation.
    
    Target: log(g_obs) = f(log(g_bar), R)
    
    Standard MOND predicts: g_obs = g_bar * ν(g_bar/a0)
    Σ-Gravity predicts: g_obs = g_bar * [1 + K(R)]
    """
    print("\n" + "="*70)
    print("DISCOVERY TARGET: Radial Acceleration Relation")
    print("="*70)
    
    # Load data - try Gaia first, then SPARC
    print("\n[1/3] Loading RAR data...")
    try:
        X, y, meta = load_gaia_rar(max_stars=max_points, seed=seed or 42)
    except FileNotFoundError:
        print("  Gaia data not found, using SPARC...")
        X, y, meta = load_sparc_rar(max_points=max_points, seed=seed or 42)
    
    print(f"  N = {len(y):,} data points")
    print(f"  g_bar range: [{X['g_bar'].min():.2f}, {X['g_bar'].max():.2f}]")
    print(f"  g_obs range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Variable order: g_bar first, then R
    var_names = ['g_bar', 'R']
    X_ordered = {'g_bar': X['g_bar'], 'R': X['R']}
    
    # Initialize engine
    print("\n[2/3] Initializing discovery engine...")
    engine = SigmaGravityDiscoveryEngine(
        var_names=var_names,
        population_size=population_size,
        max_generations=max_generations,
        num_islands=4,
        physics_mode='exhaustive' if exhaustive else 'gravity',
        use_templates=True,
        exhaustive_depth=2,
        seed=seed,
    )
    
    # Run discovery
    print("\n[3/3] Running evolutionary search...")
    best_expr, mse, r2 = engine.evolve(X_ordered, y, target='rar', verbose=True)
    
    # Results
    print("\n" + "-"*70)
    print("RESULTS: Radial Acceleration Relation")
    print("-"*70)
    
    formula = best_expr.to_string(var_names)
    print(f"\n  BEST FORMULA: log(g_obs) = {formula}")
    print(f"  R² = {r2:.6f}")
    print(f"  MSE = {mse:.6e}")
    
    # Show Pareto front
    print("\n  Pareto Front (top 10):")
    for i, result in enumerate(engine.get_pareto_front()[:10]):
        print(f"    {i+1}. R²={result.r_squared:.5f} | C={result.complexity:2d} | {result.formula_string[:40]}")
    
    # Export results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"RAR_discovery_{timestamp}.json"
    engine.export_results(str(output_file))
    
    return {
        'target': 'RAR',
        'best_formula': formula,
        'r_squared': r2,
        'mse': mse,
        'n_data': len(y),
    }


def discover_btf(
    max_generations: int = 100,
    population_size: int = 1500,
    exhaustive: bool = False,
    seed: int = None,
) -> dict:
    """
    Discover the Baryonic Tully-Fisher Relation.
    
    Target: v_flat = f(M_baryon)
    
    Standard BTF: v^4 = A * M_baryon (v ∝ M^0.25)
    """
    print("\n" + "="*70)
    print("DISCOVERY TARGET: Baryonic Tully-Fisher Relation")
    print("="*70)
    
    # Load SPARC galaxy data
    print("\n[1/3] Loading SPARC galaxy data...")
    X, y, meta = load_sparc_galaxies()
    
    print(f"  N = {len(y)} galaxies")
    print(f"  M_baryon range: [10^{X['M'].min():.1f}, 10^{X['M'].max():.1f}] M_sun")
    print(f"  v_flat range: [{y.min():.1f}, {y.max():.1f}] km/s")
    
    # Use log velocity for fitting (makes it linear in log-log space)
    y_log = np.log10(y)
    
    # Variable order: M first
    var_names = ['M', 'R', 'M_star', 'M_gas', 'f_b']
    X_ordered = {k: X[k] for k in var_names}
    
    # Initialize engine
    print("\n[2/3] Initializing discovery engine...")
    engine = SigmaGravityDiscoveryEngine(
        var_names=var_names,
        population_size=population_size,
        max_generations=max_generations,
        num_islands=4,
        physics_mode='exhaustive' if exhaustive else 'gravity',
        use_templates=True,
        exhaustive_depth=2,
        seed=seed,
    )
    
    # Run discovery
    print("\n[3/3] Running evolutionary search...")
    best_expr, mse, r2 = engine.evolve(X_ordered, y_log, target='btf', verbose=True)
    
    # Results
    print("\n" + "-"*70)
    print("RESULTS: Baryonic Tully-Fisher Relation")
    print("-"*70)
    
    formula = best_expr.to_string(var_names)
    print(f"\n  BEST FORMULA: log(v_flat) = {formula}")
    print(f"  R² = {r2:.6f}")
    print(f"  MSE = {mse:.6e}")
    
    # Check if it recovers BTF slope (0.25)
    print("\n  Standard BTF: log(v) = 0.25 × log(M) + const")
    
    # Show Pareto front
    print("\n  Pareto Front (top 10):")
    for i, result in enumerate(engine.get_pareto_front()[:10]):
        print(f"    {i+1}. R²={result.r_squared:.5f} | C={result.complexity:2d} | {result.formula_string[:40]}")
    
    # Export results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"BTF_discovery_{timestamp}.json"
    engine.export_results(str(output_file))
    
    return {
        'target': 'BTF',
        'best_formula': formula,
        'r_squared': r2,
        'mse': mse,
        'n_data': len(y),
    }


def discover_all(args):
    """Run discovery on all targets"""
    results = {}
    
    print("\n" + "#"*70)
    print("#  Σ-GRAVITY COMPREHENSIVE FORMULA DISCOVERY")
    print("#"*70)
    
    start_time = time.time()
    
    # 1. Enhancement factor K
    try:
        results['K'] = discover_enhancement_factor(
            max_stars=args.stars,
            max_generations=args.generations,
            exhaustive=args.exhaustive,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n  ✗ K discovery failed: {e}")
        results['K'] = {'error': str(e)}
    
    # 2. RAR
    try:
        results['RAR'] = discover_rar(
            max_points=args.stars,
            max_generations=args.generations,
            exhaustive=args.exhaustive,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n  ✗ RAR discovery failed: {e}")
        results['RAR'] = {'error': str(e)}
    
    # 3. BTF
    try:
        results['BTF'] = discover_btf(
            max_generations=args.generations,
            exhaustive=args.exhaustive,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n  ✗ BTF discovery failed: {e}")
        results['BTF'] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "#"*70)
    print("#  SUMMARY")
    print("#"*70)
    
    for target, res in results.items():
        if 'error' in res:
            print(f"\n  {target}: FAILED - {res['error']}")
        else:
            print(f"\n  {target}:")
            print(f"    Formula: {res['best_formula'][:50]}")
            print(f"    R² = {res['r_squared']:.6f}")
    
    print(f"\n  Total time: {total_time:.1f}s")
    
    # Export combined results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ALL_discovery_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Combined results: {output_file}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Σ-Gravity Formula Discovery Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sigma_discovery.py --target K --stars 50000
  python run_sigma_discovery.py --target rar --exhaustive
  python run_sigma_discovery.py --target btf --generations 200
  python run_sigma_discovery.py --all
        """
    )
    
    parser.add_argument(
        '--target', '-t',
        choices=['K', 'rar', 'btf', 'all'],
        default='K',
        help='Discovery target (default: K)'
    )
    
    parser.add_argument(
        '--stars', '-n',
        type=int,
        default=30000,
        help='Number of stars/points to use (default: 30000)'
    )
    
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=150,
        help='Max generations (default: 150)'
    )
    
    parser.add_argument(
        '--population', '-p',
        type=int,
        default=3000,
        help='Population size (default: 3000)'
    )
    
    parser.add_argument(
        '--exhaustive', '-e',
        action='store_true',
        help='Use exhaustive search mode'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all discovery targets'
    )
    
    args = parser.parse_args()
    
    # Check for --all flag
    if args.all or args.target == 'all':
        discover_all(args)
        return
    
    # Run specific target
    if args.target == 'K':
        discover_enhancement_factor(
            max_stars=args.stars,
            max_generations=args.generations,
            population_size=args.population,
            exhaustive=args.exhaustive,
            seed=args.seed,
        )
    
    elif args.target == 'rar':
        discover_rar(
            max_points=args.stars,
            max_generations=args.generations,
            population_size=args.population,
            exhaustive=args.exhaustive,
            seed=args.seed,
        )
    
    elif args.target == 'btf':
        discover_btf(
            max_generations=args.generations,
            population_size=args.population,
            exhaustive=args.exhaustive,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
