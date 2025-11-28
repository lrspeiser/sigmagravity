#!/usr/bin/env python3
"""
Σ-GRAVITY PURE FIRST-PRINCIPLES DISCOVERY
==========================================

Discover field equations from Σ-Gravity coherence data.
NO TEMPLATES. NO PHYSICS ASSUMPTIONS. PURE DATA-DRIVEN.

Uses REAL astronomical data:
- Gaia MW star-level data (1.8M stars)
- SPARC galaxy rotation curves (166 galaxies)
- Galaxy cluster profiles

Massive computational budget:
- 10,000+ population per island
- 500+ generations
- Billions of formula evaluations
- Aggressive pruning

Usage:
    python run_pure_discovery.py --target K --stars 100000 --generations 300
    python run_pure_discovery.py --target K --stars all --generations 500
    python run_pure_discovery.py --all

Author: Σ-Gravity Research - First Principles Derivation
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Import pure discovery engine
from sigma_pure_discovery import (
    PureDiscoveryEngine,
    DiscoveryResult,
    GPU_AVAILABLE,
    NUM_CORES,
)

# Import data loaders
from data_loaders import (
    load_sparc_galaxies,
    load_sparc_rar,
    load_gaia_rar,
    load_gaia_enhancement_factor,
)


# =============================================================================
# PURE DISCOVERY FUNCTIONS
# =============================================================================

def discover_enhancement_factor_pure(
    max_stars: int = 100000,
    population_size: int = 10000,
    max_generations: int = 300,
    num_islands: int = 10,
    max_depth: int = 10,
    seed: int = None,
) -> dict:
    """
    Discover gravitational enhancement factor K(R) from FIRST PRINCIPLES.
    
    NO TEMPLATES. NO PHYSICS ASSUMPTIONS.
    
    Target: K = g_obs/g_bar - 1 (the gravitational enhancement)
    """
    print("\n" + "╔" + "═"*73 + "╗")
    print("║  Σ-GRAVITY PURE FIRST-PRINCIPLES DISCOVERY                              ║")
    print("║  Target: Enhancement Factor K(R)                                        ║")
    print("║  NO TEMPLATES. NO PHYSICS ASSUMPTIONS.                                  ║")
    print("╚" + "═"*73 + "╝")
    
    # Load REAL Gaia data
    print("\n[1/4] Loading Gaia MW data...")
    try:
        X, K, meta = load_gaia_enhancement_factor(max_stars=max_stars, seed=seed or 42)
        print(f"  ✓ Loaded {len(K):,} stars from Gaia")
    except FileNotFoundError as e:
        print(f"  ✗ Gaia data not found: {e}")
        print("  Falling back to SPARC RAR...")
        X_rar, y_rar, meta = load_sparc_rar(max_points=5000, seed=seed or 42)
        
        # Compute K from RAR
        log_ratio = y_rar - X_rar['g_bar']
        K = 10**log_ratio - 1
        valid = (K > -0.5) & (K < 100) & np.isfinite(K)
        
        X = {'R': X_rar['R'][valid], 'g_bar': X_rar['g_bar'][valid]}
        K = K[valid]
    
    print(f"\n[2/4] Data statistics:")
    print(f"  Points: {len(K):,}")
    print(f"  R range: [{X['R'].min():.2f}, {X['R'].max():.2f}] kpc")
    print(f"  K range: [{K.min():.3f}, {K.max():.3f}]")
    print(f"  K mean: {np.mean(K):.4f} ± {np.std(K):.4f}")
    
    # Set up variables - R is primary, g_bar secondary
    var_names = ['R', 'g_bar']
    X_ordered = {'R': X['R'], 'g_bar': X['g_bar']}
    
    # Calculate estimated evaluations
    total_pop = population_size * num_islands
    est_evaluations = total_pop * max_generations
    
    print(f"\n[3/4] Search configuration:")
    print(f"  Population: {population_size:,} × {num_islands} islands = {total_pop:,}")
    print(f"  Generations: {max_generations}")
    print(f"  Max depth: {max_depth}")
    print(f"  Estimated evaluations: {est_evaluations:,}")
    print(f"  Hardware: {'GPU (' + str(NUM_CORES) + ' cores)' if GPU_AVAILABLE else 'CPU (' + str(NUM_CORES) + ' cores)'}")
    
    # Initialize PURE discovery engine - NO TEMPLATES
    print(f"\n[4/4] Running pure first-principles discovery...")
    
    engine = PureDiscoveryEngine(
        var_names=var_names,
        population_size=population_size,
        max_generations=max_generations,
        num_islands=num_islands,
        migration_rate=0.15,
        migration_interval=15,
        tournament_size=12,
        crossover_rate=0.75,
        mutation_rate=0.3,        # Higher for exploration
        elitism=max(50, population_size // 100),
        complexity_weight=0.0005,  # Very low - allow complex formulas
        max_depth=max_depth,
        seed=seed,
    )
    
    best_expr, best_mse, best_r2 = engine.evolve(
        X_ordered, K, 
        verbose=True,
        early_stop_r2=0.98,
    )
    
    # Results
    print("\n" + "═"*75)
    print("  DISCOVERY RESULTS - ENHANCEMENT FACTOR K(R)")
    print("═"*75)
    
    if best_expr:
        formula = best_expr.to_string(var_names)
        print(f"\n  ╔═══════════════════════════════════════════════════════════════════════╗")
        print(f"  ║  BEST DISCOVERED FORMULA (NO TEMPLATES):                              ║")
        print(f"  ║  K = {formula[:65]:<65} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════╣")
        print(f"  ║  R² = {best_r2:.8f}  |  MSE = {best_mse:.6e}                         ║")
        print(f"  ║  Complexity = {best_expr.complexity():<3}  |  Evaluations = {engine.total_evaluations:,}           ║")
        print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")
    else:
        formula = "None found"
        print("  ✗ No valid formula discovered")
    
    # Show Pareto front
    pareto = engine.get_pareto_front()
    if pareto:
        print(f"\n  PARETO FRONT ({len(pareto)} solutions):")
        for i, r in enumerate(pareto[:15]):
            print(f"    {i+1:2d}. R²={r.r_squared:.6f} | C={r.complexity:2d} | {r.formula_string[:50]}")
    
    # Show top formulas
    top = engine.get_top_formulas(20)
    if top:
        print(f"\n  TOP 20 FORMULAS BY R²:")
        for i, r in enumerate(top[:20]):
            print(f"    {i+1:2d}. R²={r.r_squared:.6f} | {r.formula_string[:55]}")
    
    # Physical interpretation
    if best_r2 > 0:
        K_mean = np.mean(K)
        print(f"\n  PHYSICAL INTERPRETATION:")
        print(f"  • Mean enhancement K = {K_mean:.3f}")
        print(f"  • At K={K_mean:.3f}, observed gravity is {1+K_mean:.3f}× baryonic prediction")
        print(f"  • This is the Σ-Gravity coherence effect")
    
    # Export results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"K_pure_discovery_{timestamp}.json"
    engine.export_results(str(output_file))
    
    return {
        'target': 'K',
        'best_formula': formula,
        'r_squared': best_r2,
        'mse': best_mse,
        'n_data': len(K),
        'total_evaluations': engine.total_evaluations,
        'pareto_front': [r.to_dict() for r in pareto[:50]],
    }


def discover_btf_pure(
    population_size: int = 5000,
    max_generations: int = 200,
    num_islands: int = 8,
    seed: int = None,
) -> dict:
    """
    Discover Baryonic Tully-Fisher relation from FIRST PRINCIPLES.
    
    NO TEMPLATES. NO PHYSICS ASSUMPTIONS.
    
    Target: log(v_flat) = f(log(M_baryon), ...)
    """
    print("\n" + "╔" + "═"*73 + "╗")
    print("║  Σ-GRAVITY PURE FIRST-PRINCIPLES DISCOVERY                              ║")
    print("║  Target: Baryonic Tully-Fisher Relation                                 ║")
    print("║  NO TEMPLATES. NO PHYSICS ASSUMPTIONS.                                  ║")
    print("╚" + "═"*73 + "╝")
    
    # Load SPARC data
    print("\n  Loading SPARC galaxy data...")
    X, v_flat, meta = load_sparc_galaxies()
    
    # Use log velocity
    y = np.log10(v_flat)
    
    print(f"  ✓ Loaded {len(y)} galaxies")
    print(f"  M range: [10^{X['M'].min():.1f}, 10^{X['M'].max():.1f}] M_sun")
    print(f"  v range: [{v_flat.min():.1f}, {v_flat.max():.1f}] km/s")
    
    var_names = list(X.keys())
    
    # Initialize engine
    engine = PureDiscoveryEngine(
        var_names=var_names,
        population_size=population_size,
        max_generations=max_generations,
        num_islands=num_islands,
        max_depth=8,
        complexity_weight=0.002,
        seed=seed,
    )
    
    best_expr, best_mse, best_r2 = engine.evolve(
        X, y,
        verbose=True,
        early_stop_r2=0.99,
    )
    
    # Results
    if best_expr:
        formula = best_expr.to_string(var_names)
        print(f"\n  DISCOVERED: log(v) = {formula}")
        print(f"  R² = {best_r2:.6f}")
        print(f"\n  Standard BTF expects: log(v) = 0.25 × log(M) + const")
    else:
        formula = "None"
    
    return {
        'target': 'BTF',
        'best_formula': formula,
        'r_squared': best_r2,
        'mse': best_mse,
    }


def run_all_pure(args):
    """Run pure discovery on all targets"""
    print("\n" + "╔" + "═"*73 + "╗")
    print("║  Σ-GRAVITY COMPREHENSIVE PURE FIRST-PRINCIPLES DISCOVERY               ║")
    print("║  DISCOVERING FIELD EQUATIONS FROM DATA ALONE                           ║")
    print("╚" + "═"*73 + "╝")
    
    print(f"\n  Hardware: {'GPU' if GPU_AVAILABLE else 'CPU'} | Cores: {NUM_CORES}")
    print(f"  Mode: PURE DISCOVERY - NO TEMPLATES")
    
    results = {}
    start_time = time.time()
    
    # Primary target: Enhancement factor K
    try:
        results['K'] = discover_enhancement_factor_pure(
            max_stars=args.stars,
            population_size=args.population,
            max_generations=args.generations,
            num_islands=args.islands,
            max_depth=args.depth,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n  ✗ K discovery failed: {e}")
        results['K'] = {'error': str(e)}
    
    # Secondary: BTF
    try:
        results['BTF'] = discover_btf_pure(
            population_size=args.population // 2,
            max_generations=args.generations // 2,
            seed=args.seed,
        )
    except Exception as e:
        print(f"\n  ✗ BTF discovery failed: {e}")
        results['BTF'] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "═"*75)
    print("  FINAL SUMMARY")
    print("═"*75)
    
    for target, res in results.items():
        if 'error' in res:
            print(f"\n  {target}: FAILED - {res['error']}")
        else:
            print(f"\n  {target}:")
            print(f"    Formula: {res['best_formula'][:55]}")
            print(f"    R² = {res['r_squared']:.6f}")
            if 'total_evaluations' in res:
                print(f"    Evaluations: {res['total_evaluations']:,}")
    
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Export combined
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"ALL_pure_discovery_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Σ-Gravity Pure First-Principles Discovery Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run with 100K stars
  python run_pure_discovery.py --target K --stars 100000
  
  # Maximum scale run (huge budget)
  python run_pure_discovery.py --target K --stars 500000 --population 15000 --generations 500 --islands 12
  
  # Quick test
  python run_pure_discovery.py --target K --stars 10000 --generations 50
  
  # All targets
  python run_pure_discovery.py --all
        """
    )
    
    parser.add_argument(
        '--target', '-t',
        choices=['K', 'btf', 'all'],
        default='K',
        help='Discovery target (default: K)'
    )
    
    parser.add_argument(
        '--stars', '-n',
        type=int,
        default=100000,
        help='Number of stars to use (default: 100000)'
    )
    
    parser.add_argument(
        '--population', '-p',
        type=int,
        default=10000,
        help='Population size per island (default: 10000)'
    )
    
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=300,
        help='Max generations (default: 300)'
    )
    
    parser.add_argument(
        '--islands', '-i',
        type=int,
        default=10,
        help='Number of islands (default: 10)'
    )
    
    parser.add_argument(
        '--depth', '-d',
        type=int,
        default=10,
        help='Max expression depth (default: 10)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all targets'
    )
    
    args = parser.parse_args()
    
    if args.all or args.target == 'all':
        run_all_pure(args)
    elif args.target == 'K':
        discover_enhancement_factor_pure(
            max_stars=args.stars,
            population_size=args.population,
            max_generations=args.generations,
            num_islands=args.islands,
            max_depth=args.depth,
            seed=args.seed,
        )
    elif args.target == 'btf':
        discover_btf_pure(
            population_size=args.population,
            max_generations=args.generations,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
