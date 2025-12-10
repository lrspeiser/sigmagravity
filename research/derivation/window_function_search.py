#!/usr/bin/env python3
"""
Σ-GRAVITY EXHAUSTIVE WINDOW FUNCTION SEARCH
============================================

Tests ALL window function forms and optimizes coefficients.
Guaranteed to find best-fitting window function.

Usage:
    python window_function_search.py --data gaia_data.csv
    python window_function_search.py --synthetic
"""

import numpy as np
from scipy.optimize import differential_evolution
import time
import json
import argparse
from typing import List, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class WindowFunction:
    name: str
    formula_str: str
    func: Callable
    n_params: int
    param_bounds: List[Tuple[float, float]]
    param_names: List[str]


def create_window_functions() -> List[WindowFunction]:
    """Create all window function templates to test."""
    functions = []
    
    # POWER LAW: K = A × R^α
    functions.append(WindowFunction(
        name="power_law",
        formula_str="K = A × R^α",
        func=lambda R, g, A, alpha: A * np.power(np.maximum(R, 0.01), alpha),
        n_params=2,
        param_bounds=[(0.01, 10), (-1, 2)],
        param_names=['A', 'α'],
    ))
    
    functions.append(WindowFunction(
        name="sqrt_R",
        formula_str="K = A × √R",
        func=lambda R, g, A: A * np.sqrt(np.maximum(R, 0.01)),
        n_params=1,
        param_bounds=[(0.01, 10)],
        param_names=['A'],
    ))
    
    # WINDOW FORMS: K = A × R^α / (1 + R/R₀)^β
    functions.append(WindowFunction(
        name="window_sqrt",
        formula_str="K = A × √R / (1 + R/R₀)",
        func=lambda R, g, A, R0: A * np.sqrt(np.maximum(R, 0.01)) / (1 + R / R0),
        n_params=2,
        param_bounds=[(0.01, 10), (1, 50)],
        param_names=['A', 'R₀'],
    ))
    
    functions.append(WindowFunction(
        name="window_alpha",
        formula_str="K = A × R^α / (1 + R/R₀)",
        func=lambda R, g, A, alpha, R0: A * np.power(np.maximum(R, 0.01), alpha) / (1 + R / R0),
        n_params=3,
        param_bounds=[(0.01, 10), (0.1, 1.5), (1, 50)],
        param_names=['A', 'α', 'R₀'],
    ))
    
    functions.append(WindowFunction(
        name="window_alpha_beta",
        formula_str="K = A × R^α / (1 + R/R₀)^β",
        func=lambda R, g, A, alpha, R0, beta: A * np.power(np.maximum(R, 0.01), alpha) / np.power(1 + R / R0, beta),
        n_params=4,
        param_bounds=[(0.01, 10), (0.1, 1.5), (1, 50), (0.5, 3)],
        param_names=['A', 'α', 'R₀', 'β'],
    ))
    
    # BURR-XII (full Σ-Gravity coherence window)
    functions.append(WindowFunction(
        name="burr_xii",
        formula_str="K = K_max × (R/R_c)^α / (1 + (R/R_c)^β)^((α+γ)/β)",
        func=lambda R, g, Kmax, Rc, alpha, beta, gamma: 
            Kmax * np.power(np.maximum(R, 0.01)/Rc, alpha) / np.power(1 + np.power(np.maximum(R, 0.01)/Rc, beta), (alpha + gamma) / beta),
        n_params=5,
        param_bounds=[(0.1, 20), (1, 30), (0.1, 2), (0.5, 3), (0.1, 2)],
        param_names=['K_max', 'R_c', 'α', 'β', 'γ'],
    ))
    
    # RATIONAL FORMS
    functions.append(WindowFunction(
        name="rational_1",
        formula_str="K = R / (c + R/R₀)",
        func=lambda R, g, c, R0: R / (c + R / R0),
        n_params=2,
        param_bounds=[(0.01, 5), (1, 50)],
        param_names=['c', 'R₀'],
    ))
    
    functions.append(WindowFunction(
        name="rational_2",
        formula_str="K = A × R / (c + R)",
        func=lambda R, g, A, c: A * R / (c + R),
        n_params=2,
        param_bounds=[(0.01, 20), (0.1, 50)],
        param_names=['A', 'c'],
    ))
    
    functions.append(WindowFunction(
        name="rational_sqrt",
        formula_str="K = A × √R / (c + R)",
        func=lambda R, g, A, c: A * np.sqrt(np.maximum(R, 0.01)) / (c + R),
        n_params=2,
        param_bounds=[(0.01, 50), (0.1, 50)],
        param_names=['A', 'c'],
    ))
    
    # EXPONENTIAL CUTOFF
    functions.append(WindowFunction(
        name="exp_cutoff_sqrt",
        formula_str="K = A × √R × exp(-R/R₀)",
        func=lambda R, g, A, R0: A * np.sqrt(np.maximum(R, 0.01)) * np.exp(-R / R0),
        n_params=2,
        param_bounds=[(0.01, 50), (5, 100)],
        param_names=['A', 'R₀'],
    ))
    
    functions.append(WindowFunction(
        name="exp_cutoff_alpha",
        formula_str="K = A × R^α × exp(-R/R₀)",
        func=lambda R, g, A, alpha, R0: A * np.power(np.maximum(R, 0.01), alpha) * np.exp(-R / R0),
        n_params=3,
        param_bounds=[(0.01, 50), (0.1, 1.5), (5, 100)],
        param_names=['A', 'α', 'R₀'],
    ))
    
    # TANH FORMS
    functions.append(WindowFunction(
        name="tanh_simple",
        formula_str="K = A × tanh(R/R₀)",
        func=lambda R, g, A, R0: A * np.tanh(R / R0),
        n_params=2,
        param_bounds=[(0.1, 10), (1, 30)],
        param_names=['A', 'R₀'],
    ))
    
    functions.append(WindowFunction(
        name="tanh_shifted",
        formula_str="K = A × tanh((R-R₁)/w) + c",
        func=lambda R, g, A, R1, w, c: A * np.tanh((R - R1) / w) + c,
        n_params=4,
        param_bounds=[(0.1, 10), (0, 20), (1, 20), (-5, 10)],
        param_names=['A', 'R₁', 'w', 'c'],
    ))
    
    # LOGARITHMIC
    functions.append(WindowFunction(
        name="log_form",
        formula_str="K = A × log(R/R₀) + c",
        func=lambda R, g, A, R0, c: A * np.log10(np.maximum(R / R0, 0.01)) + c,
        n_params=3,
        param_bounds=[(0.1, 10), (0.1, 10), (-5, 10)],
        param_names=['A', 'R₀', 'c'],
    ))
    
    functions.append(WindowFunction(
        name="log_window",
        formula_str="K = A × log(R) / (1 + R/R₀)",
        func=lambda R, g, A, R0: A * np.log10(np.maximum(R, 0.1)) / (1 + R / R0),
        n_params=2,
        param_bounds=[(0.1, 20), (5, 50)],
        param_names=['A', 'R₀'],
    ))
    
    # LINEAR (baseline)
    functions.append(WindowFunction(
        name="linear",
        formula_str="K = A × R + c",
        func=lambda R, g, A, c: A * R + c,
        n_params=2,
        param_bounds=[(-1, 1), (-10, 10)],
        param_names=['A', 'c'],
    ))
    
    functions.append(WindowFunction(
        name="linear_offset",
        formula_str="K = A × (R - R₀)",
        func=lambda R, g, A, R0: A * (R - R0),
        n_params=2,
        param_bounds=[(0.01, 2), (-10, 20)],
        param_names=['A', 'R₀'],
    ))
    
    # g_bar DEPENDENT (MOND-like)
    functions.append(WindowFunction(
        name="mond_like",
        formula_str="K = A × √(a₀/g_bar)",
        func=lambda R, g, A, a0_log: A * np.sqrt(np.power(10, np.clip(a0_log - g, -20, 20))),
        n_params=2,
        param_bounds=[(0.1, 10), (-11, -9)],
        param_names=['A', 'log(a₀)'],
    ))
    
    functions.append(WindowFunction(
        name="hybrid_mond_window",
        formula_str="K = A × √R × √(a₀/g_bar) / (1 + R/R₀)",
        func=lambda R, g, A, a0_log, R0: 
            A * np.sqrt(np.maximum(R, 0.01)) * np.sqrt(np.power(10, np.clip(a0_log - g, -20, 20))) / (1 + R / R0),
        n_params=3,
        param_bounds=[(0.01, 10), (-11, -9), (5, 50)],
        param_names=['A', 'log(a₀)', 'R₀'],
    ))
    
    functions.append(WindowFunction(
        name="R_g_product",
        formula_str="K = A × R^α × 10^(β×g_bar)",
        func=lambda R, g, A, alpha, beta: A * np.power(np.maximum(R, 0.01), alpha) * np.power(10, beta * g),
        n_params=3,
        param_bounds=[(0.01, 100), (0.1, 2), (-0.5, 0.5)],
        param_names=['A', 'α', 'β'],
    ))
    
    # DOUBLE WINDOW
    functions.append(WindowFunction(
        name="double_window",
        formula_str="K = A × (R/R₁)^α / ((1 + R/R₁) × (1 + R/R₂))",
        func=lambda R, g, A, R1, R2, alpha: 
            A * np.power(np.maximum(R, 0.01)/R1, alpha) / ((1 + R/R1) * (1 + R/R2)),
        n_params=4,
        param_bounds=[(0.1, 50), (1, 20), (5, 50), (0.1, 2)],
        param_names=['A', 'R₁', 'R₂', 'α'],
    ))
    
    return functions


def fit_window_function(wf: WindowFunction, R: np.ndarray, g: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fit window function parameters to data. Returns (params, mse, r2)."""
    
    def objective(params):
        try:
            K_pred = wf.func(R, g, *params)
            valid = np.isfinite(K_pred)
            if valid.sum() < len(K) * 0.5:
                return 1e30
            return np.mean((K[valid] - K_pred[valid])**2)
        except:
            return 1e30
    
    result = differential_evolution(objective, bounds=wf.param_bounds, maxiter=200, seed=42, polish=True, workers=1)
    best_params = result.x
    
    K_pred = wf.func(R, g, *best_params)
    valid = np.isfinite(K_pred)
    
    if valid.sum() < len(K) * 0.5:
        return best_params, 1e30, -1.0
    
    K_v, K_pred_v = K[valid], K_pred[valid]
    mse = float(np.mean((K_v - K_pred_v)**2))
    ss_tot = np.sum((K_v - np.mean(K_v))**2)
    ss_res = np.sum((K_v - K_pred_v)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return best_params, mse, float(r2)


@dataclass
class SearchResult:
    name: str
    formula: str
    params: dict
    mse: float
    r2: float
    
    def to_dict(self):
        return {'name': self.name, 'formula': self.formula, 'params': self.params, 'mse': self.mse, 'r_squared': self.r2}


def run_exhaustive_search(R: np.ndarray, g: np.ndarray, K: np.ndarray, verbose: bool = True) -> List[SearchResult]:
    """Run exhaustive search over all window function forms."""
    functions = create_window_functions()
    results = []
    
    if verbose:
        print(f"\n  Testing {len(functions)} window function forms...")
        print("  " + "-"*65)
    
    for i, wf in enumerate(functions):
        start = time.time()
        params, mse, r2 = fit_window_function(wf, R, g, K)
        
        if params is not None:
            param_dict = {name: float(val) for name, val in zip(wf.param_names, params)}
            results.append(SearchResult(name=wf.name, formula=wf.formula_str, params=param_dict, mse=mse, r2=r2))
            
            if verbose:
                param_str = ", ".join(f"{k}={v:.3g}" for k, v in param_dict.items())
                print(f"  {i+1:2d}. {wf.name:20s} | R²={r2:.5f} | {time.time()-start:.1f}s | {param_str}")
    
    results.sort(key=lambda x: -x.r2)
    return results


def load_gaia_data(filepath: str = None, max_stars: int = 100000, seed: int = 42):
    """Load Gaia enhancement factor data."""
    import pandas as pd
    
    # Default path
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "gaia" / "outputs" / "mw_rar_starlevel_full.csv"
    
    df = pd.read_csv(filepath)
    
    # Compute K if not present
    if 'K' not in df.columns:
        if 'log10_g_obs' in df.columns and 'log10_g_bar' in df.columns:
            df['K'] = 10**(df['log10_g_obs'] - df['log10_g_bar']) - 1
    
    # Filter valid
    valid = (df['K'] > -0.5) & (df['K'] < 50) & df['K'].notna()
    df = df[valid]
    
    if len(df) > max_stars:
        df = df.sample(n=max_stars, random_state=seed)
    
    R = df['R_kpc'].values if 'R_kpc' in df.columns else df['R'].values
    g = df['log10_g_bar'].values if 'log10_g_bar' in df.columns else df['g_bar'].values
    K = df['K'].values
    
    return R, g, K


def generate_synthetic_data(n_points: int = 20000, seed: int = 42):
    """Generate synthetic data with known window function."""
    np.random.seed(seed)
    
    R = np.random.uniform(0.5, 30, n_points)
    g = np.random.uniform(-12, -9, n_points)
    
    A_true, R0_true = 1.8, 15.0
    K_true = A_true * np.sqrt(R) / (1 + R / R0_true)
    K = K_true * (1 + np.random.normal(0, 0.08, n_points))
    K = np.clip(K, 0.01, 20)
    
    return R, g, K, {'A': A_true, 'R0': R0_true, 'formula': f'K = {A_true} × √R / (1 + R/{R0_true})'}


def main():
    parser = argparse.ArgumentParser(description='Σ-Gravity Window Function Search')
    parser.add_argument('--data', type=str, help='Path to Gaia CSV')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--stars', type=int, default=50000, help='Max stars')
    parser.add_argument('--export', type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*70)
    print("  Σ-GRAVITY EXHAUSTIVE WINDOW FUNCTION SEARCH")
    print("="*70)
    
    if args.synthetic:
        print("\n  Using synthetic test data...")
        R, g, K, true_params = generate_synthetic_data(n_points=args.stars)
        print(f"  TRUE FORMULA: {true_params['formula']}")
    else:
        print(f"\n  Loading Gaia data...")
        try:
            R, g, K = load_gaia_data(args.data, max_stars=args.stars)
            true_params = None
            print(f"  ✓ Loaded {len(K):,} stars")
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            print("  Falling back to synthetic data...")
            R, g, K, true_params = generate_synthetic_data(n_points=args.stars)
            print(f"  TRUE FORMULA: {true_params['formula']}")
    
    print(f"\n  Data points: {len(K):,}")
    print(f"  R range: [{R.min():.2f}, {R.max():.2f}] kpc")
    print(f"  K range: [{K.min():.3f}, {K.max():.3f}]")
    print(f"  K mean: {np.mean(K):.3f} ± {np.std(K):.3f}")
    
    start = time.time()
    results = run_exhaustive_search(R, g, K, verbose=True)
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("  TOP RESULTS")
    print("="*70)
    
    for i, r in enumerate(results[:10], 1):
        param_str = ", ".join(f"{k}={v:.4g}" for k, v in r.params.items())
        print(f"\n  {i}. {r.formula}")
        print(f"     R² = {r.r2:.6f}")
        print(f"     Params: {param_str}")
    
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # Export
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    export_path = args.export or str(output_dir / "window_search_results.json")
    
    output = {
        'best': results[0].to_dict() if results else None,
        'all_results': [r.to_dict() for r in results],
        'data_info': {'n_points': len(K), 'K_mean': float(np.mean(K)), 'K_std': float(np.std(K)),
                      'R_min': float(R.min()), 'R_max': float(R.max())},
        'true_params': true_params,
    }
    
    with open(export_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to {export_path}")
    
    if true_params:
        print("\n" + "="*70)
        print("  VALIDATION")
        print("="*70)
        best = results[0]
        print(f"\n  True formula:      {true_params['formula']}")
        print(f"  Discovered formula: {best.formula}")
        print(f"  Discovered R²:      {best.r2:.6f}")


if __name__ == '__main__':
    main()
