#!/usr/bin/env python3
"""
MULTI-VARIABLE K(R) DISCOVERY
=============================

The single-variable K(R) only explains 33% of variance.
This script searches for K(R, z, v_R, v_phi, v_z, [Fe/H], g_bar)

Key hypothesis: Coherence depends on:
- R: galactocentric radius
- z: height above disk (vertical coherence)
- v_phi: rotation velocity (dynamical state)
- sigma_v: velocity dispersion (phase space density)
- g_bar: baryonic field strength (MOND connection)
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import json
import time
from pathlib import Path


@dataclass
class MultiVarFormula:
    name: str
    formula: str
    func: Callable
    n_params: int
    bounds: List[Tuple[float, float]]
    param_names: List[str]


def create_multivariable_templates() -> List[MultiVarFormula]:
    """
    Create templates that include multiple variables.
    
    Physical motivations:
    - z-dependence: Coherence may be disk-confined
    - v_phi dependence: Rotation affects coherence buildup
    - g_bar dependence: MOND-like transition
    - Combined: Full phase-space dependence
    """
    templates = []
    
    # === R + z dependence (disk confinement) ===
    
    templates.append(MultiVarFormula(
        name="tanh_z_suppression",
        formula="K = [A×tanh((R-R₁)/w) + c] × exp(-|z|/h_z)",
        func=lambda df, A, R1, w, c, h_z: 
            (A * np.tanh((df['R'] - R1) / w) + c) * np.exp(-np.abs(df['z']) / h_z),
        n_params=5,
        bounds=[(0.1, 3), (0, 15), (0.5, 10), (0, 3), (0.1, 5)],
        param_names=['A', 'R₁', 'w', 'c', 'h_z'],
    ))
    
    templates.append(MultiVarFormula(
        name="tanh_z_gaussian",
        formula="K = [A×tanh((R-R₁)/w) + c] × exp(-z²/2σ_z²)",
        func=lambda df, A, R1, w, c, sigma_z:
            (A * np.tanh((df['R'] - R1) / w) + c) * np.exp(-df['z']**2 / (2 * sigma_z**2)),
        n_params=5,
        bounds=[(0.1, 3), (0, 15), (0.5, 10), (0, 3), (0.1, 3)],
        param_names=['A', 'R₁', 'w', 'c', 'σ_z'],
    ))
    
    templates.append(MultiVarFormula(
        name="window_z_confined",
        formula="K = A×√R/(1+R/R₀) × sech²(z/h_z)",
        func=lambda df, A, R0, h_z:
            A * np.sqrt(np.maximum(df['R'], 0.1)) / (1 + df['R'] / R0) / np.cosh(df['z'] / h_z)**2,
        n_params=3,
        bounds=[(0.1, 10), (5, 50), (0.1, 3)],
        param_names=['A', 'R₀', 'h_z'],
    ))
    
    # === R + g_bar dependence (MOND connection) ===
    
    templates.append(MultiVarFormula(
        name="tanh_mond_hybrid",
        formula="K = [A×tanh((R-R₁)/w) + c] × ν(g_bar/a₀)",
        func=lambda df, A, R1, w, c, log_a0:
            (A * np.tanh((df['R'] - R1) / w) + c) * 
            (1 + np.sqrt(np.power(10, np.clip(log_a0 - df['g_bar'], -10, 10)))),
        n_params=5,
        bounds=[(0.05, 1), (0, 15), (0.5, 10), (0, 1), (-11, -9)],
        param_names=['A', 'R₁', 'w', 'c', 'log_a₀'],
    ))
    
    templates.append(MultiVarFormula(
        name="sqrt_R_g_product",
        formula="K = A × R^α × (a₀/g_bar)^β",
        func=lambda df, A, alpha, beta, log_a0:
            A * np.power(np.maximum(df['R'], 0.1), alpha) * np.power(10, np.clip(beta * (log_a0 - df['g_bar']), -10, 10)),
        n_params=4,
        bounds=[(0.001, 10), (0.1, 2), (0, 1), (-11, -9)],
        param_names=['A', 'α', 'β', 'log_a₀'],
    ))
    
    # === R + z + g_bar (full 3-variable) ===
    
    templates.append(MultiVarFormula(
        name="full_3var_tanh",
        formula="K = [A×tanh((R-R₁)/w) + c] × exp(-|z|/h_z) × (1 + (a₀/g_bar)^β)",
        func=lambda df, A, R1, w, c, h_z, beta, log_a0:
            (A * np.tanh((df['R'] - R1) / w) + c) * 
            np.exp(-np.abs(df['z']) / h_z) *
            (1 + np.power(10, np.clip(beta * (log_a0 - df['g_bar']), -10, 10))),
        n_params=7,
        bounds=[(0.05, 1), (0, 15), (0.5, 10), (0, 1), (0.1, 5), (0, 0.5), (-11, -9)],
        param_names=['A', 'R₁', 'w', 'c', 'h_z', 'β', 'log_a₀'],
    ))
    
    # === Velocity dependence ===
    
    templates.append(MultiVarFormula(
        name="tanh_vphi_modulated",
        formula="K = [A×tanh((R-R₁)/w) + c] × (v_phi/v₀)^γ",
        func=lambda df, A, R1, w, c, v0, gamma:
            (A * np.tanh((df['R'] - R1) / w) + c) * 
            np.power(np.abs(df['v_phi']) / v0, gamma),
        n_params=6,
        bounds=[(0.1, 2), (0, 15), (0.5, 10), (0, 2), (100, 300), (-0.5, 0.5)],
        param_names=['A', 'R₁', 'w', 'c', 'v₀', 'γ'],
    ))
    
    templates.append(MultiVarFormula(
        name="coherence_vphi_sigma",
        formula="K = A × (R/R₀)^α / (1 + R/R₀) × (v_phi/σ_v)^β",
        func=lambda df, A, R0, alpha, beta:
            A * np.power(df['R'] / R0, alpha) / (1 + df['R'] / R0) *
            np.power(np.abs(df['v_phi']) / (df['sigma_v'] + 10), beta),
        n_params=4,
        bounds=[(0.01, 10), (1, 30), (0.1, 2), (-0.5, 0.5)],
        param_names=['A', 'R₀', 'α', 'β'],
    ))
    
    # === Cylindrical R-z combined ===
    
    templates.append(MultiVarFormula(
        name="cylindrical_coherence",
        formula="K = A × exp(-(R-R_c)²/2σ_R² - z²/2σ_z²) + K_floor",
        func=lambda df, A, R_c, sigma_R, sigma_z, K_floor:
            A * np.exp(-((df['R'] - R_c)**2) / (2 * sigma_R**2) - df['z']**2 / (2 * sigma_z**2)) + K_floor,
        n_params=5,
        bounds=[(0.5, 5), (5, 15), (1, 10), (0.1, 3), (0, 2)],
        param_names=['A', 'R_c', 'σ_R', 'σ_z', 'K_floor'],
    ))
    
    templates.append(MultiVarFormula(
        name="spherical_coherence",
        formula="K = A × r_sph^α / (1 + r_sph/r₀), r_sph = √(R² + z²)",
        func=lambda df, A, r0, alpha:
            A * np.power(np.sqrt(df['R']**2 + df['z']**2), alpha) / 
            (1 + np.sqrt(df['R']**2 + df['z']**2) / r0),
        n_params=3,
        bounds=[(0.1, 10), (5, 50), (0.1, 1.5)],
        param_names=['A', 'r₀', 'α'],
    ))
    
    # === Phase space density proxy ===
    
    templates.append(MultiVarFormula(
        name="phase_space_coherence",
        formula="K = A × (ρ_proxy)^α × (R/R₀)^β, ρ_proxy = exp(-R/R_d - |z|/h_z)",
        func=lambda df, A, R_d, h_z, alpha, R0, beta:
            A * np.power(np.exp(-df['R']/R_d - np.abs(df['z'])/h_z) + 0.01, alpha) *
            np.power(df['R'] / R0, beta),
        n_params=6,
        bounds=[(0.1, 50), (1, 10), (0.1, 2), (-1, 1), (1, 20), (0, 2)],
        param_names=['A', 'R_d', 'h_z', 'α', 'R₀', 'β'],
    ))
    
    # === Baseline comparisons ===
    
    templates.append(MultiVarFormula(
        name="baseline_tanh_R_only",
        formula="K = A×tanh((R-R₁)/w) + c",
        func=lambda df, A, R1, w, c:
            A * np.tanh((df['R'] - R1) / w) + c,
        n_params=4,
        bounds=[(0.1, 3), (0, 15), (0.5, 10), (0, 3)],
        param_names=['A', 'R₁', 'w', 'c'],
    ))
    
    templates.append(MultiVarFormula(
        name="baseline_linear_R_only",
        formula="K = A×R + c",
        func=lambda df, A, c:
            A * df['R'] + c,
        n_params=2,
        bounds=[(-1, 1), (-5, 5)],
        param_names=['A', 'c'],
    ))
    
    return templates


def fit_formula(formula: MultiVarFormula, df: pd.DataFrame, K: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fit a multi-variable formula to data."""
    
    def objective(params):
        try:
            K_pred = formula.func(df, *params)
            valid = np.isfinite(K_pred)
            if valid.sum() < len(K) * 0.5:
                return 1e30
            return np.mean((K[valid] - K_pred[valid])**2)
        except:
            return 1e30
    
    result = differential_evolution(
        objective,
        bounds=formula.bounds,
        maxiter=300,
        seed=42,
        polish=True,
        workers=1,
    )
    
    K_pred = formula.func(df, *result.x)
    valid = np.isfinite(K_pred)
    
    if valid.sum() < len(K) * 0.5:
        return result.x, 1e30, -1.0
    
    K_v, K_pred_v = K[valid], K_pred[valid]
    mse = float(np.mean((K_v - K_pred_v)**2))
    ss_tot = np.sum((K_v - np.mean(K_v))**2)
    r2 = 1 - np.sum((K_v - K_pred_v)**2) / ss_tot if ss_tot > 0 else 0.0
    
    return result.x, mse, float(r2)


def load_gaia_full(filepath: str = None, max_stars: int = 200000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load Gaia data with ALL available columns."""
    
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "gaia" / "outputs" / "mw_rar_starlevel_full.csv"
    
    df = pd.read_csv(filepath)
    
    # Compute K
    if 'K' not in df.columns:
        df['K'] = 10**(df['log10_g_obs'] - df['log10_g_bar']) - 1
    
    # Standardize column names
    rename_map = {
        'R_kpc': 'R',
        'z_kpc': 'z',
        'log10_g_bar': 'g_bar',
        'log10_g_obs': 'g_obs',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Quality cuts
    valid = (df['K'] > -0.5) & (df['K'] < 50) & df['K'].notna()
    if 'R' in df.columns:
        valid &= (df['R'] > 0) & (df['R'] < 30)
    if 'z' in df.columns:
        valid &= np.abs(df['z']) < 5
    
    df = df[valid].copy()
    
    # Add derived quantities
    if 'v_R' in df.columns and 'v_z' in df.columns:
        df['sigma_v'] = np.sqrt(df['v_R']**2 + df['v_z']**2)
    else:
        df['sigma_v'] = 30.0
    
    if 'z' not in df.columns:
        df['z'] = 0.0
    
    if 'v_phi' not in df.columns:
        df['v_phi'] = 220.0
    
    # Sample if needed
    if len(df) > max_stars:
        df = df.sample(n=max_stars, random_state=42)
    
    K = df['K'].values
    
    return df, K


def run_multivariable_search(
    filepath: str = None,
    max_stars: int = 200000,
    output_file: str = None
):
    """Run comprehensive multi-variable search."""
    
    print("="*70)
    print("  MULTI-VARIABLE K(R, z, v, g) DISCOVERY")
    print("="*70)
    
    # Load data
    print(f"\n  Loading data...")
    df, K = load_gaia_full(filepath, max_stars)
    
    print(f"  Stars: {len(K):,}")
    print(f"  K: {K.mean():.3f} ± {K.std():.3f}")
    
    # Check available columns
    available = []
    for col in ['R', 'z', 'g_bar', 'v_phi', 'v_R', 'v_z', 'sigma_v']:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.5:
            available.append(col)
    print(f"  Available columns: {available}")
    
    # Create templates
    templates = create_multivariable_templates()
    print(f"\n  Testing {len(templates)} multi-variable formulas...")
    print("  " + "-"*60)
    
    results = []
    
    for i, formula in enumerate(templates):
        start = time.time()
        
        try:
            params, mse, r2 = fit_formula(formula, df, K)
            elapsed = time.time() - start
            
            param_dict = {name: float(val) for name, val in zip(formula.param_names, params)}
            
            result = {
                'name': formula.name,
                'formula': formula.formula,
                'params': param_dict,
                'mse': mse,
                'r_squared': r2,
                'n_params': formula.n_params,
            }
            results.append(result)
            
            param_str = ", ".join(f"{k}={v:.3g}" for k, v in list(param_dict.items())[:4])
            print(f"  {i+1:2d}. {formula.name:25s} | R²={r2:.4f} | {elapsed:.1f}s | {param_str}")
            
        except Exception as e:
            print(f"  {i+1:2d}. {formula.name:25s} | FAILED: {e}")
    
    # Sort by R²
    results.sort(key=lambda x: -x['r_squared'])
    
    # Report
    print("\n" + "="*70)
    print("  TOP RESULTS")
    print("="*70)
    
    baseline_r2 = next((r['r_squared'] for r in results if 'baseline_tanh' in r['name']), 0.33)
    
    for i, r in enumerate(results[:10], 1):
        improvement = (r['r_squared'] - baseline_r2) / baseline_r2 * 100 if baseline_r2 > 0 else 0
        print(f"\n  {i}. {r['formula']}")
        print(f"     R² = {r['r_squared']:.5f}  ({improvement:+.1f}% vs baseline)")
        param_str = ", ".join(f"{k}={v:.4g}" for k, v in r['params'].items())
        print(f"     Params: {param_str}")
    
    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if output_file is None:
        output_file = str(output_dir / 'multivar_results.json')
    
    output = {
        'results': results,
        'baseline_r2': baseline_r2,
        'best_r2': results[0]['r_squared'] if results else 0,
        'data_info': {
            'n_stars': len(K),
            'K_mean': float(K.mean()),
            'K_std': float(K.std()),
            'columns': available,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to Gaia CSV')
    parser.add_argument('--stars', type=int, default=100000)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    run_multivariable_search(args.data, args.stars, args.output)
