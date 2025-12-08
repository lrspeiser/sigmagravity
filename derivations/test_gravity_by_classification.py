#!/usr/bin/env python3
"""
Systematic Gravity Formula Testing by Classification
=====================================================

This uses the comprehensive SPARC classification system to systematically
test how different gravity formulas perform on different galaxy subsamples.

KEY QUESTIONS:
1. Do certain formulas work better for specific galaxy types?
2. Are there systematic patterns in residuals by classification?
3. Can we identify which physical properties drive formula performance?
4. Does the coherence survival model show different patterns than MOND?

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8
H0_SI = 2.27e-18
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_SI = 6.674e-11

g_dagger = cH0 / (4 * math.sqrt(math.pi))
a0_mond = 1.2e-10


# =============================================================================
# GRAVITY MODELS
# =============================================================================

def h_function(g: np.ndarray, g_dag: float = g_dagger) -> np.ndarray:
    """Standard h(g) function."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)


def predict_mond_simple(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND simple interpolation."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    x = np.maximum(x, 1e-10)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    
    g_obs = g_bar * nu
    return np.sqrt(g_obs * R_m) / 1000


def predict_mond_standard(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND standard interpolation."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    y = g_bar / a0_mond
    y = np.maximum(y, 1e-10)
    nu = np.sqrt(0.5 + 0.5 * np.sqrt(1 + 4 / y**2))
    
    g_obs = g_bar * nu
    return np.sqrt(g_obs * R_m) / 1000


def predict_sigma_original(R_kpc: np.ndarray, V_bar: np.ndarray, 
                           R_d: float = 3.0, A: float = np.sqrt(3)) -> np.ndarray:
    """Original Σ-Gravity with W(r) window."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    xi = (2/3) * R_d
    W = 1 - (xi / (xi + R_kpc)) ** 0.5
    h = h_function(g_bar)
    
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_survival_threshold(R_kpc: np.ndarray, V_bar: np.ndarray,
                               r_char: float = 20.0, alpha: float = 0.1,
                               beta: float = 0.3, A: float = np.sqrt(3)) -> np.ndarray:
    """Coherence survival threshold model."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    r_ratio = r_char / np.maximum(R_kpc, 0.01)
    g_ratio = g_bar / g_dagger
    
    exponent = -np.power(r_ratio, beta) * np.power(g_ratio, alpha)
    P_survive = np.exp(exponent)
    
    h = h_function(g_bar)
    Sigma = 1.0 + A * P_survive * h
    return V_bar * np.sqrt(Sigma)


def predict_survival_nonlocal(R_kpc: np.ndarray, V_bar: np.ndarray,
                              sigma_v_kms: float = 30.0,
                              source_weight: float = 0.3,
                              A: float = np.sqrt(3)) -> np.ndarray:
    """Nonlocal coherence survival model."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    sigma_v_ms = sigma_v_kms * 1000
    
    g_bar = V_bar_ms**2 / R_m
    
    # Local decoherence rate
    rho = V_bar_ms**2 / (4 * np.pi * G_SI * R_m**2)
    gamma = g_bar / g_dagger + sigma_v_ms / 100e3 + rho / 1e-21
    
    # Path-integrated survival from peak
    lambda_D = 10 * kpc_to_m / np.maximum(gamma, 0.01)
    
    # Simplified path survival
    sort_idx = np.argsort(R_m)
    P_path = np.ones_like(R_m)
    cumulative = 0.0
    for i in range(1, len(R_m)):
        idx = sort_idx[i]
        idx_prev = sort_idx[i-1]
        dr = R_m[idx] - R_m[idx_prev]
        avg_rate = 0.5 * (1.0/lambda_D[idx] + 1.0/lambda_D[idx_prev])
        cumulative += dr * avg_rate
        P_path[idx] = np.exp(-cumulative)
    
    # Local survival
    r_char = 10.0
    r_ratio = r_char / np.maximum(R_kpc, 0.01)
    g_ratio = g_bar / g_dagger
    P_local = np.exp(-np.power(r_ratio, 0.5) * np.power(g_ratio, 0.2))
    
    P_combined = source_weight * P_path + (1 - source_weight) * P_local
    
    h = h_function(g_bar)
    Sigma = 1.0 + A * P_combined * h
    return V_bar * np.sqrt(Sigma)


# All models to test
GRAVITY_MODELS = {
    'MOND_simple': predict_mond_simple,
    'MOND_standard': predict_mond_standard,
    'Sigma_original': lambda R, V: predict_sigma_original(R, V, R_d=3.0),
    'Survival_threshold': predict_survival_threshold,
    'Survival_nonlocal': predict_survival_nonlocal,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rotation_curve(rotmod_path: Path) -> Optional[Dict]:
    """Load rotation curve data."""
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    
    with open(rotmod_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]))
                except ValueError:
                    continue
    
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    V_bar_sq = np.sign(V_gas) * V_gas**2 + np.sign(V_disk) * V_disk**2 + V_bulge**2
    V_bar_sq = np.maximum(V_bar_sq, 0)
    V_bar = np.sqrt(V_bar_sq)
    
    return {'R': R, 'V_obs': V_obs, 'V_err': V_err, 'V_bar': V_bar}


def load_classifications(json_path: Path) -> Dict:
    """Load galaxy classifications."""
    with open(json_path, 'r') as f:
        return json.load(f)


# =============================================================================
# METRICS
# =============================================================================

def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """RMS velocity error."""
    return np.sqrt(np.mean((V_obs - V_pred)**2))


def compute_chi2_red(V_obs: np.ndarray, V_pred: np.ndarray, V_err: np.ndarray) -> float:
    """Reduced chi-squared."""
    V_err_safe = np.maximum(V_err, 1.0)
    chi2 = np.sum(((V_obs - V_pred) / V_err_safe)**2)
    dof = len(V_obs) - 1
    return chi2 / max(dof, 1)


def compute_outer_rms(R: np.ndarray, V_obs: np.ndarray, V_pred: np.ndarray,
                      outer_frac: float = 0.5) -> float:
    """RMS in outer region only."""
    outer_mask = R > outer_frac * R.max()
    if outer_mask.sum() < 2:
        return np.nan
    return np.sqrt(np.mean((V_obs[outer_mask] - V_pred[outer_mask])**2))


def compute_inner_rms(R: np.ndarray, V_obs: np.ndarray, V_pred: np.ndarray,
                      inner_frac: float = 0.3) -> float:
    """RMS in inner region only."""
    inner_mask = R < inner_frac * R.max()
    if inner_mask.sum() < 2:
        return np.nan
    return np.sqrt(np.mean((V_obs[inner_mask] - V_pred[inner_mask])**2))


# =============================================================================
# TESTING FRAMEWORK
# =============================================================================

@dataclass
class ModelResults:
    """Results for a single model on a galaxy subset."""
    model_name: str
    category: str
    n_galaxies: int
    mean_rms: float
    median_rms: float
    std_rms: float
    mean_outer_rms: float
    mean_inner_rms: float
    win_rate_vs_mond: float = 0.0
    galaxy_results: Dict = None


def test_model_on_subset(
    model_func: Callable,
    model_name: str,
    galaxy_names: List[str],
    rotation_curves: Dict[str, Dict],
    category: str
) -> ModelResults:
    """Test a model on a subset of galaxies."""
    
    rms_list = []
    outer_rms_list = []
    inner_rms_list = []
    galaxy_results = {}
    
    for name in galaxy_names:
        if name not in rotation_curves:
            continue
        
        rc = rotation_curves[name]
        R = rc['R']
        V_obs = rc['V_obs']
        V_bar = rc['V_bar']
        
        try:
            V_pred = model_func(R, V_bar)
            rms = compute_rms(V_obs, V_pred)
            outer_rms = compute_outer_rms(R, V_obs, V_pred)
            inner_rms = compute_inner_rms(R, V_obs, V_pred)
            
            if np.isfinite(rms):
                rms_list.append(rms)
                galaxy_results[name] = {'rms': rms, 'outer_rms': outer_rms, 'inner_rms': inner_rms}
            if np.isfinite(outer_rms):
                outer_rms_list.append(outer_rms)
            if np.isfinite(inner_rms):
                inner_rms_list.append(inner_rms)
        except:
            continue
    
    if len(rms_list) == 0:
        return ModelResults(
            model_name=model_name, category=category, n_galaxies=0,
            mean_rms=np.nan, median_rms=np.nan, std_rms=np.nan,
            mean_outer_rms=np.nan, mean_inner_rms=np.nan
        )
    
    return ModelResults(
        model_name=model_name,
        category=category,
        n_galaxies=len(rms_list),
        mean_rms=np.mean(rms_list),
        median_rms=np.median(rms_list),
        std_rms=np.std(rms_list),
        mean_outer_rms=np.mean(outer_rms_list) if outer_rms_list else np.nan,
        mean_inner_rms=np.mean(inner_rms_list) if inner_rms_list else np.nan,
        galaxy_results=galaxy_results
    )


def compute_win_rates(
    results_dict: Dict[str, Dict[str, ModelResults]],
    reference_model: str = 'MOND_simple'
) -> Dict[str, Dict[str, float]]:
    """Compute win rates for each model vs reference."""
    
    win_rates = {}
    
    for category, model_results in results_dict.items():
        win_rates[category] = {}
        
        if reference_model not in model_results:
            continue
        
        ref_results = model_results[reference_model].galaxy_results
        if ref_results is None:
            continue
        
        for model_name, results in model_results.items():
            if model_name == reference_model:
                continue
            if results.galaxy_results is None:
                continue
            
            wins = 0
            total = 0
            
            for galaxy, res in results.galaxy_results.items():
                if galaxy in ref_results:
                    if res['rms'] < ref_results[galaxy]['rms']:
                        wins += 1
                    total += 1
            
            win_rates[category][model_name] = wins / total if total > 0 else 0.0
    
    return win_rates


def run_systematic_test(
    classifications: Dict,
    rotation_curves: Dict[str, Dict],
    scheme_name: str
) -> Tuple[Dict, Dict]:
    """Run systematic test for a classification scheme."""
    
    # Group galaxies by category
    categories = defaultdict(list)
    for name, data in classifications.items():
        cat = data['classifications'].get(scheme_name, 'unknown')
        categories[cat].append(name)
    
    # Test each model on each category
    results = {}
    
    for category, galaxy_names in categories.items():
        if len(galaxy_names) < 3:
            continue
        
        results[category] = {}
        
        for model_name, model_func in GRAVITY_MODELS.items():
            model_results = test_model_on_subset(
                model_func, model_name, galaxy_names, rotation_curves, category
            )
            results[category][model_name] = model_results
    
    # Compute win rates
    win_rates = compute_win_rates(results)
    
    return results, win_rates


# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def print_scheme_results(scheme_name: str, results: Dict, win_rates: Dict):
    """Print results for a classification scheme."""
    
    print(f"\n{'='*80}")
    print(f"RESULTS BY {scheme_name.upper()}")
    print(f"{'='*80}")
    
    # Header
    models = list(GRAVITY_MODELS.keys())
    header = f"{'Category':<25}"
    for m in models:
        header += f" {m[:12]:<12}"
    print(header)
    print("-" * 90)
    
    # Results by category
    for category in sorted(results.keys()):
        cat_results = results[category]
        n = cat_results[models[0]].n_galaxies if models[0] in cat_results else 0
        
        row = f"{category[:22]:<22}({n:>2})"
        for m in models:
            if m in cat_results:
                rms = cat_results[m].mean_rms
                row += f" {rms:>11.1f}"
            else:
                row += f" {'N/A':>11}"
        print(row)
    
    # Win rates vs MOND
    print("\n" + "-" * 90)
    print("WIN RATES vs MOND_simple:")
    print("-" * 90)
    
    for category in sorted(win_rates.keys()):
        cat_wins = win_rates[category]
        row = f"{category[:25]:<25}"
        for m in models[1:]:  # Skip MOND_simple
            if m in cat_wins:
                row += f" {100*cat_wins[m]:>10.1f}%"
            else:
                row += f" {'N/A':>11}"
        print(row)


def find_best_model_by_category(results: Dict) -> Dict[str, str]:
    """Find which model performs best for each category."""
    best_models = {}
    
    for category, cat_results in results.items():
        best_model = None
        best_rms = np.inf
        
        for model_name, model_res in cat_results.items():
            if model_res.mean_rms < best_rms:
                best_rms = model_res.mean_rms
                best_model = model_name
        
        best_models[category] = best_model
    
    return best_models


def analyze_model_preferences(all_results: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    """Analyze which categories prefer which models."""
    
    preferences = defaultdict(lambda: defaultdict(int))
    
    for scheme_name, results in all_results.items():
        best_models = find_best_model_by_category(results)
        
        for category, best_model in best_models.items():
            preferences[best_model][scheme_name] += 1
    
    return dict(preferences)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SYSTEMATIC GRAVITY FORMULA TESTING BY CLASSIFICATION")
    print("=" * 80)
    
    # Paths
    base_dir = Path("/Users/leonardspeiser/Projects/sigmagravity")
    classifications_path = base_dir / "derivations" / "sparc_galaxy_classifications.json"
    rotmod_dir = base_dir / "data" / "Rotmod_LTG"
    
    # Load data
    print("\nLoading classifications...")
    classifications = load_classifications(classifications_path)
    print(f"  Loaded {len(classifications)} galaxy classifications")
    
    print("\nLoading rotation curves...")
    rotation_curves = {}
    for rotmod_file in rotmod_dir.glob('*_rotmod.dat'):
        name = rotmod_file.stem.replace('_rotmod', '')
        rc = load_rotation_curve(rotmod_file)
        if rc is not None:
            rotation_curves[name] = rc
    print(f"  Loaded {len(rotation_curves)} rotation curves")
    
    # Schemes to test
    schemes_to_test = [
        'type_group',
        'bulge_presence',
        'rc_shape',
        'enhancement',
        'size',
        'surface_brightness',
        'acceleration_regime',
        'dm_fraction',
        'coherence_survival',
        'gas_fraction',
        'luminosity',
        'data_quality',
    ]
    
    all_results = {}
    all_win_rates = {}
    
    # Run tests
    for scheme_name in schemes_to_test:
        print(f"\nTesting by {scheme_name}...")
        results, win_rates = run_systematic_test(
            classifications, rotation_curves, scheme_name
        )
        all_results[scheme_name] = results
        all_win_rates[scheme_name] = win_rates
        
        print_scheme_results(scheme_name, results, win_rates)
    
    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: WHICH MODEL WINS WHERE?")
    print("=" * 80)
    
    # Count wins by model across all categories
    model_wins = defaultdict(int)
    total_categories = 0
    
    for scheme_name, results in all_results.items():
        best_models = find_best_model_by_category(results)
        for category, best_model in best_models.items():
            model_wins[best_model] += 1
            total_categories += 1
    
    print(f"\nBest model counts across {total_categories} category tests:")
    for model, wins in sorted(model_wins.items(), key=lambda x: -x[1]):
        print(f"  {model}: {wins} ({100*wins/total_categories:.1f}%)")
    
    # Identify where survival models beat MOND
    print("\n" + "-" * 80)
    print("CATEGORIES WHERE SURVIVAL BEATS MOND:")
    print("-" * 80)
    
    for scheme_name, win_rates in all_win_rates.items():
        survival_advantages = []
        
        for category, rates in win_rates.items():
            for model_name, rate in rates.items():
                if 'Survival' in model_name and rate > 0.6:
                    survival_advantages.append((scheme_name, category, model_name, rate))
        
        for scheme, cat, model, rate in survival_advantages:
            print(f"  {scheme}/{cat}: {model} wins {100*rate:.0f}%")
    
    # Identify where MOND beats survival
    print("\n" + "-" * 80)
    print("CATEGORIES WHERE MOND BEATS SURVIVAL:")
    print("-" * 80)
    
    for scheme_name, win_rates in all_win_rates.items():
        mond_advantages = []
        
        for category, rates in win_rates.items():
            for model_name, rate in rates.items():
                if 'Survival' in model_name and rate < 0.4:
                    mond_advantages.append((scheme_name, category, model_name, rate))
        
        for scheme, cat, model, rate in mond_advantages:
            print(f"  {scheme}/{cat}: {model} wins only {100*rate:.0f}%")
    
    # Save detailed results
    output_path = base_dir / "derivations" / "gravity_test_by_classification.json"
    
    output = {}
    for scheme_name in schemes_to_test:
        output[scheme_name] = {}
        for category, cat_results in all_results[scheme_name].items():
            output[scheme_name][category] = {}
            for model_name, model_res in cat_results.items():
                output[scheme_name][category][model_name] = {
                    'n_galaxies': model_res.n_galaxies,
                    'mean_rms': model_res.mean_rms,
                    'median_rms': model_res.median_rms,
                    'mean_outer_rms': model_res.mean_outer_rms,
                    'mean_inner_rms': model_res.mean_inner_rms,
                }
            if category in all_win_rates[scheme_name]:
                output[scheme_name][category]['win_rates'] = all_win_rates[scheme_name][category]
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved detailed results to {output_path}")
    
    return all_results, all_win_rates


if __name__ == "__main__":
    all_results, all_win_rates = main()

