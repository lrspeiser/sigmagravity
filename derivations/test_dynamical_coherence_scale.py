#!/usr/bin/env python3
"""
Test Dynamical Coherence Scale Hypothesis

This script tests whether replacing the phenomenological coherence scale ξ = (2/3)R_d
with a dynamically-motivated scale improves predictions while maintaining the
spatial/instantaneous nature of the theory.

Key insight: The correlation between fitted r₀ and dynamical timescale doesn't mean
coherence "builds up over time" - rather, the radius where coherence transitions
is set by instantaneous dynamical rates (orbital frequency, vorticity, dispersion).

Two parameterizations tested:

(A) Timescale proxy (spatial via dimensionless ratio):
    ξ = ξ₀ × (T_dyn / T₀)^β

(B) Fully instantaneous rate-based form (theoretically preferred):
    ξ = k × σ_eff / Ω_d
    where Ω_d = V(R_d) / R_d

Tests performed:
1. K-fold cross-validation (fit on 80%, evaluate on 20%)
2. Check if it reduces mass/bulge systematics
3. Ablation study comparing baseline vs dynamical scales
4. Cluster consistency check

Usage:
    python test_dynamical_coherence_scale.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, minimize_scalar
import json
import math
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))

# Fixed amplitude (from optimizer)
A_COEFF = 1.6
B_COEFF = 109.0
G_GALAXY = 0.038
A_GALAXY = np.sqrt(A_COEFF + B_COEFF * G_GALAXY**2)

# Reference timescale for Option A
T0_GYR = 0.3  # Gyr

# Typical velocity dispersions
SIGMA_GAS = 10.0    # km/s
SIGMA_DISK = 25.0   # km/s
SIGMA_BULGE = 120.0 # km/s


@dataclass
class GalaxyData:
    """Galaxy data container."""
    name: str
    R: np.ndarray
    V_obs: np.ndarray
    V_bar: np.ndarray
    V_gas: np.ndarray
    V_disk: np.ndarray
    V_bulge: np.ndarray
    
    # Derived properties
    R_d: float = 0.0
    V_flat: float = 0.0
    gas_fraction: float = 0.0
    bulge_fraction: float = 0.0
    
    # Dynamical quantities (instantaneous)
    Omega_d: float = 0.0      # Angular frequency at R_d: V(R_d)/R_d
    sigma_eff: float = 0.0    # Effective velocity dispersion
    T_orbit_Rd: float = 0.0   # Orbital period at R_d (Gyr)
    
    # Classification
    mass_class: str = ""
    bulge_class: str = ""


def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """
    Coherence window W(r) = 1 - (ξ/(ξ+r))^0.5
    
    This is the spatial coherence factor - NOT time-dependent.
    The exponent 0.5 is derived from decoherence statistics.
    Only ξ is being tested for dynamical replacement.
    """
    return 1 - np.power(xi / (xi + r), 0.5)


def predict_sigma_gravity(R_kpc: np.ndarray, V_bar: np.ndarray, xi: float) -> np.ndarray:
    """Predict rotation velocity using Σ-Gravity with given coherence scale ξ."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    W = W_coherence(R_kpc, xi)
    
    Sigma = 1 + A_GALAXY * W * h
    return V_bar * np.sqrt(Sigma)


def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """Compute RMS error."""
    return np.sqrt(((V_obs - V_pred)**2).mean())


def load_sparc_galaxies(data_dir: Path) -> List[GalaxyData]:
    """Load all SPARC galaxies."""
    sparc_dir = data_dir / "Rotmod_LTG"
    galaxy_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    
    galaxies = []
    for gf in galaxy_files:
        data = []
        with open(gf) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    data.append({
                        'R': float(parts[0]),
                        'V_obs': float(parts[1]),
                        'V_gas': float(parts[3]),
                        'V_disk': float(parts[4]),
                        'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                    })
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        
        # Apply M/L corrections
        V_disk_scaled = df['V_disk'] * np.sqrt(0.5)
        V_bulge_scaled = df['V_bulge'] * np.sqrt(0.7)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    V_disk_scaled**2 + V_bulge_scaled**2)
        V_bar = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (V_bar > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        if valid.sum() < 5:
            continue
        
        gal = GalaxyData(
            name=gf.stem.replace('_rotmod', ''),
            R=df.loc[valid, 'R'].values,
            V_obs=df.loc[valid, 'V_obs'].values,
            V_bar=V_bar[valid].values,
            V_gas=df.loc[valid, 'V_gas'].values,
            V_disk=V_disk_scaled[valid].values,
            V_bulge=V_bulge_scaled[valid].values
        )
        
        galaxies.append(gal)
    
    return galaxies


def compute_dynamical_quantities(gal: GalaxyData) -> None:
    """Compute instantaneous dynamical quantities for a galaxy."""
    R = gal.R
    V_obs = gal.V_obs
    V_disk = gal.V_disk
    V_bulge = gal.V_bulge
    V_gas = gal.V_gas
    
    # Estimate disk scale length R_d
    if len(V_disk) > 0 and np.abs(V_disk).max() > 0:
        peak_idx = np.argmax(np.abs(V_disk))
        gal.R_d = R[peak_idx] if peak_idx > 0 else R.max() / 3
    else:
        gal.R_d = R.max() / 3
    
    # V_flat
    gal.V_flat = np.median(V_obs[-3:]) if len(V_obs) >= 3 else V_obs[-1]
    
    # Component fractions
    V_gas_max = np.abs(V_gas).max() if len(V_gas) > 0 else 0
    V_disk_max = np.abs(V_disk).max() if len(V_disk) > 0 else 0
    V_bulge_max = np.abs(V_bulge).max() if len(V_bulge) > 0 else 0
    V_total_sq = V_gas_max**2 + V_disk_max**2 + V_bulge_max**2
    
    if V_total_sq > 0:
        gal.gas_fraction = V_gas_max**2 / V_total_sq
        gal.bulge_fraction = V_bulge_max**2 / V_total_sq
    
    # Angular frequency at R_d: Ω_d = V(R_d) / R_d
    # This is an INSTANTANEOUS quantity (not time-dependent)
    V_at_Rd = np.interp(gal.R_d, R, V_obs) if gal.R_d <= R.max() else gal.V_flat
    gal.Omega_d = V_at_Rd / gal.R_d  # km/s / kpc
    
    # Effective velocity dispersion (weighted by component fractions)
    disk_fraction = max(0, 1 - gal.gas_fraction - gal.bulge_fraction)
    gal.sigma_eff = (gal.gas_fraction * SIGMA_GAS + 
                     disk_fraction * SIGMA_DISK + 
                     gal.bulge_fraction * SIGMA_BULGE)
    
    # Orbital period at R_d (for reference, not used in spatial formula)
    T_orbit = 2 * np.pi * gal.R_d / V_at_Rd if V_at_Rd > 0 else 0  # kpc / (km/s)
    gal.T_orbit_Rd = T_orbit * kpc_to_m / (1e9 * 3.15e7 * 1000)  # Convert to Gyr
    
    # Classifications
    if gal.V_flat < 80:
        gal.mass_class = "dwarf"
    elif gal.V_flat < 150:
        gal.mass_class = "normal"
    else:
        gal.mass_class = "massive"
    
    if gal.bulge_fraction > 0.3:
        gal.bulge_class = "bulge-dominated"
    else:
        gal.bulge_class = "disk-dominated"


# =============================================================================
# COHERENCE SCALE MODELS
# =============================================================================

def xi_baseline(gal: GalaxyData) -> float:
    """Baseline: ξ = (2/3) R_d"""
    return (2/3) * gal.R_d


def xi_fixed_global(gal: GalaxyData, r0: float = 5.0) -> float:
    """Fixed global r₀ (current optimizer result)"""
    return r0


def xi_timescale_proxy(gal: GalaxyData, xi0: float, beta: float) -> float:
    """
    Option A: Timescale proxy
    ξ = ξ₀ × (T_dyn / T₀)^β
    
    Spatial via dimensionless ratio - the scale is set by the
    instantaneous orbital period, not accumulated time.
    """
    T_dyn = gal.T_orbit_Rd
    if T_dyn <= 0:
        return xi0
    return xi0 * (T_dyn / T0_GYR) ** beta


def xi_rate_based(gal: GalaxyData, k: float) -> float:
    """
    Option B: Fully instantaneous rate-based form (theoretically preferred)
    ξ = k × σ_eff / Ω_d
    
    This is purely spatial/instantaneous:
    - σ_eff is the current velocity dispersion
    - Ω_d is the current angular frequency
    - Their ratio has units of length
    
    Physical interpretation: coherence scale is where
    random motions (σ) become comparable to ordered rotation (Ω×r)
    """
    if gal.Omega_d <= 0:
        return 5.0  # Fallback
    return k * gal.sigma_eff / gal.Omega_d


def xi_combined(gal: GalaxyData, k: float, alpha: float) -> float:
    """
    Combined: ξ = k × σ_eff / Ω_d × (R_d)^α
    
    Allows for additional size dependence.
    """
    if gal.Omega_d <= 0:
        return 5.0
    xi_rate = k * gal.sigma_eff / gal.Omega_d
    return xi_rate * (gal.R_d ** alpha)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(galaxies: List[GalaxyData], 
                   xi_func: callable, 
                   **params) -> Dict:
    """Evaluate a coherence scale model on all galaxies."""
    results = []
    
    for gal in galaxies:
        xi = xi_func(gal, **params)
        V_pred = predict_sigma_gravity(gal.R, gal.V_bar, xi)
        rms = compute_rms(gal.V_obs, V_pred)
        
        # Residual analysis
        residuals = gal.V_obs - V_pred
        inner_mask = gal.R < gal.R.max() / 2
        outer_mask = ~inner_mask
        
        results.append({
            'name': gal.name,
            'xi': xi,
            'rms': rms,
            'mean_residual': residuals.mean(),
            'inner_residual': residuals[inner_mask].mean() if inner_mask.sum() > 0 else 0,
            'outer_residual': residuals[outer_mask].mean() if outer_mask.sum() > 0 else 0,
            'mass_class': gal.mass_class,
            'bulge_class': gal.bulge_class,
            'V_flat': gal.V_flat,
            'bulge_fraction': gal.bulge_fraction
        })
    
    df = pd.DataFrame(results)
    
    return {
        'mean_rms': df['rms'].mean(),
        'median_rms': df['rms'].median(),
        'std_rms': df['rms'].std(),
        'mean_xi': df['xi'].mean(),
        'mean_residual': df['mean_residual'].mean(),
        'inner_residual': df['inner_residual'].mean(),
        'outer_residual': df['outer_residual'].mean(),
        'by_mass': df.groupby('mass_class')['rms'].mean().to_dict(),
        'by_bulge': df.groupby('bulge_class')['rms'].mean().to_dict(),
        'detailed': df
    }


def optimize_params(galaxies: List[GalaxyData], 
                    xi_func: callable, 
                    param_bounds: Dict[str, Tuple[float, float]]) -> Dict:
    """Optimize parameters for a coherence scale model."""
    
    def objective(x):
        params = dict(zip(param_bounds.keys(), x))
        result = evaluate_model(galaxies, xi_func, **params)
        return result['mean_rms']
    
    bounds = list(param_bounds.values())
    x0 = [(b[0] + b[1]) / 2 for b in bounds]
    
    from scipy.optimize import minimize
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    optimal_params = dict(zip(param_bounds.keys(), result.x))
    return optimal_params


def kfold_cross_validation(galaxies: List[GalaxyData], 
                           xi_func: callable, 
                           param_bounds: Dict[str, Tuple[float, float]],
                           k: int = 5) -> Dict:
    """K-fold cross-validation."""
    np.random.seed(42)
    indices = np.random.permutation(len(galaxies))
    fold_size = len(galaxies) // k
    
    train_rms = []
    test_rms = []
    fitted_params = []
    
    for i in range(k):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        train_gals = [galaxies[j] for j in train_idx]
        test_gals = [galaxies[j] for j in test_idx]
        
        # Fit on training set
        params = optimize_params(train_gals, xi_func, param_bounds)
        fitted_params.append(params)
        
        # Evaluate on both
        train_result = evaluate_model(train_gals, xi_func, **params)
        test_result = evaluate_model(test_gals, xi_func, **params)
        
        train_rms.append(train_result['mean_rms'])
        test_rms.append(test_result['mean_rms'])
    
    return {
        'train_rms_mean': np.mean(train_rms),
        'train_rms_std': np.std(train_rms),
        'test_rms_mean': np.mean(test_rms),
        'test_rms_std': np.std(test_rms),
        'overfitting': np.mean(test_rms) - np.mean(train_rms),
        'fitted_params': fitted_params
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_ablation_study(galaxies: List[GalaxyData]) -> Dict:
    """Run ablation study comparing different coherence scale models."""
    
    results = {}
    
    # 1. Baseline: ξ = (2/3) R_d
    print("\n1. Testing baseline: ξ = (2/3) R_d")
    results['baseline'] = evaluate_model(galaxies, xi_baseline)
    print(f"   Mean RMS: {results['baseline']['mean_rms']:.2f} km/s")
    
    # 2. Fixed global r₀ = 5 kpc (current optimizer)
    print("\n2. Testing fixed global: ξ = 5 kpc")
    results['fixed_5kpc'] = evaluate_model(galaxies, xi_fixed_global, r0=5.0)
    print(f"   Mean RMS: {results['fixed_5kpc']['mean_rms']:.2f} km/s")
    
    # 3. Option A: Timescale proxy ξ = ξ₀ × (T_dyn / T₀)^β
    print("\n3. Testing Option A: ξ = ξ₀ × (T_dyn / T₀)^β")
    params_A = optimize_params(galaxies, xi_timescale_proxy, 
                               {'xi0': (1.0, 20.0), 'beta': (0.1, 2.0)})
    results['timescale_proxy'] = evaluate_model(galaxies, xi_timescale_proxy, **params_A)
    results['timescale_proxy']['params'] = params_A
    print(f"   Optimal: ξ₀ = {params_A['xi0']:.2f} kpc, β = {params_A['beta']:.2f}")
    print(f"   Mean RMS: {results['timescale_proxy']['mean_rms']:.2f} km/s")
    
    # 4. Option B: Rate-based ξ = k × σ_eff / Ω_d (theoretically preferred)
    print("\n4. Testing Option B: ξ = k × σ_eff / Ω_d (rate-based)")
    params_B = optimize_params(galaxies, xi_rate_based, {'k': (0.01, 2.0)})
    results['rate_based'] = evaluate_model(galaxies, xi_rate_based, **params_B)
    results['rate_based']['params'] = params_B
    print(f"   Optimal: k = {params_B['k']:.3f}")
    print(f"   Mean RMS: {results['rate_based']['mean_rms']:.2f} km/s")
    
    # 5. Combined: ξ = k × σ_eff / Ω_d × R_d^α
    print("\n5. Testing Combined: ξ = k × σ_eff / Ω_d × R_d^α")
    params_C = optimize_params(galaxies, xi_combined, 
                               {'k': (0.01, 2.0), 'alpha': (-0.5, 1.0)})
    results['combined'] = evaluate_model(galaxies, xi_combined, **params_C)
    results['combined']['params'] = params_C
    print(f"   Optimal: k = {params_C['k']:.3f}, α = {params_C['alpha']:.2f}")
    print(f"   Mean RMS: {results['combined']['mean_rms']:.2f} km/s")
    
    return results


def check_systematics(results: Dict) -> None:
    """Check if dynamical models reduce known failure modes."""
    
    print("\n" + "=" * 80)
    print("SYSTEMATIC FAILURE MODE ANALYSIS")
    print("=" * 80)
    
    models = ['baseline', 'fixed_5kpc', 'timescale_proxy', 'rate_based', 'combined']
    
    # Mass class comparison
    print("\n--- RMS by Mass Class ---")
    print(f"{'Model':<20} {'Dwarf':>10} {'Normal':>10} {'Massive':>10} {'Massive-Dwarf':>15}")
    print("-" * 70)
    
    for model in models:
        by_mass = results[model]['by_mass']
        dwarf = by_mass.get('dwarf', 0)
        normal = by_mass.get('normal', 0)
        massive = by_mass.get('massive', 0)
        diff = massive - dwarf
        print(f"{model:<20} {dwarf:>10.2f} {normal:>10.2f} {massive:>10.2f} {diff:>+15.2f}")
    
    # Bulge class comparison
    print("\n--- RMS by Bulge Class ---")
    print(f"{'Model':<20} {'Disk-dom':>12} {'Bulge-dom':>12} {'Difference':>12}")
    print("-" * 60)
    
    for model in models:
        by_bulge = results[model]['by_bulge']
        disk = by_bulge.get('disk-dominated', 0)
        bulge = by_bulge.get('bulge-dominated', 0)
        diff = bulge - disk
        print(f"{model:<20} {disk:>12.2f} {bulge:>12.2f} {diff:>+12.2f}")
    
    # Inner/outer residual comparison
    print("\n--- Radial Residuals ---")
    print(f"{'Model':<20} {'Inner':>10} {'Outer':>10} {'Trend':>10}")
    print("-" * 55)
    
    for model in models:
        inner = results[model]['inner_residual']
        outer = results[model]['outer_residual']
        trend = outer - inner
        print(f"{model:<20} {inner:>+10.2f} {outer:>+10.2f} {trend:>+10.2f}")


def run_cross_validation(galaxies: List[GalaxyData]) -> Dict:
    """Run k-fold cross-validation for best models."""
    
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION (k=5)")
    print("=" * 80)
    
    cv_results = {}
    
    # Rate-based model (theoretically preferred)
    print("\nRate-based model: ξ = k × σ_eff / Ω_d")
    cv_results['rate_based'] = kfold_cross_validation(
        galaxies, xi_rate_based, {'k': (0.01, 2.0)}, k=5
    )
    print(f"  Train RMS: {cv_results['rate_based']['train_rms_mean']:.2f} ± {cv_results['rate_based']['train_rms_std']:.2f}")
    print(f"  Test RMS:  {cv_results['rate_based']['test_rms_mean']:.2f} ± {cv_results['rate_based']['test_rms_std']:.2f}")
    print(f"  Overfitting: {cv_results['rate_based']['overfitting']:+.2f} km/s")
    
    # Timescale proxy
    print("\nTimescale proxy: ξ = ξ₀ × (T_dyn / T₀)^β")
    cv_results['timescale_proxy'] = kfold_cross_validation(
        galaxies, xi_timescale_proxy, {'xi0': (1.0, 20.0), 'beta': (0.1, 2.0)}, k=5
    )
    print(f"  Train RMS: {cv_results['timescale_proxy']['train_rms_mean']:.2f} ± {cv_results['timescale_proxy']['train_rms_std']:.2f}")
    print(f"  Test RMS:  {cv_results['timescale_proxy']['test_rms_mean']:.2f} ± {cv_results['timescale_proxy']['test_rms_std']:.2f}")
    print(f"  Overfitting: {cv_results['timescale_proxy']['overfitting']:+.2f} km/s")
    
    return cv_results


def print_summary(ablation_results: Dict, cv_results: Dict) -> None:
    """Print final summary and recommendations."""
    
    print("\n" + "=" * 80)
    print("SUMMARY: DYNAMICAL COHERENCE SCALE HYPOTHESIS")
    print("=" * 80)
    
    # Rank models by RMS
    models = ['baseline', 'fixed_5kpc', 'timescale_proxy', 'rate_based', 'combined']
    ranked = sorted(models, key=lambda m: ablation_results[m]['mean_rms'])
    
    print("\n--- Model Ranking by Mean RMS ---")
    print(f"{'Rank':<6} {'Model':<20} {'Mean RMS':>10} {'vs Baseline':>12}")
    print("-" * 55)
    
    baseline_rms = ablation_results['baseline']['mean_rms']
    for i, model in enumerate(ranked, 1):
        rms = ablation_results[model]['mean_rms']
        improvement = (baseline_rms - rms) / baseline_rms * 100
        print(f"{i:<6} {model:<20} {rms:>10.2f} {improvement:>+12.1f}%")
    
    # Best model analysis
    best_model = ranked[0]
    print(f"\n--- Best Model: {best_model} ---")
    
    if best_model in ['timescale_proxy', 'rate_based', 'combined']:
        params = ablation_results[best_model].get('params', {})
        print(f"Parameters: {params}")
        print(f"Mean ξ: {ablation_results[best_model]['mean_xi']:.2f} kpc")
    
    # Cross-validation results
    print("\n--- Cross-Validation Summary ---")
    for model in ['rate_based', 'timescale_proxy']:
        if model in cv_results:
            cv = cv_results[model]
            print(f"\n{model}:")
            print(f"  Generalization gap: {cv['overfitting']:+.2f} km/s")
            print(f"  Test performance: {cv['test_rms_mean']:.2f} ± {cv['test_rms_std']:.2f} km/s")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    rate_based_rms = ablation_results['rate_based']['mean_rms']
    baseline_rms = ablation_results['baseline']['mean_rms']
    fixed_rms = ablation_results['fixed_5kpc']['mean_rms']
    
    if rate_based_rms < min(baseline_rms, fixed_rms) - 0.5:
        print("""
✓ ADOPT RATE-BASED COHERENCE SCALE

The rate-based model ξ = k × σ_eff / Ω_d:
1. Improves predictions over baseline
2. Is theoretically preferred (purely instantaneous/spatial)
3. Maintains spatial nature of the theory (no time accumulation)

Physical interpretation:
- ξ is where random motions (σ) become comparable to ordered rotation (Ω×r)
- This is an INSTANTANEOUS property of the velocity field
- Consistent with "coherence transition radius" interpretation

Recommended update to theory:
  Replace ξ = (2/3)R_d with ξ = k × σ_eff / Ω_d
  where k ≈ {:.3f} (fitted)
""".format(ablation_results['rate_based']['params']['k']))
    
    elif rate_based_rms < baseline_rms:
        print("""
○ MARGINAL IMPROVEMENT

The dynamical models show slight improvement but not dramatic.
This suggests:
1. The current phenomenological scale ξ = (2/3)R_d is already reasonable
2. The dynamical relationship exists but may not be the dominant effect
3. Consider keeping the simpler form unless more data supports the change

Recommendation: Document the finding but don't change the core formula yet.
""")
    
    else:
        print("""
✗ NO IMPROVEMENT

The dynamical models do not improve over baseline.
This suggests:
1. The correlation between r₀ and T_dyn may be a secondary effect
2. The current coherence scale captures the key physics adequately
3. The remaining scatter is due to other factors

Recommendation: Keep the current formulation.
""")


def main():
    print("=" * 80)
    print("TESTING DYNAMICAL COHERENCE SCALE HYPOTHESIS")
    print("=" * 80)
    print("""
Testing whether replacing the phenomenological coherence scale ξ = (2/3)R_d
with a dynamically-motivated scale improves predictions.

Key constraint: The theory must remain spatial/instantaneous (no time accumulation).
""")
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    
    print("Loading SPARC galaxies...")
    galaxies = load_sparc_galaxies(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Compute dynamical quantities
    print("\nComputing dynamical quantities...")
    for gal in galaxies:
        compute_dynamical_quantities(gal)
    
    # Run ablation study
    print("\n" + "=" * 80)
    print("ABLATION STUDY: COMPARING COHERENCE SCALE MODELS")
    print("=" * 80)
    ablation_results = run_ablation_study(galaxies)
    
    # Check systematics
    check_systematics(ablation_results)
    
    # Cross-validation
    cv_results = run_cross_validation(galaxies)
    
    # Summary
    print_summary(ablation_results, cv_results)
    
    # Save results
    output_dir = Path(__file__).parent / "dynamical_coherence_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        'models': {},
        'cv_results': {}
    }
    
    for model in ['baseline', 'fixed_5kpc', 'timescale_proxy', 'rate_based', 'combined']:
        summary['models'][model] = {
            'mean_rms': float(ablation_results[model]['mean_rms']),
            'median_rms': float(ablation_results[model]['median_rms']),
            'mean_xi': float(ablation_results[model]['mean_xi']),
            'by_mass': {k: float(v) for k, v in ablation_results[model]['by_mass'].items()},
            'by_bulge': {k: float(v) for k, v in ablation_results[model]['by_bulge'].items()},
            'params': ablation_results[model].get('params', {})
        }
    
    for model in cv_results:
        summary['cv_results'][model] = {
            'train_rms_mean': float(cv_results[model]['train_rms_mean']),
            'test_rms_mean': float(cv_results[model]['test_rms_mean']),
            'overfitting': float(cv_results[model]['overfitting'])
        }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

