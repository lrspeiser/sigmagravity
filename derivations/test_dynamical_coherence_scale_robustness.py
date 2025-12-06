#!/usr/bin/env python3
"""
Robustness checks for the dynamical coherence scale:

    ξ = k × σ_eff / Ω_d

This script tests different choices for Ω_d to ensure the improvement
doesn't depend on using V_obs (which would be circular):

  - obs:  Ω_d = V_obs(R_d) / R_d   (current test - may be circular)
  - bar:  Ω_d = V_bar(R_d) / R_d   (baryonic-only; avoids V_obs)
  - self: Ω_d = V_pred(R_d) / R_d  (self-consistent; avoids V_obs)

It also optionally uses photometric R_d from MasterSheet_SPARC.mrt.

Run:
  python derivations/test_dynamical_coherence_scale_robustness.py

Outputs:
  derivations/dynamical_coherence_robustness/robustness_summary.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
import json
import math
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8
H0_SI = 2.27e-18
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))

A_COEFF = 1.6
B_COEFF = 109.0
G_GALAXY = 0.038
A_GALAXY = np.sqrt(A_COEFF + B_COEFF * G_GALAXY**2)

SIGMA_GAS = 10.0
SIGMA_DISK = 25.0
SIGMA_BULGE = 120.0


@dataclass
class GalaxyData:
    name: str
    R: np.ndarray
    V_obs: np.ndarray
    V_bar: np.ndarray
    V_gas: np.ndarray
    V_disk: np.ndarray
    V_bulge: np.ndarray
    
    R_d: float = 0.0
    R_d_photometric: float = 0.0  # From MasterSheet if available
    V_flat: float = 0.0
    gas_fraction: float = 0.0
    bulge_fraction: float = 0.0
    sigma_eff: float = 0.0
    
    mass_class: str = ""
    bulge_class: str = ""


def h_function(g: np.ndarray) -> np.ndarray:
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    xi = max(xi, 0.01)
    return 1 - np.power(xi / (xi + r), 0.5)


def predict_sigma_gravity(R_kpc: np.ndarray, V_bar: np.ndarray, xi: float) -> np.ndarray:
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    W = W_coherence(R_kpc, xi)
    
    Sigma = 1 + A_GALAXY * W * h
    return V_bar * np.sqrt(Sigma)


def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    return np.sqrt(((V_obs - V_pred)**2).mean())


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()


def load_mastersheet_rd(data_dir: Path) -> Dict[str, float]:
    """Load photometric R_d from MasterSheet_SPARC.mrt if available."""
    candidates = [
        data_dir / "Rotmod_LTG" / "MasterSheet_SPARC.mrt",
        data_dir / "sparc" / "MasterSheet_SPARC.mrt",
        data_dir / "MasterSheet_SPARC.mrt",
    ]
    
    ms_path = next((p for p in candidates if p.exists()), None)
    if ms_path is None:
        return {}
    
    rd_map = {}
    
    try:
        # Try to parse as fixed-width or CSV
        with open(ms_path, 'r') as f:
            lines = f.readlines()
        
        # Find header line
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Galaxy' in line or 'Name' in line:
                header_idx = i
                break
        
        # Simple parsing - look for columns
        for line in lines[header_idx + 1:]:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                name = parts[0]
                # R_d is typically column 4 or 5 in SPARC MasterSheet
                for idx in [4, 5, 6]:
                    if idx < len(parts):
                        try:
                            rd = float(parts[idx])
                            if 0.1 < rd < 50:  # Reasonable R_d range
                                rd_map[normalize_name(name)] = rd
                                break
                        except ValueError:
                            continue
            except:
                continue
        
    except Exception as e:
        print(f"[warn] Could not parse MasterSheet: {e}")
    
    return rd_map


def load_sparc_galaxies(data_dir: Path) -> List[GalaxyData]:
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


def compute_dynamical_quantities(gal: GalaxyData, rd_map: Dict[str, float] = None) -> None:
    R = gal.R
    V_obs = gal.V_obs
    V_disk = gal.V_disk
    V_bulge = gal.V_bulge
    V_gas = gal.V_gas
    
    # Estimate R_d from disk velocity peak
    if len(V_disk) > 0 and np.abs(V_disk).max() > 0:
        peak_idx = np.argmax(np.abs(V_disk))
        gal.R_d = R[peak_idx] if peak_idx > 0 else R.max() / 3
    else:
        gal.R_d = R.max() / 3
    
    # Override with photometric R_d if available
    if rd_map:
        key = normalize_name(gal.name)
        if key in rd_map:
            gal.R_d_photometric = rd_map[key]
    
    gal.V_flat = np.median(V_obs[-3:]) if len(V_obs) >= 3 else V_obs[-1]
    
    V_gas_max = np.abs(V_gas).max() if len(V_gas) > 0 else 0
    V_disk_max = np.abs(V_disk).max() if len(V_disk) > 0 else 0
    V_bulge_max = np.abs(V_bulge).max() if len(V_bulge) > 0 else 0
    V_total_sq = V_gas_max**2 + V_disk_max**2 + V_bulge_max**2
    
    if V_total_sq > 0:
        gal.gas_fraction = V_gas_max**2 / V_total_sq
        gal.bulge_fraction = V_bulge_max**2 / V_total_sq
    
    disk_fraction = max(0, 1 - gal.gas_fraction - gal.bulge_fraction)
    gal.sigma_eff = (gal.gas_fraction * SIGMA_GAS + 
                     disk_fraction * SIGMA_DISK + 
                     gal.bulge_fraction * SIGMA_BULGE)
    
    if gal.V_flat < 80:
        gal.mass_class = "dwarf"
    elif gal.V_flat < 150:
        gal.mass_class = "normal"
    else:
        gal.mass_class = "massive"
    
    gal.bulge_class = "bulge-dominated" if gal.bulge_fraction > 0.3 else "disk-dominated"


# =============================================================================
# COHERENCE SCALE VARIANTS
# =============================================================================

def xi_obs(gal: GalaxyData, k: float, use_photometric_rd: bool = False) -> float:
    """ξ = k × σ_eff / Ω_d where Ω_d = V_obs(R_d) / R_d"""
    R_d = gal.R_d_photometric if (use_photometric_rd and gal.R_d_photometric > 0) else gal.R_d
    if R_d <= 0:
        return 5.0
    
    V_at_Rd = np.interp(R_d, gal.R, gal.V_obs)
    if V_at_Rd <= 0:
        return 5.0
    
    Omega_d = V_at_Rd / R_d
    return k * gal.sigma_eff / Omega_d


def xi_bar(gal: GalaxyData, k: float, use_photometric_rd: bool = False) -> float:
    """ξ = k × σ_eff / Ω_d where Ω_d = V_bar(R_d) / R_d (baryonic-only)"""
    R_d = gal.R_d_photometric if (use_photometric_rd and gal.R_d_photometric > 0) else gal.R_d
    if R_d <= 0:
        return 5.0
    
    V_at_Rd = np.interp(R_d, gal.R, gal.V_bar)
    if V_at_Rd <= 0:
        return 5.0
    
    Omega_d = V_at_Rd / R_d
    return k * gal.sigma_eff / Omega_d


def xi_self(gal: GalaxyData, k: float, use_photometric_rd: bool = False) -> float:
    """
    Self-consistent ξ: solve for ξ such that Ω_d = V_pred(R_d) / R_d
    
    This avoids using V_obs entirely - the prediction is self-consistent.
    """
    R_d = gal.R_d_photometric if (use_photometric_rd and gal.R_d_photometric > 0) else gal.R_d
    if R_d <= 0:
        return 5.0
    
    sigma = gal.sigma_eff
    V_bar_Rd = np.interp(R_d, gal.R, gal.V_bar)
    if V_bar_Rd <= 0:
        return 5.0
    
    # g_bar at R_d
    g_bar_Rd = (V_bar_Rd * 1000)**2 / (R_d * kpc_to_m)
    h_Rd = h_function(np.array([g_bar_Rd]))[0]
    
    # Fixed-point iteration
    Omega_0 = V_bar_Rd / R_d
    xi = k * sigma / Omega_0
    
    for _ in range(50):
        W = 1 - np.sqrt(xi / (xi + R_d))
        Sigma = 1 + A_GALAXY * W * h_Rd
        V_pred_Rd = V_bar_Rd * np.sqrt(max(Sigma, 1e-6))
        
        xi_new = k * sigma * R_d / max(V_pred_Rd, 1e-6)
        
        # Damped update for stability
        xi_next = 0.5 * xi + 0.5 * xi_new
        if abs(xi_next - xi) / max(xi, 1e-6) < 1e-4:
            return xi_next
        xi = xi_next
    
    return xi


def xi_baseline(gal: GalaxyData, use_photometric_rd: bool = False) -> float:
    """Baseline: ξ = (2/3) R_d"""
    R_d = gal.R_d_photometric if (use_photometric_rd and gal.R_d_photometric > 0) else gal.R_d
    return (2/3) * R_d


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(galaxies: List[GalaxyData], 
                   xi_func: Callable,
                   use_photometric_rd: bool = False,
                   **params) -> Dict:
    results = []
    
    for gal in galaxies:
        if 'k' in params:
            xi = xi_func(gal, params['k'], use_photometric_rd)
        else:
            xi = xi_func(gal, use_photometric_rd)
        
        V_pred = predict_sigma_gravity(gal.R, gal.V_bar, xi)
        rms = compute_rms(gal.V_obs, V_pred)
        
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
            'bulge_class': gal.bulge_class
        })
    
    df = pd.DataFrame(results)
    
    return {
        'mean_rms': df['rms'].mean(),
        'median_rms': df['rms'].median(),
        'std_rms': df['rms'].std(),
        'mean_xi': df['xi'].mean(),
        'inner_residual': df['inner_residual'].mean(),
        'outer_residual': df['outer_residual'].mean(),
        'by_mass': df.groupby('mass_class')['rms'].mean().to_dict(),
        'by_bulge': df.groupby('bulge_class')['rms'].mean().to_dict()
    }


def optimize_k(galaxies: List[GalaxyData], 
               xi_func: Callable,
               use_photometric_rd: bool = False) -> float:
    
    def objective(k):
        if k <= 0.01:
            return 1e10
        total_rms = 0
        for gal in galaxies:
            xi = xi_func(gal, k, use_photometric_rd)
            V_pred = predict_sigma_gravity(gal.R, gal.V_bar, xi)
            total_rms += compute_rms(gal.V_obs, V_pred)
        return total_rms / len(galaxies)
    
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0.05, 3.0), method='bounded')
    return result.x


def kfold_cv(galaxies: List[GalaxyData], 
             xi_func: Callable,
             use_photometric_rd: bool = False,
             k_folds: int = 5) -> Dict:
    
    np.random.seed(42)
    indices = np.random.permutation(len(galaxies))
    fold_size = len(galaxies) // k_folds
    
    train_rms = []
    test_rms = []
    fitted_k = []
    
    for i in range(k_folds):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        train_gals = [galaxies[j] for j in train_idx]
        test_gals = [galaxies[j] for j in test_idx]
        
        k = optimize_k(train_gals, xi_func, use_photometric_rd)
        fitted_k.append(k)
        
        train_result = evaluate_model(train_gals, xi_func, use_photometric_rd, k=k)
        test_result = evaluate_model(test_gals, xi_func, use_photometric_rd, k=k)
        
        train_rms.append(train_result['mean_rms'])
        test_rms.append(test_result['mean_rms'])
    
    return {
        'train_rms_mean': np.mean(train_rms),
        'train_rms_std': np.std(train_rms),
        'test_rms_mean': np.mean(test_rms),
        'test_rms_std': np.std(test_rms),
        'overfitting': np.mean(test_rms) - np.mean(train_rms),
        'k_mean': np.mean(fitted_k),
        'k_std': np.std(fitted_k)
    }


def main():
    print("=" * 80)
    print("ROBUSTNESS TEST: DYNAMICAL COHERENCE SCALE")
    print("=" * 80)
    print("""
Testing whether the 19% improvement depends on using V_obs (circular)
or holds with baryonic-only / self-consistent Ω_d.
""")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load galaxies
    print("Loading SPARC galaxies...")
    galaxies = load_sparc_galaxies(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Load photometric R_d
    print("\nLoading photometric R_d from MasterSheet...")
    rd_map = load_mastersheet_rd(data_dir)
    print(f"Found photometric R_d for {len(rd_map)} galaxies")
    
    # Compute quantities
    for gal in galaxies:
        compute_dynamical_quantities(gal, rd_map)
    
    n_with_photo_rd = sum(1 for g in galaxies if g.R_d_photometric > 0)
    print(f"Matched photometric R_d for {n_with_photo_rd}/{len(galaxies)} galaxies")
    
    # ==========================================================================
    # Run all variants
    # ==========================================================================
    
    results = {}
    
    print("\n" + "=" * 80)
    print("TESTING ALL VARIANTS")
    print("=" * 80)
    
    variants = [
        ('baseline', xi_baseline, False, False),
        ('obs (V_obs)', xi_obs, False, True),
        ('bar (V_bar)', xi_bar, False, True),
        ('self (V_pred)', xi_self, False, True),
    ]
    
    # Add photometric R_d variants if available
    if n_with_photo_rd > 50:
        variants.extend([
            ('obs + photo_Rd', xi_obs, True, True),
            ('bar + photo_Rd', xi_bar, True, True),
            ('self + photo_Rd', xi_self, True, True),
        ])
    
    for name, xi_func, use_photo, needs_k in variants:
        print(f"\n--- {name} ---")
        
        if needs_k:
            k = optimize_k(galaxies, xi_func, use_photo)
            result = evaluate_model(galaxies, xi_func, use_photo, k=k)
            cv = kfold_cv(galaxies, xi_func, use_photo)
            result['k'] = k
            result['cv'] = cv
            print(f"  Optimal k: {k:.3f}")
            print(f"  Mean RMS: {result['mean_rms']:.2f} km/s")
            print(f"  CV Test RMS: {cv['test_rms_mean']:.2f} ± {cv['test_rms_std']:.2f}")
            print(f"  Overfitting: {cv['overfitting']:+.2f} km/s")
        else:
            result = evaluate_model(galaxies, xi_func, use_photo)
            result['k'] = None
            result['cv'] = None
            print(f"  Mean RMS: {result['mean_rms']:.2f} km/s")
        
        results[name] = result
    
    # ==========================================================================
    # Summary comparison
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    baseline_rms = results['baseline']['mean_rms']
    
    print(f"\n{'Variant':<20} {'Mean RMS':>10} {'vs Baseline':>12} {'k':>8} {'CV gap':>10}")
    print("-" * 65)
    
    for name, result in results.items():
        rms = result['mean_rms']
        improvement = (baseline_rms - rms) / baseline_rms * 100
        k_str = f"{result['k']:.3f}" if result['k'] else "N/A"
        cv_gap = f"{result['cv']['overfitting']:+.2f}" if result['cv'] else "N/A"
        print(f"{name:<20} {rms:>10.2f} {improvement:>+12.1f}% {k_str:>8} {cv_gap:>10}")
    
    # ==========================================================================
    # Key question: Does bar/self retain the improvement?
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does improvement hold without V_obs?")
    print("=" * 80)
    
    obs_rms = results['obs (V_obs)']['mean_rms']
    bar_rms = results['bar (V_bar)']['mean_rms']
    self_rms = results['self (V_pred)']['mean_rms']
    
    obs_improvement = (baseline_rms - obs_rms) / baseline_rms * 100
    bar_improvement = (baseline_rms - bar_rms) / baseline_rms * 100
    self_improvement = (baseline_rms - self_rms) / baseline_rms * 100
    
    print(f"\nBaseline (ξ = 2/3 R_d): {baseline_rms:.2f} km/s")
    print(f"V_obs-based:            {obs_rms:.2f} km/s ({obs_improvement:+.1f}%)")
    print(f"V_bar-based:            {bar_rms:.2f} km/s ({bar_improvement:+.1f}%)")
    print(f"Self-consistent:        {self_rms:.2f} km/s ({self_improvement:+.1f}%)")
    
    # Retention ratio
    if obs_improvement > 0:
        bar_retention = bar_improvement / obs_improvement * 100
        self_retention = self_improvement / obs_improvement * 100
        print(f"\nImprovement retention:")
        print(f"  V_bar retains: {bar_retention:.0f}% of V_obs improvement")
        print(f"  Self retains:  {self_retention:.0f}% of V_obs improvement")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if bar_improvement > 10 and self_improvement > 10:
        print("""
✓ IMPROVEMENT IS ROBUST

The dynamical coherence scale ξ = k × σ_eff / Ω_d shows substantial
improvement even when Ω_d is computed from baryons only or self-consistently.

This means:
1. The improvement is NOT circular (not just fitting V_obs)
2. The formula can be used predictively (baryons → prediction)
3. Safe to update the core theory

Recommended formula for predictive use:
  ξ = k × σ_eff / Ω_d  where  Ω_d = V_bar(R_d) / R_d
""")
    elif bar_improvement > 5:
        print("""
○ PARTIAL IMPROVEMENT

The V_bar-based formula retains some improvement but less than V_obs-based.
This suggests:
1. Part of the improvement is from using V_obs (somewhat circular)
2. But there's still a real effect from the dynamical scaling
3. Consider using V_bar-based for conservative estimates
""")
    else:
        print("""
✗ IMPROVEMENT DEPENDS ON V_OBS

The improvement largely disappears when using V_bar or self-consistent Ω_d.
This suggests:
1. The improvement may be partially circular
2. The dynamical timescale correlation is real but the formula needs work
3. Keep the baseline ξ = (2/3)R_d for now
""")
    
    # Save results
    output_dir = Path(__file__).parent / "dynamical_coherence_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable
    def to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        return obj
    
    summary = {
        'baseline_rms': baseline_rms,
        'variants': {name: to_json(result) for name, result in results.items()},
        'improvement_retention': {
            'bar_vs_obs': bar_improvement / obs_improvement * 100 if obs_improvement > 0 else 0,
            'self_vs_obs': self_improvement / obs_improvement * 100 if obs_improvement > 0 else 0
        }
    }
    
    with open(output_dir / "robustness_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

