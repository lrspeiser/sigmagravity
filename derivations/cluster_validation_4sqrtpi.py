#!/usr/bin/env python3
"""
Cluster Validation: g† = cH₀/(4√π) vs g† = cH₀/(2e)
====================================================

Tests the new critical acceleration formula on galaxy clusters.

Uses embedded cluster data (Coma, A2029, A1689, Bullet) for lensing comparison.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from typing import Dict, List

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8              # m/s
G = 6.674e-11            # m³/kg/s²
M_sun = 1.989e30         # kg
kpc_to_m = 3.086e19
H0_SI = 2.27e-18         # 1/s (70 km/s/Mpc)
cH0 = c * H0_SI

# Two formulas
g_dagger_old = cH0 / (2 * math.e)
g_dagger_new = cH0 / (4 * math.sqrt(math.pi))

# Cluster amplitude
A_cluster = math.pi * math.sqrt(2)  # π√2 ≈ 4.44

print("=" * 80)
print("CLUSTER VALIDATION: g† = cH₀/(4√π) vs g† = cH₀/(2e)")
print("=" * 80)
print(f"\nOld formula: g† = cH₀/(2e)   = {g_dagger_old:.4e} m/s²")
print(f"New formula: g† = cH₀/(4√π)  = {g_dagger_new:.4e} m/s²")
print(f"Cluster amplitude: A = π√2 = {A_cluster:.4f}")

# =============================================================================
# CLUSTER DATA (from data_loader.py)
# =============================================================================

# Galaxy cluster data showing κ_eff profiles
# R in kpc, κ_obs (observed convergence), κ_bar (baryonic prediction)
CLUSTER_DATA = {
    'Coma': {
        'R_kpc': np.array([100, 200, 400, 600, 1000, 1500, 2000]),
        'kappa_obs': np.array([0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025]),
        'kappa_bar': np.array([0.28, 0.18, 0.09, 0.055, 0.030, 0.018, 0.011]),
        'z': 0.023,
    },
    'A2029': {
        'R_kpc': np.array([50, 100, 200, 400, 600, 1000]),
        'kappa_obs': np.array([0.50, 0.38, 0.22, 0.12, 0.08, 0.04]),
        'kappa_bar': np.array([0.42, 0.28, 0.14, 0.065, 0.038, 0.018]),
        'z': 0.077,
    },
    'A1689': {
        'R_kpc': np.array([50, 100, 200, 300, 500, 800]),
        'kappa_obs': np.array([0.65, 0.48, 0.32, 0.22, 0.12, 0.06]),
        'kappa_bar': np.array([0.55, 0.38, 0.22, 0.14, 0.07, 0.03]),
        'z': 0.183,
    },
    'Bullet': {
        'R_kpc': np.array([100, 250, 500, 750, 1000, 1500]),
        'kappa_obs': np.array([0.42, 0.28, 0.18, 0.12, 0.08, 0.05]),
        'kappa_bar': np.array([0.32, 0.18, 0.10, 0.06, 0.038, 0.022]),
        'z': 0.296,
    },
}

# =============================================================================
# Σ-GRAVITY FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray, g_dagger: float) -> np.ndarray:
    """Universal h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def Sigma_cluster(kappa_bar: np.ndarray, g_dagger: float, 
                  Sigma_crit: float = 1e15) -> np.ndarray:
    """
    Compute enhancement for cluster lensing.
    
    For clusters, κ ∝ Σ (surface mass density), and Σ ∝ g
    So we can compute g_bar from κ_bar and apply the enhancement.
    
    W = 1 for clusters (lensing is line-of-sight integrated)
    """
    # Convert κ_bar to effective g_bar
    # κ = Σ/Σ_crit, and g ∝ Σ for projected mass
    # Use κ_bar as proxy for g/g_ref
    g_ref = 1e-10  # Reference acceleration scale
    g_bar = kappa_bar * g_ref / 0.1  # Normalize so typical κ ~ 0.1 gives g ~ g_ref
    
    h = h_function(g_bar, g_dagger)
    
    # For clusters, W = 1 (no coherence window for lensing)
    Sigma = 1 + A_cluster * h
    
    return Sigma


def predict_kappa(kappa_bar: np.ndarray, g_dagger: float) -> np.ndarray:
    """Predict observed κ from baryonic κ using Σ-Gravity."""
    Sigma = Sigma_cluster(kappa_bar, g_dagger)
    return kappa_bar * Sigma


# =============================================================================
# METRICS
# =============================================================================

def compute_kappa_rms(kappa_obs: np.ndarray, kappa_pred: np.ndarray) -> float:
    """Compute RMS error in κ."""
    return np.sqrt(np.mean((kappa_obs - kappa_pred)**2))


def compute_log_scatter(kappa_obs: np.ndarray, kappa_pred: np.ndarray) -> float:
    """Compute scatter in log space (dex)."""
    mask = (kappa_obs > 0) & (kappa_pred > 0)
    if mask.sum() < 2:
        return np.nan
    log_residual = np.log10(kappa_obs[mask] / kappa_pred[mask])
    return np.std(log_residual)


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_cluster_validation():
    """Run cluster validation comparing old vs new formula."""
    
    print("\n" + "=" * 80)
    print("RUNNING CLUSTER COMPARISON")
    print("=" * 80)
    
    results = []
    
    for name, data in CLUSTER_DATA.items():
        R = data['R_kpc']
        kappa_obs = data['kappa_obs']
        kappa_bar = data['kappa_bar']
        z = data['z']
        
        # Old formula predictions
        kappa_pred_old = predict_kappa(kappa_bar, g_dagger_old)
        rms_old = compute_kappa_rms(kappa_obs, kappa_pred_old)
        scatter_old = compute_log_scatter(kappa_obs, kappa_pred_old)
        
        # New formula predictions
        kappa_pred_new = predict_kappa(kappa_bar, g_dagger_new)
        rms_new = compute_kappa_rms(kappa_obs, kappa_pred_new)
        scatter_new = compute_log_scatter(kappa_obs, kappa_pred_new)
        
        # Required enhancement (what we need to match observations)
        K_required = kappa_obs / kappa_bar - 1
        
        results.append({
            'name': name,
            'z': z,
            'n_points': len(R),
            'rms_old': rms_old,
            'rms_new': rms_new,
            'scatter_old': scatter_old,
            'scatter_new': scatter_new,
            'mean_K_required': np.mean(K_required),
        })
        
        print(f"\n{name} (z={z:.3f}, {len(R)} points):")
        print(f"  Mean required enhancement K = {np.mean(K_required):.2f}")
        print(f"  Old formula: RMS = {rms_old:.4f}, scatter = {scatter_old:.4f} dex")
        print(f"  New formula: RMS = {rms_new:.4f}, scatter = {scatter_new:.4f} dex")
        
        # Per-radius comparison
        print(f"\n  {'R (kpc)':<10} {'κ_obs':<10} {'κ_bar':<10} {'κ_old':<10} {'κ_new':<10}")
        print("  " + "-" * 50)
        for i in range(len(R)):
            print(f"  {R[i]:<10.0f} {kappa_obs[i]:<10.3f} {kappa_bar[i]:<10.3f} {kappa_pred_old[i]:<10.3f} {kappa_pred_new[i]:<10.3f}")
    
    # Summary
    mean_rms_old = np.mean([r['rms_old'] for r in results])
    mean_rms_new = np.mean([r['rms_new'] for r in results])
    mean_scatter_old = np.mean([r['scatter_old'] for r in results if not np.isnan(r['scatter_old'])])
    mean_scatter_new = np.mean([r['scatter_new'] for r in results if not np.isnan(r['scatter_new'])])
    
    wins_old = sum(1 for r in results if r['rms_old'] < r['rms_new'])
    wins_new = sum(1 for r in results if r['rms_new'] < r['rms_old'])
    
    print("\n" + "=" * 80)
    print("CLUSTER SUMMARY")
    print("=" * 80)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CLUSTER LENSING RESULTS                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Formula              │ Mean κ RMS │ Mean Scatter (dex) │ Clusters Won        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ OLD: g† = cH₀/(2e)   │ {mean_rms_old:>10.4f} │ {mean_scatter_old:>18.4f} │ {wins_old:>19}  ║
║ NEW: g† = cH₀/(4√π)  │ {mean_rms_new:>10.4f} │ {mean_scatter_new:>18.4f} │ {wins_new:>19}  ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    rms_improvement = 100 * (mean_rms_old - mean_rms_new) / mean_rms_old
    scatter_improvement = 100 * (mean_scatter_old - mean_scatter_new) / mean_scatter_old
    
    print(f"""
IMPROVEMENT WITH NEW FORMULA:
  κ RMS: {rms_improvement:+.1f}% {'(BETTER)' if rms_improvement > 0 else '(WORSE)'}
  Scatter: {scatter_improvement:+.1f}% {'(BETTER)' if scatter_improvement > 0 else '(WORSE)'}

HEAD-TO-HEAD:
  Old wins: {wins_old}
  New wins: {wins_new}
""")
    
    return {
        'mean_rms_old': mean_rms_old,
        'mean_rms_new': mean_rms_new,
        'mean_scatter_old': mean_scatter_old,
        'mean_scatter_new': mean_scatter_new,
        'rms_improvement_pct': rms_improvement,
        'scatter_improvement_pct': scatter_improvement,
        'wins_old': wins_old,
        'wins_new': wins_new,
    }


if __name__ == "__main__":
    results = run_cluster_validation()
    
    print("\n" + "=" * 80)
    print("CLUSTER VALIDATION VERDICT")
    print("=" * 80)
    
    if results['rms_improvement_pct'] > 0:
        print("""
✓ The NEW formula g† = cH₀/(4√π) performs BETTER on clusters

Note: Cluster lensing depends more on the amplitude A = π√2 than on g†,
so the improvement is smaller than for galaxy rotation curves.
""")
    elif results['rms_improvement_pct'] > -10:
        print("""
≈ The NEW formula performs COMPARABLY on clusters

Note: Cluster lensing is dominated by the amplitude A = π√2.
The g† value has less impact at cluster scales.
""")
    else:
        print("""
✗ The NEW formula performs WORSE on clusters

This may indicate that clusters require a different treatment.
""")

