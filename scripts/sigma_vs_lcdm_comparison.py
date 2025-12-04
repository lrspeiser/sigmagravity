#!/usr/bin/env python3
"""
Σ-Gravity vs ΛCDM Comparison Script

Performs systematic comparison of Σ-Gravity against ΛCDM (NFW dark matter halos)
on the SPARC galaxy sample. Implements the methodology described in the paper.

This script provides:
1. Fair head-to-head comparison with equal numbers of free parameters
2. Bootstrap uncertainty estimation
3. RAR scatter comparison
4. Detailed output for reproducibility

Usage:
    python sigma_vs_lcdm_comparison.py [--n_galaxies N] [--output_dir DIR] [--bootstrap N]

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.special import ellipk
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

G_KPC = 4.30091e-6  # G in (km/s)² kpc / M_sun
C_KMS = 299792.458  # Speed of light in km/s
H0_KMS_MPC = 70.0   # Hubble constant km/s/Mpc

# Σ-Gravity parameters (from paper)
G_DAGGER = 1.14e-10  # Critical acceleration m/s² = cH₀/6
A_DISK = np.sqrt(3)  # Enhancement amplitude for disks
P_COH = 0.75         # Coherence exponent
N_COH = 0.5          # Coherence decay exponent

# Convert g† to (km/s)²/kpc units
# 1 m/s² = 1e-6 km²/s² / (3.086e16 km) = 3.24e-23 (km/s)²/kpc
G_DAGGER_KPC = G_DAGGER * 3.24e-14  # (km/s)²/kpc


# =============================================================================
# DATA LOADING
# =============================================================================

def find_sparc_data() -> Path:
    """Find SPARC data directory."""
    possible_paths = [
        Path(__file__).parent.parent / "data" / "sparc" / "Rotmod_LTG",
        Path(__file__).parent.parent / "vendor" / "sparc" / "Rotmod_LTG",
        Path(__file__).parent.parent / "coherence-field-theory" / "data" / "sparc" / "Rotmod_LTG",
    ]
    
    for p in possible_paths:
        if p.exists():
            return p
    
    raise FileNotFoundError(
        "SPARC Rotmod_LTG data not found. Please download from "
        "http://astroweb.cwru.edu/SPARC/ and place in data/sparc/Rotmod_LTG/"
    )


def load_sparc_galaxy(filepath: Path) -> Dict:
    """
    Load a single SPARC galaxy rotation curve.
    
    Returns dict with:
        r: radius (kpc)
        v_obs: observed velocity (km/s)
        v_err: velocity error (km/s)
        v_gas: gas contribution (km/s)
        v_disk: disk contribution (km/s)
        v_bulge: bulge contribution (km/s)
        v_bar: total baryonic velocity (km/s)
        g_bar: baryonic acceleration ((km/s)²/kpc)
        g_obs: observed acceleration ((km/s)²/kpc)
    """
    data = np.loadtxt(filepath, comments='#')
    
    r = data[:, 0]
    v_obs = data[:, 1]
    v_err = data[:, 2]
    v_gas = data[:, 3]
    v_disk = data[:, 4]
    v_bulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(r)
    
    # Total baryonic velocity
    v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
    
    # Accelerations
    mask = r > 0
    g_bar = np.zeros_like(r)
    g_obs = np.zeros_like(r)
    g_bar[mask] = v_bar[mask]**2 / r[mask]
    g_obs[mask] = v_obs[mask]**2 / r[mask]
    
    return {
        'name': filepath.stem,
        'r': r,
        'v_obs': v_obs,
        'v_err': np.maximum(v_err, 0.1),  # Floor on errors
        'v_gas': v_gas,
        'v_disk': v_disk,
        'v_bulge': v_bulge,
        'v_bar': v_bar,
        'g_bar': g_bar,
        'g_obs': g_obs,
    }


def load_all_sparc(data_dir: Path, max_galaxies: Optional[int] = None) -> List[Dict]:
    """Load all SPARC galaxies."""
    galaxies = []
    
    files = sorted(data_dir.glob("*.dat"))
    if max_galaxies:
        files = files[:max_galaxies]
    
    for f in files:
        try:
            gal = load_sparc_galaxy(f)
            if len(gal['r']) >= 5:  # Need at least 5 points
                galaxies.append(gal)
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")
    
    print(f"Loaded {len(galaxies)} galaxies from SPARC")
    return galaxies


# =============================================================================
# Σ-GRAVITY MODEL
# =============================================================================

def sigma_gravity_enhancement(r: np.ndarray, g_bar: np.ndarray, 
                               A: float, xi: float) -> np.ndarray:
    """
    Compute Σ-Gravity enhancement factor.
    
    Σ = 1 + A × W(r) × h(g)
    
    where:
        W(r) = 1 - (1 + (r/ξ)^p)^(-n_coh)  [coherence window]
        h(g) = √(g†/g) × g†/(g†+g)          [acceleration dependence]
    
    Parameters:
        r: radius (kpc)
        g_bar: baryonic acceleration ((km/s)²/kpc)
        A: amplitude (fitted)
        xi: coherence scale (kpc, fitted)
    
    Returns:
        Σ: enhancement factor (dimensionless)
    """
    # Coherence window
    W = 1.0 - (1.0 + (r / xi)**P_COH)**(-N_COH)
    
    # Acceleration dependence
    # Convert g_bar to m/s² for comparison with g†
    g_bar_si = g_bar * 3.086e16 / 1e6  # (km/s)²/kpc → m/s²
    
    # Avoid division by zero
    g_safe = np.maximum(g_bar_si, 1e-15)
    
    h = np.sqrt(G_DAGGER / g_safe) * (G_DAGGER / (G_DAGGER + g_safe))
    
    # Enhancement factor
    Sigma = 1.0 + A * W * h
    
    return Sigma


def sigma_gravity_velocity(r: np.ndarray, v_bar: np.ndarray, 
                            A: float, xi: float) -> np.ndarray:
    """
    Compute Σ-Gravity predicted velocity.
    
    v_pred = v_bar × √Σ
    """
    g_bar = v_bar**2 / np.maximum(r, 0.01)
    Sigma = sigma_gravity_enhancement(r, g_bar, A, xi)
    
    return v_bar * np.sqrt(Sigma)


def fit_sigma_gravity(gal: Dict, method: str = 'global') -> Dict:
    """
    Fit Σ-Gravity model to a galaxy.
    
    Free parameters: A (amplitude), ξ (coherence scale)
    Fixed parameters: g†, p, n_coh (from theory)
    
    Returns dict with best-fit parameters and diagnostics.
    """
    r = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    v_bar = gal['v_bar']
    
    # Estimate scale from data
    R_d_est = np.median(r) / 2.5
    
    def chi_squared(params):
        A, xi = params
        if A < 0 or xi < 0.1:
            return 1e10
        
        v_pred = sigma_gravity_velocity(r, v_bar, A, xi)
        chi2 = np.sum(((v_obs - v_pred) / v_err)**2)
        return chi2
    
    # Bounds: A ∈ [0, 5], ξ ∈ [0.1, 50] kpc
    bounds = [(0.01, 5.0), (0.1, 50.0)]
    
    if method == 'global':
        result = differential_evolution(chi_squared, bounds, 
                                       maxiter=200, seed=42, 
                                       workers=1, polish=True)
        best_params = result.x
        chi2_min = result.fun
    else:
        p0 = [A_DISK, R_d_est]
        result = minimize(chi_squared, p0, method='L-BFGS-B', bounds=bounds)
        best_params = result.x
        chi2_min = result.fun
    
    # Degrees of freedom
    n_params = 2
    dof = len(r) - n_params
    chi2_red = chi2_min / dof if dof > 0 else chi2_min
    
    # Compute predicted values
    v_pred = sigma_gravity_velocity(r, v_bar, best_params[0], best_params[1])
    
    return {
        'model': 'Sigma-Gravity',
        'params': {'A': best_params[0], 'xi': best_params[1]},
        'chi2': chi2_min,
        'chi2_red': chi2_red,
        'dof': dof,
        'n_params': n_params,
        'v_pred': v_pred,
        'residuals': (v_obs - v_pred) / v_err,
    }


# =============================================================================
# ΛCDM / NFW MODEL
# =============================================================================

def nfw_velocity(r: np.ndarray, M200: float, c: float) -> np.ndarray:
    """
    Compute NFW halo circular velocity.
    
    v²(r) = G M(<r) / r
    
    where M(<r) = 4π ρ_s r_s³ [ln(1+x) - x/(1+x)]
    and x = r/r_s, r_s = r_200/c
    
    Parameters:
        r: radius (kpc)
        M200: virial mass (M_sun)
        c: concentration parameter
    
    Returns:
        v_NFW: NFW contribution to circular velocity (km/s)
    """
    # Virial radius (approximate)
    # r_200 ≈ (3 M200 / (4π × 200 × ρ_crit))^(1/3)
    # For ρ_crit ≈ 140 M_sun/kpc³:
    rho_crit = 140.0  # M_sun/kpc³
    r_200 = (3 * M200 / (4 * np.pi * 200 * rho_crit))**(1/3)
    
    r_s = r_200 / c
    
    # NFW scale density
    f_c = np.log(1 + c) - c / (1 + c)
    rho_s = M200 / (4 * np.pi * r_s**3 * f_c)
    
    # Circular velocity
    x = r / r_s
    x = np.maximum(x, 0.001)  # Avoid numerical issues
    
    M_enc = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))
    
    v_NFW = np.sqrt(G_KPC * M_enc / r)
    
    return v_NFW


def lcdm_velocity(r: np.ndarray, v_bar: np.ndarray, 
                  M200: float, c: float) -> np.ndarray:
    """
    Compute total ΛCDM velocity (baryons + NFW halo).
    
    v² = v_bar² + v_NFW²
    """
    v_NFW = nfw_velocity(r, M200, c)
    return np.sqrt(v_bar**2 + v_NFW**2)


def fit_lcdm(gal: Dict, method: str = 'global') -> Dict:
    """
    Fit ΛCDM (NFW) model to a galaxy.
    
    Free parameters: M200 (virial mass), c (concentration)
    
    Note: We use the same baryonic model as Σ-Gravity for fair comparison.
    
    Returns dict with best-fit parameters and diagnostics.
    """
    r = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    v_bar = gal['v_bar']
    
    # Estimate halo mass from velocity
    v_max = np.max(v_obs)
    M200_est = (v_max**2 * np.max(r) / G_KPC) * 10  # Rough estimate
    
    def chi_squared(params):
        log_M200, c = params
        M200 = 10**log_M200
        
        if M200 < 1e6 or c < 1 or c > 50:
            return 1e10
        
        v_pred = lcdm_velocity(r, v_bar, M200, c)
        chi2 = np.sum(((v_obs - v_pred) / v_err)**2)
        return chi2
    
    # Bounds: M200 ∈ [10^6, 10^14] M_sun, c ∈ [1, 50]
    bounds = [(6.0, 14.0), (1.0, 50.0)]
    
    if method == 'global':
        result = differential_evolution(chi_squared, bounds, 
                                       maxiter=200, seed=42, 
                                       workers=1, polish=True)
        best_params = result.x
        chi2_min = result.fun
    else:
        p0 = [np.log10(M200_est), 10.0]
        result = minimize(chi_squared, p0, method='L-BFGS-B', bounds=bounds)
        best_params = result.x
        chi2_min = result.fun
    
    # Degrees of freedom
    n_params = 2
    dof = len(r) - n_params
    chi2_red = chi2_min / dof if dof > 0 else chi2_min
    
    # Compute predicted values
    M200 = 10**best_params[0]
    c = best_params[1]
    v_pred = lcdm_velocity(r, v_bar, M200, c)
    
    return {
        'model': 'LCDM-NFW',
        'params': {'log_M200': best_params[0], 'M200': M200, 'c': c},
        'chi2': chi2_min,
        'chi2_red': chi2_red,
        'dof': dof,
        'n_params': n_params,
        'v_pred': v_pred,
        'residuals': (v_obs - v_pred) / v_err,
    }


# =============================================================================
# RAR ANALYSIS
# =============================================================================

def compute_rar_scatter(galaxies: List[Dict], model: str = 'sigma') -> Dict:
    """
    Compute Radial Acceleration Relation scatter for a model.
    
    RAR: g_obs vs g_bar
    Scatter is computed as RMS of log10(g_obs/g_pred)
    
    Returns dict with scatter statistics.
    """
    all_g_bar = []
    all_g_obs = []
    all_g_pred = []
    
    for gal in galaxies:
        r = gal['r']
        v_bar = gal['v_bar']
        v_obs = gal['v_obs']
        
        mask = (r > 0) & (v_bar > 0) & (v_obs > 0)
        
        g_bar = v_bar[mask]**2 / r[mask]
        g_obs = v_obs[mask]**2 / r[mask]
        
        if model == 'sigma':
            fit = fit_sigma_gravity(gal)
            v_pred = fit['v_pred']
        else:
            fit = fit_lcdm(gal)
            v_pred = fit['v_pred']
        
        g_pred = v_pred[mask]**2 / r[mask]
        
        all_g_bar.extend(g_bar)
        all_g_obs.extend(g_obs)
        all_g_pred.extend(g_pred)
    
    g_bar = np.array(all_g_bar)
    g_obs = np.array(all_g_obs)
    g_pred = np.array(all_g_pred)
    
    # RAR scatter
    log_ratio = np.log10(g_obs / g_pred)
    scatter = np.std(log_ratio)
    
    # Also compute intrinsic scatter (relative to mean relation)
    log_obs_bar = np.log10(g_obs / g_bar)
    intrinsic_scatter = np.std(log_obs_bar)
    
    return {
        'scatter_dex': scatter,
        'intrinsic_scatter_dex': intrinsic_scatter,
        'n_points': len(g_bar),
        'g_bar_range': (g_bar.min(), g_bar.max()),
        'g_obs_range': (g_obs.min(), g_obs.max()),
    }


# =============================================================================
# COMPARISON ANALYSIS
# =============================================================================

def compare_models(galaxies: List[Dict], 
                   output_dir: Path,
                   n_bootstrap: int = 100) -> Dict:
    """
    Comprehensive comparison of Σ-Gravity vs ΛCDM.
    
    Performs:
    1. Per-galaxy fits for both models
    2. χ² comparison
    3. RAR scatter comparison
    4. Bootstrap uncertainty estimation
    
    Returns dict with all comparison results.
    """
    results = {
        'galaxies': [],
        'sigma_wins': 0,
        'lcdm_wins': 0,
        'ties': 0,
        'sigma_chi2_red_all': [],
        'lcdm_chi2_red_all': [],
    }
    
    print(f"\nFitting {len(galaxies)} galaxies...")
    print("-" * 70)
    
    for i, gal in enumerate(galaxies):
        print(f"  [{i+1}/{len(galaxies)}] {gal['name']}", end=" ")
        
        try:
            # Fit both models
            sigma_fit = fit_sigma_gravity(gal)
            lcdm_fit = fit_lcdm(gal)
            
            # Determine winner
            ratio = sigma_fit['chi2_red'] / lcdm_fit['chi2_red']
            if ratio < 0.95:
                winner = 'Sigma'
                results['sigma_wins'] += 1
            elif ratio > 1.05:
                winner = 'LCDM'
                results['lcdm_wins'] += 1
            else:
                winner = 'Tie'
                results['ties'] += 1
            
            results['sigma_chi2_red_all'].append(sigma_fit['chi2_red'])
            results['lcdm_chi2_red_all'].append(lcdm_fit['chi2_red'])
            
            results['galaxies'].append({
                'name': gal['name'],
                'n_points': len(gal['r']),
                'sigma': sigma_fit,
                'lcdm': lcdm_fit,
                'ratio': ratio,
                'winner': winner,
            })
            
            print(f"χ²_red: Σ={sigma_fit['chi2_red']:.2f}, ΛCDM={lcdm_fit['chi2_red']:.2f} → {winner}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Summary statistics
    sigma_chi2 = np.array(results['sigma_chi2_red_all'])
    lcdm_chi2 = np.array(results['lcdm_chi2_red_all'])
    
    results['summary'] = {
        'n_galaxies': len(results['galaxies']),
        'sigma_wins': results['sigma_wins'],
        'lcdm_wins': results['lcdm_wins'],
        'ties': results['ties'],
        'sigma_chi2_red_mean': np.mean(sigma_chi2),
        'sigma_chi2_red_median': np.median(sigma_chi2),
        'lcdm_chi2_red_mean': np.mean(lcdm_chi2),
        'lcdm_chi2_red_median': np.median(lcdm_chi2),
        'ratio_mean': np.mean(sigma_chi2 / lcdm_chi2),
    }
    
    # Bootstrap uncertainty on win rate
    if n_bootstrap > 0:
        print(f"\nBootstrap analysis ({n_bootstrap} iterations)...")
        win_rates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(sigma_chi2), len(sigma_chi2), replace=True)
            wins = np.sum(sigma_chi2[idx] < lcdm_chi2[idx])
            win_rates.append(wins / len(sigma_chi2))
        
        results['bootstrap'] = {
            'sigma_win_rate_mean': np.mean(win_rates),
            'sigma_win_rate_std': np.std(win_rates),
            'sigma_win_rate_95ci': (np.percentile(win_rates, 2.5), 
                                    np.percentile(win_rates, 97.5)),
        }
    
    return results


def create_comparison_plots(results: Dict, output_dir: Path):
    """Create comparison visualizations."""
    
    # 1. χ² comparison scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sigma_chi2 = np.array(results['sigma_chi2_red_all'])
    lcdm_chi2 = np.array(results['lcdm_chi2_red_all'])
    
    ax = axes[0]
    ax.scatter(lcdm_chi2, sigma_chi2, alpha=0.6, s=30)
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.5, label='Equal fit')
    ax.set_xlabel('ΛCDM χ²_red', fontsize=12)
    ax.set_ylabel('Σ-Gravity χ²_red', fontsize=12)
    ax.set_title('Fit Quality Comparison', fontsize=14)
    ax.set_xlim(0, min(10, np.percentile(lcdm_chi2, 95) * 1.5))
    ax.set_ylim(0, min(10, np.percentile(sigma_chi2, 95) * 1.5))
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add text summary
    n_total = len(sigma_chi2)
    text = (f"Σ-Gravity wins: {results['sigma_wins']}/{n_total} ({100*results['sigma_wins']/n_total:.0f}%)\n"
            f"ΛCDM wins: {results['lcdm_wins']}/{n_total} ({100*results['lcdm_wins']/n_total:.0f}%)\n"
            f"Ties: {results['ties']}/{n_total}")
    ax.text(0.95, 0.05, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Histogram of χ² ratios
    ax = axes[1]
    ratios = sigma_chi2 / lcdm_chi2
    ax.hist(ratios, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal fit')
    ax.axvline(np.median(ratios), color='blue', linestyle='-', linewidth=2, 
               label=f'Median = {np.median(ratios):.2f}')
    ax.set_xlabel('χ²_red(Σ-Gravity) / χ²_red(ΛCDM)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Fit Quality Ratios', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sigma_vs_lcdm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_dir / 'sigma_vs_lcdm_comparison.png'}")


def export_results(results: Dict, output_dir: Path):
    """Export results to CSV and JSON."""
    
    # CSV with per-galaxy results
    rows = []
    for gal_result in results['galaxies']:
        rows.append({
            'galaxy': gal_result['name'],
            'n_points': gal_result['n_points'],
            'sigma_chi2_red': gal_result['sigma']['chi2_red'],
            'sigma_A': gal_result['sigma']['params']['A'],
            'sigma_xi': gal_result['sigma']['params']['xi'],
            'lcdm_chi2_red': gal_result['lcdm']['chi2_red'],
            'lcdm_log_M200': gal_result['lcdm']['params']['log_M200'],
            'lcdm_c': gal_result['lcdm']['params']['c'],
            'ratio': gal_result['ratio'],
            'winner': gal_result['winner'],
        })
    
    df = pd.DataFrame(rows)
    csv_path = output_dir / 'sigma_vs_lcdm_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved per-galaxy results to {csv_path}")
    
    # JSON with full results
    json_results = {
        'summary': results['summary'],
        'bootstrap': results.get('bootstrap', {}),
        'parameters': {
            'sigma_gravity': {
                'g_dagger_m_s2': G_DAGGER,
                'A_disk': A_DISK,
                'p_coh': P_COH,
                'n_coh': N_COH,
            },
            'lcdm': {
                'profile': 'NFW',
                'rho_crit_Msun_kpc3': 140.0,
            }
        }
    }
    
    json_path = output_dir / 'sigma_vs_lcdm_summary.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved summary to {json_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Σ-Gravity vs ΛCDM Comparison')
    parser.add_argument('--n_galaxies', type=int, default=None,
                       help='Number of galaxies to fit (default: all)')
    parser.add_argument('--output_dir', type=str, default='outputs/comparison',
                       help='Output directory')
    parser.add_argument('--bootstrap', type=int, default=100,
                       help='Number of bootstrap iterations')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Σ-GRAVITY vs ΛCDM COMPARISON")
    print("=" * 70)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n1. Loading SPARC data...")
    data_dir = find_sparc_data()
    galaxies = load_all_sparc(data_dir, max_galaxies=args.n_galaxies)
    
    # Run comparison
    print("\n2. Running model comparison...")
    start_time = time.time()
    results = compare_models(galaxies, output_dir, n_bootstrap=args.bootstrap)
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal galaxies: {results['summary']['n_galaxies']}")
    print(f"Σ-Gravity wins: {results['summary']['sigma_wins']} ({100*results['summary']['sigma_wins']/results['summary']['n_galaxies']:.1f}%)")
    print(f"ΛCDM wins: {results['summary']['lcdm_wins']} ({100*results['summary']['lcdm_wins']/results['summary']['n_galaxies']:.1f}%)")
    print(f"Ties: {results['summary']['ties']}")
    print(f"\nMean χ²_red:")
    print(f"  Σ-Gravity: {results['summary']['sigma_chi2_red_mean']:.3f}")
    print(f"  ΛCDM: {results['summary']['lcdm_chi2_red_mean']:.3f}")
    print(f"  Ratio: {results['summary']['ratio_mean']:.3f}")
    
    if 'bootstrap' in results:
        print(f"\nBootstrap (95% CI):")
        print(f"  Σ-Gravity win rate: {results['bootstrap']['sigma_win_rate_mean']:.1%} "
              f"({results['bootstrap']['sigma_win_rate_95ci'][0]:.1%}, "
              f"{results['bootstrap']['sigma_win_rate_95ci'][1]:.1%})")
    
    print(f"\nElapsed time: {elapsed:.1f}s")
    
    # Create outputs
    print("\n3. Creating outputs...")
    create_comparison_plots(results, output_dir)
    export_results(results, output_dir)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()

