#!/usr/bin/env python3
"""
Test Suite: Validating n_coh = 0.5 Derivation Approaches
========================================================

This script tests the 5 theoretical approaches from N_COH_DERIVATION.md
to determine which (if any) are consistent with real data.

Approaches:
1. Random Walk / Coherence Patches: K(R) = 1/√(N_patches) = 1/√(1 + R/ℓ₀)
2. 2D Thin Disk Geometry: Effective dimensionality argument
3. Diffusive Phase / Lévy: Cauchy distribution (α=1) → power-law decay
4. Anomalous Diffusion: Marginally anomalous (β=0)
5. Effective Dimension: d_eff ≈ 3 for n=0.5

Tests performed:
- Power-law vs exponential correlation decay
- Fitted n_coh value and uncertainty
- Lévy index estimation from heavy tails
- Anomalous diffusion exponent estimation
- Model comparison (χ², BIC, AIC)

See: theory/coherence_exponent_derivation/N_COH_DERIVATION.md
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr, levy_stable, norm
from scipy.spatial import cKDTree
from scipy.special import gamma as gamma_func
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ========== MODEL DEFINITIONS ==========

def K_powerlaw(r, ell0, n_coh):
    """
    Approach 1: Power-law coherence kernel (random walk patches).
    K(R) = (ℓ₀/(ℓ₀+R))^n_coh
    
    For n_coh = 0.5: K = 1/√(1 + R/ℓ₀) = 1/√(N_patches)
    """
    r = np.asarray(r)
    return (ell0 / (ell0 + r)) ** n_coh


def K_powerlaw_full(r, A, ell0, n_coh):
    """Power-law with amplitude for fitting."""
    return A * K_powerlaw(r, ell0, n_coh)


def K_exponential(r, ell0, n_coh):
    """
    Approach 3 (Gaussian case): Exponential decay from diffusive phase evolution.
    K(R) = exp(-n_coh × R / ℓ₀)
    
    This is what standard diffusion predicts (too aggressive decay).
    """
    r = np.asarray(r)
    return np.exp(-n_coh * r / ell0)


def K_exponential_full(r, A, ell0, n_coh):
    """Exponential with amplitude for fitting."""
    return A * K_exponential(r, ell0, n_coh)


def K_stretched_exp(r, A, ell0, beta):
    """
    Stretched exponential (Kohlrausch-Williams-Watts).
    K(R) = A × exp(-(R/ℓ₀)^β)
    
    β < 1: slower than exponential (anomalous)
    β = 1: standard exponential
    β = 0.5: matches power-law at intermediate scales
    """
    r = np.asarray(r)
    return A * np.exp(-(r / ell0) ** beta)


def K_levy(r, ell0, alpha):
    """
    Approach 3 (Lévy case): Power-law decay from Lévy flights.
    K(R) ∝ (ℓ₀/R)^(α/2)
    
    For α = 1 (Cauchy): K ∝ R^(-0.5) → n_coh = 0.5
    """
    r = np.asarray(r)
    # Regularized form to avoid singularity at r=0
    return (ell0 / (ell0 + r)) ** (alpha / 2)


def K_anomalous(r, ell0, beta_mem):
    """
    Approach 4: Anomalous diffusion from memory effects.
    K(R) ~ R^(-(1-β)/2)
    
    For β = 0 (no memory decay): K ~ R^(-0.5) → n_coh = 0.5
    """
    r = np.asarray(r)
    exponent = (1 - beta_mem) / 2
    return (ell0 / (ell0 + r)) ** exponent


def N_patches_1D(r, ell0):
    """
    Approach 1: Number of coherence patches (1D).
    N = 1 + R/ℓ₀
    
    Coherence fraction: K = 1/√N
    """
    r = np.asarray(r)
    return 1 + r / ell0


# ========== DATA LOADING ==========

def load_gaia_data():
    """Load Gaia data for correlation analysis."""
    # Try processed corrected data first (1.8M stars)
    gaia_path = ROOT / "data" / "gaia" / "gaia_processed_corrected.csv"
    
    if not gaia_path.exists():
        gaia_path = ROOT / "data" / "gaia" / "gaia_processed.csv"
    
    if not gaia_path.exists():
        print(f"ERROR: Gaia data not found at {gaia_path}")
        return None
    
    print(f"Loading Gaia data from: {gaia_path}")
    df = pd.read_csv(gaia_path)
    print(f"  Total stars: {len(df):,}")
    
    # Map columns
    if 'R_cyl' in df.columns:
        df['R_kpc'] = df['R_cyl']
    if 'z' in df.columns:
        df['z_kpc'] = df['z']
    
    # Compute velocity residuals from flat rotation curve approximation
    v_flat = 220.0  # km/s
    if 'v_phi' in df.columns:
        df['v_obs'] = df['v_phi']
        df['delta_v'] = df['v_phi'] - v_flat
    elif 'v_tangential' in df.columns:
        df['v_obs'] = df['v_tangential']
        df['delta_v'] = df['v_tangential'] - v_flat
    else:
        print("ERROR: No velocity column found")
        return None
    
    # Filter to disk stars
    disk_mask = (df['R_kpc'] >= 4) & (df['R_kpc'] <= 16) & (np.abs(df['z_kpc']) < 1.0)
    df_disk = df[disk_mask].copy()
    print(f"  Disk stars (4 < R < 16 kpc, |z| < 1 kpc): {len(df_disk):,}")
    
    return df_disk


def compute_velocity_correlations(df, n_sample=50000, r_bins=None, seed=42):
    """
    Compute velocity correlation function C(r) from star pairs.
    
    C(r) = <δv(R) δv(R')> / σ_v²
    
    where δv = v_obs - v_expected
    """
    np.random.seed(seed)
    
    if r_bins is None:
        r_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    
    # Subsample if needed
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
        print(f"  Subsampled to {n_sample:,} stars for correlation analysis")
    
    # Get coordinates and residuals
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    delta_v = df['delta_v'].values
    
    # Normalize residuals
    sigma_v = np.std(delta_v)
    delta_v_norm = delta_v / sigma_v
    print(f"  Velocity dispersion: σ_v = {sigma_v:.1f} km/s")
    
    # Build KD-tree for efficient pair finding
    coords = np.column_stack([R, z])
    tree = cKDTree(coords)
    
    # Find pairs within max separation
    r_max = r_bins[-1] * 1.1
    pairs = tree.query_pairs(r=r_max, output_type='ndarray')
    
    if len(pairs) == 0:
        print("  WARNING: No pairs found!")
        return None, None, None, None
    
    print(f"  Found {len(pairs):,} pairs within {r_max:.1f} kpc")
    
    # Compute separations
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    
    dR = R[i_idx] - R[j_idx]
    dz = z[i_idx] - z[j_idx]
    separations = np.sqrt(dR**2 + dz**2)
    
    # Compute products of normalized residuals
    products = delta_v_norm[i_idx] * delta_v_norm[j_idx]
    
    # Bin by separation
    r_centers = []
    correlations = []
    correlation_errs = []
    n_pairs_list = []
    
    for i in range(len(r_bins) - 1):
        r_lo, r_hi = r_bins[i], r_bins[i + 1]
        mask = (separations >= r_lo) & (separations < r_hi)
        
        n_pairs = np.sum(mask)
        if n_pairs < 100:
            continue
        
        corr = np.mean(products[mask])
        corr_err = np.std(products[mask]) / np.sqrt(n_pairs)
        
        r_centers.append(np.sqrt(r_lo * r_hi))  # Geometric mean
        correlations.append(corr)
        correlation_errs.append(corr_err)
        n_pairs_list.append(n_pairs)
    
    return (np.array(r_centers), np.array(correlations), 
            np.array(correlation_errs), np.array(n_pairs_list))


# ========== APPROACH TESTING ==========

def test_approach_1_random_walk(r_data, corr_data, corr_err):
    """
    Test Approach 1: Random Walk / Coherence Patches
    
    Prediction: K(R) = (ℓ₀/(ℓ₀+R))^0.5 = 1/√(1 + R/ℓ₀)
    
    Key test: Does n_coh = 0.5 fit better than other values?
    """
    print("\n" + "="*70)
    print("APPROACH 1: Random Walk / Coherence Patches")
    print("="*70)
    print("Prediction: K(R) = 1/√(1 + R/ℓ₀) → n_coh = 0.5 exactly")
    
    results = {'approach': 'random_walk_patches', 'tests': {}}
    
    # Fit with n_coh fixed to 0.5 (prediction)
    try:
        def K_fixed_05(r, A, ell0):
            return A * (ell0 / (ell0 + r)) ** 0.5
        
        popt_05, pcov_05 = curve_fit(K_fixed_05, r_data, corr_data, 
                                      p0=[0.3, 5.0], sigma=corr_err,
                                      bounds=([0, 0.5], [2, 20]))
        A_05, ell0_05 = popt_05
        chi2_05 = np.sum(((corr_data - K_fixed_05(r_data, *popt_05)) / corr_err)**2)
        dof_05 = len(r_data) - 2
        
        print(f"\n  n_coh = 0.5 (fixed):")
        print(f"    A = {A_05:.4f} ± {np.sqrt(pcov_05[0,0]):.4f}")
        print(f"    ℓ₀ = {ell0_05:.2f} ± {np.sqrt(pcov_05[1,1]):.2f} kpc")
        print(f"    χ²/dof = {chi2_05:.2f}/{dof_05} = {chi2_05/dof_05:.2f}")
        
        results['tests']['ncoh_0.5_fixed'] = {
            'A': float(A_05), 'ell0': float(ell0_05),
            'chi2': float(chi2_05), 'dof': int(dof_05)
        }
    except Exception as e:
        print(f"  ERROR fitting n_coh=0.5: {e}")
    
    # Fit with n_coh free
    try:
        popt_free, pcov_free = curve_fit(K_powerlaw_full, r_data, corr_data,
                                          p0=[0.3, 5.0, 0.5], sigma=corr_err,
                                          bounds=([0, 0.5, 0.1], [2, 20, 2.0]))
        A_free, ell0_free, n_free = popt_free
        chi2_free = np.sum(((corr_data - K_powerlaw_full(r_data, *popt_free)) / corr_err)**2)
        dof_free = len(r_data) - 3
        
        print(f"\n  n_coh free:")
        print(f"    A = {A_free:.4f} ± {np.sqrt(pcov_free[0,0]):.4f}")
        print(f"    ℓ₀ = {ell0_free:.2f} ± {np.sqrt(pcov_free[1,1]):.2f} kpc")
        print(f"    n_coh = {n_free:.3f} ± {np.sqrt(pcov_free[2,2]):.3f}")
        print(f"    χ²/dof = {chi2_free:.2f}/{dof_free} = {chi2_free/dof_free:.2f}")
        
        # Is n_coh = 0.5 within 2σ?
        n_err = np.sqrt(pcov_free[2,2])
        deviation = abs(n_free - 0.5) / n_err
        consistent = deviation < 2.0
        
        print(f"\n  ★ n_coh = 0.5 deviation: {deviation:.2f}σ → {'CONSISTENT' if consistent else 'INCONSISTENT'}")
        
        results['tests']['ncoh_free'] = {
            'A': float(A_free), 'ell0': float(ell0_free), 'n_coh': float(n_free),
            'n_coh_err': float(n_err), 'chi2': float(chi2_free), 'dof': int(dof_free),
            'deviation_sigma': float(deviation), 'consistent_with_0.5': consistent
        }
    except Exception as e:
        print(f"  ERROR fitting n_coh free: {e}")
    
    # Test 1/√N model directly
    print("\n  Direct test: C(r) = const / √(1 + r/ℓ₀)")
    
    return results


def test_approach_3_levy_vs_gaussian(r_data, corr_data, corr_err):
    """
    Test Approach 3: Lévy vs Gaussian Phase Statistics
    
    Gaussian → exponential decay (K = exp(-r/ℓ))
    Lévy (α=1, Cauchy) → power-law decay (K = (ℓ/(ℓ+r))^0.5)
    
    Key test: Power-law fits better than exponential
    """
    print("\n" + "="*70)
    print("APPROACH 3: Lévy vs Gaussian Phase Statistics")
    print("="*70)
    print("Gaussian prediction: K(R) = exp(-R/ℓ₀) [exponential]")
    print("Lévy (α=1) prediction: K(R) = (ℓ₀/R)^0.5 [power-law]")
    
    results = {'approach': 'levy_vs_gaussian', 'tests': {}}
    
    # Fit exponential (Gaussian prediction)
    try:
        popt_exp, pcov_exp = curve_fit(K_exponential_full, r_data, corr_data,
                                        p0=[0.5, 3.0, 1.0], sigma=corr_err,
                                        bounds=([0, 0.1, 0.1], [2, 20, 5]))
        A_exp, ell0_exp, n_exp = popt_exp
        chi2_exp = np.sum(((corr_data - K_exponential_full(r_data, *popt_exp)) / corr_err)**2)
        dof_exp = len(r_data) - 3
        
        print(f"\n  Exponential (Gaussian):")
        print(f"    A = {A_exp:.4f}, ℓ₀ = {ell0_exp:.2f} kpc")
        print(f"    χ²/dof = {chi2_exp:.2f}/{dof_exp} = {chi2_exp/dof_exp:.2f}")
        
        results['tests']['exponential'] = {
            'A': float(A_exp), 'ell0': float(ell0_exp),
            'chi2': float(chi2_exp), 'dof': int(dof_exp)
        }
    except Exception as e:
        print(f"  ERROR fitting exponential: {e}")
        chi2_exp = np.inf
    
    # Fit power-law (Lévy prediction)
    try:
        popt_pow, pcov_pow = curve_fit(K_powerlaw_full, r_data, corr_data,
                                        p0=[0.3, 5.0, 0.5], sigma=corr_err,
                                        bounds=([0, 0.5, 0.1], [2, 20, 2.0]))
        A_pow, ell0_pow, n_pow = popt_pow
        chi2_pow = np.sum(((corr_data - K_powerlaw_full(r_data, *popt_pow)) / corr_err)**2)
        dof_pow = len(r_data) - 3
        
        print(f"\n  Power-law (Lévy):")
        print(f"    A = {A_pow:.4f}, ℓ₀ = {ell0_pow:.2f} kpc, n = {n_pow:.3f}")
        print(f"    χ²/dof = {chi2_pow:.2f}/{dof_pow} = {chi2_pow/dof_pow:.2f}")
        
        results['tests']['powerlaw'] = {
            'A': float(A_pow), 'ell0': float(ell0_pow), 'n_coh': float(n_pow),
            'chi2': float(chi2_pow), 'dof': int(dof_pow)
        }
    except Exception as e:
        print(f"  ERROR fitting power-law: {e}")
        chi2_pow = np.inf
    
    # Compare models
    if chi2_exp != np.inf and chi2_pow != np.inf:
        delta_chi2 = chi2_exp - chi2_pow
        print(f"\n  ★ Δχ² (exp - power) = {delta_chi2:.2f}")
        
        if delta_chi2 > 0:
            print(f"    → POWER-LAW PREFERRED (Lévy/Cauchy over Gaussian)")
            winner = 'powerlaw'
        else:
            print(f"    → EXPONENTIAL PREFERRED (Gaussian over Lévy)")
            winner = 'exponential'
        
        results['comparison'] = {
            'delta_chi2': float(delta_chi2),
            'winner': winner,
            'supports_levy': delta_chi2 > 0
        }
    
    return results


def test_approach_4_anomalous_diffusion(r_data, corr_data, corr_err):
    """
    Test Approach 4: Anomalous Diffusion / Memory Effects
    
    Prediction: K(R) ~ R^(-(1-β)/2)
    For β = 0: K ~ R^(-0.5) → n_coh = 0.5
    
    Key test: Extract effective β from data
    """
    print("\n" + "="*70)
    print("APPROACH 4: Anomalous Diffusion / Memory Effects")
    print("="*70)
    print("Prediction: K(R) ~ R^(-(1-β)/2)")
    print("For β = 0 (no memory decay): n_coh = 0.5")
    
    results = {'approach': 'anomalous_diffusion', 'tests': {}}
    
    # From fitted n_coh, extract implied β
    try:
        popt, pcov = curve_fit(K_powerlaw_full, r_data, corr_data,
                               p0=[0.3, 5.0, 0.5], sigma=corr_err,
                               bounds=([0, 0.5, 0.1], [2, 20, 2.0]))
        n_fitted = popt[2]
        n_err = np.sqrt(pcov[2,2])
        
        # Invert: n_coh = (1-β)/2 → β = 1 - 2*n_coh
        beta_implied = 1 - 2 * n_fitted
        beta_err = 2 * n_err
        
        print(f"\n  Fitted n_coh = {n_fitted:.3f} ± {n_err:.3f}")
        print(f"  Implied memory exponent β = {beta_implied:.3f} ± {beta_err:.3f}")
        
        # Is β consistent with 0?
        deviation = abs(beta_implied) / beta_err
        consistent = deviation < 2.0
        
        print(f"\n  β = 0 deviation: {deviation:.2f}σ → {'CONSISTENT' if consistent else 'INCONSISTENT'}")
        
        if consistent:
            print("  → Supports MARGINALLY ANOMALOUS DIFFUSION (critical point)")
        else:
            if beta_implied > 0:
                print("  → Suggests SUBDIFFUSION (slower than normal)")
            else:
                print("  → Suggests SUPERDIFFUSION (faster than normal)")
        
        results['tests']['beta_inference'] = {
            'n_coh': float(n_fitted), 'n_coh_err': float(n_err),
            'beta': float(beta_implied), 'beta_err': float(beta_err),
            'deviation_sigma': float(deviation), 'consistent_with_0': consistent
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return results


def test_approach_5_effective_dimension(r_data, corr_data, corr_err):
    """
    Test Approach 5: Effective Dimension Analysis
    
    For random-walk coherence in d dimensions:
    K(R) ~ (ℓ₀/R)^(d/2 - 1)
    
    For n_coh = 0.5: d/2 - 1 = 0.5 → d = 3
    
    Key test: Does effective dimension ≈ 3 make sense for MW disk?
    """
    print("\n" + "="*70)
    print("APPROACH 5: Effective Dimensionality")
    print("="*70)
    print("For d dimensions: K(R) ~ (ℓ₀/R)^(d/2 - 1)")
    print("For n = 0.5: d_eff = 3")
    
    results = {'approach': 'effective_dimension', 'tests': {}}
    
    # From fitted n_coh, extract d_eff
    try:
        popt, pcov = curve_fit(K_powerlaw_full, r_data, corr_data,
                               p0=[0.3, 5.0, 0.5], sigma=corr_err,
                               bounds=([0, 0.5, 0.1], [2, 20, 2.0]))
        n_fitted = popt[2]
        n_err = np.sqrt(pcov[2,2])
        
        # Invert: n_coh = d/2 - 1 → d = 2*(n_coh + 1)
        d_eff = 2 * (n_fitted + 1)
        d_err = 2 * n_err
        
        print(f"\n  Fitted n_coh = {n_fitted:.3f} ± {n_err:.3f}")
        print(f"  Implied effective dimension d_eff = {d_eff:.2f} ± {d_err:.2f}")
        
        # Is d consistent with 3?
        deviation = abs(d_eff - 3) / d_err
        consistent = deviation < 2.0
        
        print(f"\n  d = 3 deviation: {deviation:.2f}σ → {'CONSISTENT' if consistent else 'INCONSISTENT'}")
        
        # Physical interpretation
        print(f"\n  Physical interpretation:")
        print(f"    d_eff ≈ 2: Purely 2D (thin disk limit)")
        print(f"    d_eff ≈ 3: Full 3D (thick or spherical)")
        print(f"    d_eff ≈ 2.5: Transitional (realistic disk)")
        
        if 2.5 < d_eff < 3.5:
            print(f"  → Measured d_eff = {d_eff:.2f}: REALISTIC for MW disk")
        
        results['tests']['dimension_inference'] = {
            'n_coh': float(n_fitted), 'n_coh_err': float(n_err),
            'd_eff': float(d_eff), 'd_err': float(d_err),
            'deviation_sigma': float(deviation), 'consistent_with_3': consistent
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return results


def test_approach_2_2d_geometry(r_data, corr_data, corr_err):
    """
    Test Approach 2: 2D Thin Disk Geometry
    
    This approach predicts n = 1 for pure 2D, but n = 0.5 emerges when
    accounting for amplitude vs potential.
    
    Key test: Is there evidence of 2D geometry effects?
    """
    print("\n" + "="*70)
    print("APPROACH 2: 2D Thin Disk Geometry")
    print("="*70)
    print("Pure 2D prediction: n = 1.0")
    print("With amplitude correction: n = 0.5")
    
    results = {'approach': '2d_thin_disk', 'tests': {}}
    
    # Fit with different n_coh values to test geometry
    n_values = [0.3, 0.5, 0.75, 1.0, 1.5]
    chi2_values = []
    
    for n_test in n_values:
        try:
            def K_test(r, A, ell0):
                return A * (ell0 / (ell0 + r)) ** n_test
            
            popt, pcov = curve_fit(K_test, r_data, corr_data,
                                   p0=[0.3, 5.0], sigma=corr_err,
                                   bounds=([0, 0.5], [2, 20]))
            chi2 = np.sum(((corr_data - K_test(r_data, *popt)) / corr_err)**2)
            chi2_values.append(chi2)
            print(f"  n = {n_test}: χ² = {chi2:.2f}")
        except:
            chi2_values.append(np.inf)
    
    best_idx = np.argmin(chi2_values)
    best_n = n_values[best_idx]
    
    print(f"\n  ★ Best-fit n from grid: {best_n}")
    
    if abs(best_n - 0.5) < 0.3:
        print("  → Supports AMPLITUDE-CORRECTED 2D model (n ≈ 0.5)")
    elif abs(best_n - 1.0) < 0.3:
        print("  → Supports PURE 2D model (n ≈ 1.0)")
    else:
        print(f"  → Neither 2D model strongly supported")
    
    results['tests']['n_grid_search'] = {
        'n_values': n_values,
        'chi2_values': [float(c) if c != np.inf else None for c in chi2_values],
        'best_n': float(best_n)
    }
    
    return results


# ========== VISUALIZATION ==========

def plot_all_approaches(r_data, corr_data, corr_err, results, output_path=None):
    """Generate comprehensive comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    r_theory = np.linspace(0.1, 12, 200)
    
    # Panel 1: Data vs n=0.5 prediction
    ax = axes[0, 0]
    ax.errorbar(r_data, corr_data, yerr=corr_err, fmt='ko', capsize=3, label='Gaia data')
    ax.plot(r_theory, 0.3 * K_powerlaw(r_theory, 5.0, 0.5), 'r-', lw=2, 
            label=r'$n_{coh}=0.5$ (prediction)')
    ax.plot(r_theory, 0.3 * K_powerlaw(r_theory, 5.0, 1.0), 'b--', lw=1.5, 
            label=r'$n_{coh}=1.0$')
    ax.set_xlabel('Separation [kpc]')
    ax.set_ylabel('Correlation')
    ax.set_title('Approach 1: Random Walk Patches')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Power-law vs exponential
    ax = axes[0, 1]
    ax.errorbar(r_data, corr_data, yerr=corr_err, fmt='ko', capsize=3, label='Data')
    ax.plot(r_theory, 0.3 * K_powerlaw(r_theory, 5.0, 0.5), 'r-', lw=2, 
            label='Power-law (Lévy)')
    ax.plot(r_theory, 0.3 * K_exponential(r_theory, 3.0, 1.0), 'b--', lw=2, 
            label='Exponential (Gaussian)')
    ax.set_xlabel('Separation [kpc]')
    ax.set_ylabel('Correlation')
    ax.set_title('Approach 3: Lévy vs Gaussian')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Log-log for power-law check
    ax = axes[0, 2]
    mask = corr_data > 0
    if np.sum(mask) > 3:
        ax.errorbar(r_data[mask], corr_data[mask], yerr=corr_err[mask], 
                    fmt='ko', capsize=3, label='Data')
        ax.plot(r_theory, 0.3 * K_powerlaw(r_theory, 5.0, 0.5), 'r-', lw=2,
                label=r'$\propto r^{-0.5}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('Separation [kpc]')
    ax.set_ylabel('Correlation')
    ax.set_title('Log-Log Scale (Power-law Test)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 4: χ² comparison
    ax = axes[1, 0]
    if 'levy_vs_gaussian' in results:
        tests = results['levy_vs_gaussian'].get('tests', {})
        chi2_exp = tests.get('exponential', {}).get('chi2', 0)
        chi2_pow = tests.get('powerlaw', {}).get('chi2', 0)
        
        bars = ax.bar(['Exponential\n(Gaussian)', 'Power-law\n(Lévy)'], 
                      [chi2_exp, chi2_pow], color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel(r'$\chi^2$ (lower is better)')
        ax.set_title('Model Comparison')
        
        for bar, val in zip(bars, [chi2_exp, chi2_pow]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 5: n_coh inference
    ax = axes[1, 1]
    if 'random_walk_patches' in results:
        tests = results['random_walk_patches'].get('tests', {})
        free_fit = tests.get('ncoh_free', {})
        if free_fit:
            n_val = free_fit.get('n_coh', 0.5)
            n_err = free_fit.get('n_coh_err', 0.1)
            
            ax.errorbar([1], [n_val], yerr=[n_err], fmt='ko', capsize=5, markersize=10)
            ax.axhline(0.5, color='r', linestyle='--', lw=2, label='Prediction: 0.5')
            ax.fill_between([0.5, 1.5], [0.5-0.05]*2, [0.5+0.05]*2, 
                           color='red', alpha=0.2, label='±0.05')
            ax.set_xlim(0.5, 1.5)
            ax.set_ylim(0, 1.5)
            ax.set_xticks([1])
            ax.set_xticklabels(['Fitted'])
            ax.set_ylabel(r'$n_{coh}$')
            ax.set_title(f'Fitted n_coh = {n_val:.3f} ± {n_err:.3f}')
            ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "DERIVATION VALIDATION SUMMARY\n" + "="*40 + "\n\n"
    
    # Check each approach
    if 'random_walk_patches' in results:
        tests = results['random_walk_patches'].get('tests', {})
        free_fit = tests.get('ncoh_free', {})
        if free_fit.get('consistent_with_0.5', False):
            summary_text += "✅ Approach 1 (Random Walk): SUPPORTED\n"
        else:
            summary_text += "❌ Approach 1 (Random Walk): NOT SUPPORTED\n"
    
    if 'levy_vs_gaussian' in results:
        comp = results['levy_vs_gaussian'].get('comparison', {})
        if comp.get('supports_levy', False):
            summary_text += "✅ Approach 3 (Lévy/Cauchy): SUPPORTED\n"
        else:
            summary_text += "❌ Approach 3 (Lévy/Cauchy): NOT SUPPORTED\n"
    
    if 'anomalous_diffusion' in results:
        tests = results['anomalous_diffusion'].get('tests', {})
        beta_test = tests.get('beta_inference', {})
        if beta_test.get('consistent_with_0', False):
            summary_text += "✅ Approach 4 (Anomalous Diff): SUPPORTED\n"
        else:
            summary_text += "❌ Approach 4 (Anomalous Diff): NOT SUPPORTED\n"
    
    if 'effective_dimension' in results:
        tests = results['effective_dimension'].get('tests', {})
        dim_test = tests.get('dimension_inference', {})
        if dim_test.get('consistent_with_3', False):
            summary_text += "✅ Approach 5 (Effective Dim): SUPPORTED\n"
        else:
            summary_text += "❌ Approach 5 (Effective Dim): NOT SUPPORTED\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: {output_path}")
    
    plt.close()


# ========== MAIN ==========

def main():
    print("="*70)
    print("n_coh = 0.5 DERIVATION VALIDATION TEST SUITE")
    print("="*70)
    print("\nThis tests whether the 5 theoretical approaches from")
    print("N_COH_DERIVATION.md are consistent with real Gaia data.\n")
    
    # Load data
    df = load_gaia_data()
    if df is None:
        print("\nERROR: Could not load Gaia data")
        return
    
    # Compute velocity correlations
    print("\nComputing velocity correlation function...")
    r_data, corr_data, corr_err, n_pairs = compute_velocity_correlations(df)
    
    if r_data is None or len(r_data) < 5:
        print("\nERROR: Insufficient correlation data")
        return
    
    print(f"\nCorrelation function computed:")
    print(f"  Bins: {len(r_data)}")
    print(f"  r range: {r_data.min():.2f} - {r_data.max():.2f} kpc")
    print(f"  Total pairs: {n_pairs.sum():,}")
    
    # Run all approach tests
    all_results = {}
    
    all_results['random_walk_patches'] = test_approach_1_random_walk(r_data, corr_data, corr_err)
    all_results['2d_thin_disk'] = test_approach_2_2d_geometry(r_data, corr_data, corr_err)
    all_results['levy_vs_gaussian'] = test_approach_3_levy_vs_gaussian(r_data, corr_data, corr_err)
    all_results['anomalous_diffusion'] = test_approach_4_anomalous_diffusion(r_data, corr_data, corr_err)
    all_results['effective_dimension'] = test_approach_5_effective_dimension(r_data, corr_data, corr_err)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Generate plot
    plot_path = OUTPUT_DIR / "ncoh_derivation_validation.png"
    plot_all_approaches(r_data, corr_data, corr_err, all_results, plot_path)
    
    # Save results
    results_path = OUTPUT_DIR / "ncoh_derivation_results.json"
    
    # Add data summary
    all_results['data_summary'] = {
        'n_stars': len(df),
        'n_bins': len(r_data),
        'r_range': [float(r_data.min()), float(r_data.max())],
        'total_pairs': int(n_pairs.sum()),
        'r_centers': r_data.tolist(),
        'correlations': corr_data.tolist(),
        'correlation_errors': corr_err.tolist()
    }
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)
    print(f"\nSaved results: {results_path}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
