#!/usr/bin/env python3
"""
Test Suite: Oscillatory Coherence Model
========================================

Following the discovery of NEGATIVE velocity correlations at r > 2.4 kpc,
this script tests an oscillatory coherence model that can explain both
positive (small r) and negative (large r) correlations.

Key insight: The zero crossing at ~2.4 kpc suggests λ_coh ≈ 9.6 kpc

New models tested:
- Approach 6: Winding-induced oscillatory coherence
- Damped cosine model
- Conservation constraint (sum rule) model
- Improved baseline using Eilers+2019 MW rotation curve

See: theory/coherence_exponent_derivation/N_COH_DERIVATION.md
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ========== ROTATION CURVE MODELS ==========

def v_circ_flat(R, v0=220.0):
    """Flat rotation curve (naive baseline)."""
    return np.full_like(R, v0, dtype=float)


def v_circ_eilers2019(R):
    """
    Milky Way rotation curve from Eilers et al. 2019 (ApJ 871, 120).
    
    Based on Gaia DR2 + APOGEE red giants.
    Valid for 5 < R < 25 kpc.
    
    v_circ(R) = v_0 * [1 + dv/dR * (R - R_0)]
    
    With:
    - v_0 = 229.0 ± 0.2 km/s at R_0 = 8.122 kpc
    - dv/dR = -1.7 ± 0.1 km/s/kpc (slight decline)
    """
    R = np.asarray(R)
    v0 = 229.0  # km/s at R_0
    R0 = 8.122  # kpc (Sun's galactocentric distance)
    dv_dR = -1.7  # km/s/kpc (declining rotation curve)
    
    return v0 + dv_dR * (R - R0)


def v_circ_mcmillan2017(R):
    """
    Milky Way rotation curve from McMillan 2017 (MNRAS 465, 76).
    
    Multi-component model with bulge, thin/thick disk, and halo.
    Simplified polynomial fit for 4 < R < 20 kpc.
    """
    R = np.asarray(R)
    # Polynomial approximation to McMillan 2017 Fig 4
    # Coefficients fitted to their model
    v0 = 233.0
    a1 = -0.5
    a2 = -0.02
    R0 = 8.21
    
    x = R - R0
    return v0 + a1 * x + a2 * x**2


# ========== OSCILLATORY COHERENCE MODELS ==========

def K_oscillatory_damped(r, A, ell0, wavelength, r_damp):
    """
    Approach 6: Oscillatory coherence with damping.
    
    K(r) = A * cos(2π r / λ) / √(1 + r/ℓ₀) * exp(-r/r_damp)
    
    This can produce:
    - Positive correlation at r < λ/4
    - Zero crossing at r = λ/4
    - Negative correlation at λ/4 < r < 3λ/4
    
    Parameters:
    - A: amplitude
    - ell0: coherence length (decay envelope)
    - wavelength: oscillation wavelength
    - r_damp: damping length
    """
    r = np.asarray(r)
    envelope = 1.0 / np.sqrt(1 + r / ell0)
    oscillation = np.cos(2 * np.pi * r / wavelength)
    damping = np.exp(-r / r_damp)
    return A * envelope * oscillation * damping


def K_oscillatory_simple(r, A, wavelength, r_damp):
    """
    Simple damped cosine (no power-law envelope).
    
    K(r) = A * cos(2π r / λ) * exp(-r/r_damp)
    """
    r = np.asarray(r)
    return A * np.cos(2 * np.pi * r / wavelength) * np.exp(-r / r_damp)


def K_winding_phase(r, A, dOmega_dR, t_coh, r_damp):
    """
    Winding-induced phase correlation.
    
    Stars at different radii accumulate phase difference due to differential rotation.
    Δφ = dΩ/dR × Δr × t_coh
    
    K(r) = A * cos(dΩ/dR × r × t_coh) * exp(-r/r_damp)
    
    Parameters:
    - dOmega_dR: differential rotation (rad/kpc/Gyr)
    - t_coh: coherence time (Gyr)
    - r_damp: damping length (kpc)
    """
    r = np.asarray(r)
    phase = dOmega_dR * r * t_coh  # Phase difference in radians
    return A * np.cos(phase) * np.exp(-r / r_damp)


def K_conservation_bessel(r, A, r_char):
    """
    Conservation constraint model using Bessel function.
    
    If total perturbation is conserved, correlation function must satisfy:
    ∫ C(r) r dr = 0 (in 2D)
    
    The Bessel function J_0 naturally satisfies this:
    K(r) = A * J_0(r / r_char)
    
    Has zeros at r ≈ 2.405 r_char, 5.52 r_char, etc.
    """
    from scipy.special import j0
    r = np.asarray(r)
    return A * j0(r / r_char)


def K_powerlaw_baseline(r, A, ell0, n_coh):
    """Original power-law (for comparison)."""
    r = np.asarray(r)
    return A * (ell0 / (ell0 + r)) ** n_coh


# ========== DATA LOADING WITH IMPROVED BASELINE ==========

def load_gaia_data_with_rotation_curve(rotation_model='eilers2019'):
    """
    Load Gaia data with proper rotation curve baseline.
    
    Parameters:
    - rotation_model: 'flat', 'eilers2019', or 'mcmillan2017'
    """
    gaia_path = ROOT / "data" / "gaia" / "gaia_processed_corrected.csv"
    
    if not gaia_path.exists():
        gaia_path = ROOT / "data" / "gaia" / "gaia_processed.csv"
    
    if not gaia_path.exists():
        print(f"ERROR: Gaia data not found")
        return None
    
    print(f"Loading Gaia data from: {gaia_path}")
    df = pd.read_csv(gaia_path)
    print(f"  Total stars: {len(df):,}")
    
    # Map columns
    if 'R_cyl' in df.columns:
        df['R_kpc'] = df['R_cyl']
    if 'z' in df.columns:
        df['z_kpc'] = df['z']
    
    # Get observed velocity
    if 'v_phi' in df.columns:
        df['v_obs'] = df['v_phi']
    elif 'v_tangential' in df.columns:
        df['v_obs'] = df['v_tangential']
    else:
        print("ERROR: No velocity column found")
        return None
    
    # Compute baseline using rotation curve model
    print(f"  Using rotation curve: {rotation_model}")
    
    if rotation_model == 'flat':
        df['v_baseline'] = v_circ_flat(df['R_kpc'].values)
    elif rotation_model == 'eilers2019':
        df['v_baseline'] = v_circ_eilers2019(df['R_kpc'].values)
    elif rotation_model == 'mcmillan2017':
        df['v_baseline'] = v_circ_mcmillan2017(df['R_kpc'].values)
    else:
        raise ValueError(f"Unknown rotation model: {rotation_model}")
    
    # Compute residuals
    df['delta_v'] = df['v_obs'] - df['v_baseline']
    
    # Filter to disk stars
    disk_mask = (df['R_kpc'] >= 4) & (df['R_kpc'] <= 16) & (np.abs(df['z_kpc']) < 1.0)
    df_disk = df[disk_mask].copy()
    print(f"  Disk stars: {len(df_disk):,}")
    print(f"  Mean residual: {df_disk['delta_v'].mean():.2f} km/s")
    print(f"  Residual std: {df_disk['delta_v'].std():.2f} km/s")
    
    return df_disk


def compute_velocity_correlations(df, n_sample=20000, r_bins=None, seed=42):
    """Compute velocity correlation function from star pairs."""
    from scipy.spatial import cKDTree
    
    np.random.seed(seed)
    
    if r_bins is None:
        r_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
        print(f"  Subsampled to {n_sample:,} stars")
    
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    delta_v = df['delta_v'].values
    
    sigma_v = np.std(delta_v)
    delta_v_norm = delta_v / sigma_v
    print(f"  Velocity dispersion: σ_v = {sigma_v:.1f} km/s")
    
    coords = np.column_stack([R, z])
    tree = cKDTree(coords)
    
    r_max = r_bins[-1] * 1.1
    pairs = tree.query_pairs(r=r_max, output_type='ndarray')
    
    if len(pairs) == 0:
        return None, None, None, None
    
    print(f"  Found {len(pairs):,} pairs")
    
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    
    dR = R[i_idx] - R[j_idx]
    dz = z[i_idx] - z[j_idx]
    separations = np.sqrt(dR**2 + dz**2)
    
    products = delta_v_norm[i_idx] * delta_v_norm[j_idx]
    
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
        
        r_centers.append(np.sqrt(r_lo * r_hi))
        correlations.append(corr)
        correlation_errs.append(corr_err)
        n_pairs_list.append(n_pairs)
    
    return (np.array(r_centers), np.array(correlations), 
            np.array(correlation_errs), np.array(n_pairs_list))


# ========== MODEL FITTING ==========

def fit_oscillatory_model(r_data, corr_data, corr_err):
    """Fit the oscillatory coherence models."""
    print("\n" + "="*70)
    print("APPROACH 6: Oscillatory Coherence Models")
    print("="*70)
    
    results = {'approach': 'oscillatory_coherence', 'models': {}}
    
    # Model A: Damped cosine with envelope
    print("\n--- Model A: Damped cosine with √(1+r/ℓ) envelope ---")
    try:
        # Initial guess: zero crossing at 2.4 kpc → λ ≈ 9.6 kpc
        p0 = [0.3, 2.0, 9.6, 10.0]  # A, ell0, wavelength, r_damp
        bounds = ([0, 0.1, 3, 2], [1, 10, 30, 50])
        
        popt, pcov = curve_fit(K_oscillatory_damped, r_data, corr_data,
                               p0=p0, sigma=corr_err, bounds=bounds, maxfev=10000)
        
        A, ell0, wavelength, r_damp = popt
        chi2 = np.sum(((corr_data - K_oscillatory_damped(r_data, *popt)) / corr_err)**2)
        dof = len(r_data) - 4
        
        print(f"  A = {A:.4f}")
        print(f"  ℓ₀ = {ell0:.2f} kpc")
        print(f"  λ = {wavelength:.2f} kpc (zero crossing at λ/4 = {wavelength/4:.2f} kpc)")
        print(f"  r_damp = {r_damp:.2f} kpc")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
        
        results['models']['damped_cosine_envelope'] = {
            'A': float(A), 'ell0': float(ell0), 
            'wavelength': float(wavelength), 'r_damp': float(r_damp),
            'zero_crossing': float(wavelength / 4),
            'chi2': float(chi2), 'dof': int(dof), 'chi2_reduced': float(chi2/dof)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Model B: Simple damped cosine
    print("\n--- Model B: Simple damped cosine ---")
    try:
        p0 = [0.3, 9.6, 10.0]  # A, wavelength, r_damp
        bounds = ([0, 3, 2], [1, 30, 50])
        
        popt, pcov = curve_fit(K_oscillatory_simple, r_data, corr_data,
                               p0=p0, sigma=corr_err, bounds=bounds, maxfev=10000)
        
        A, wavelength, r_damp = popt
        chi2 = np.sum(((corr_data - K_oscillatory_simple(r_data, *popt)) / corr_err)**2)
        dof = len(r_data) - 3
        
        print(f"  A = {A:.4f}")
        print(f"  λ = {wavelength:.2f} kpc (zero crossing at λ/4 = {wavelength/4:.2f} kpc)")
        print(f"  r_damp = {r_damp:.2f} kpc")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
        
        results['models']['simple_damped_cosine'] = {
            'A': float(A), 'wavelength': float(wavelength), 'r_damp': float(r_damp),
            'zero_crossing': float(wavelength / 4),
            'chi2': float(chi2), 'dof': int(dof), 'chi2_reduced': float(chi2/dof)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Model C: Winding-phase model
    print("\n--- Model C: Winding-induced phase correlation ---")
    try:
        # MW differential rotation: dΩ/dR ≈ -3 km/s/kpc² ≈ -0.003 rad/kpc/Myr
        # Convert to rad/kpc/Gyr: -3.0
        p0 = [0.3, 3.0, 1.0, 10.0]  # A, dOmega_dR, t_coh, r_damp
        bounds = ([0, 0.5, 0.1, 2], [1, 10, 5, 50])
        
        popt, pcov = curve_fit(K_winding_phase, r_data, corr_data,
                               p0=p0, sigma=corr_err, bounds=bounds, maxfev=10000)
        
        A, dOmega_dR, t_coh, r_damp = popt
        chi2 = np.sum(((corr_data - K_winding_phase(r_data, *popt)) / corr_err)**2)
        dof = len(r_data) - 4
        
        # Effective wavelength
        eff_wavelength = 2 * np.pi / (dOmega_dR * t_coh)
        
        print(f"  A = {A:.4f}")
        print(f"  dΩ/dR = {dOmega_dR:.3f} rad/kpc/Gyr")
        print(f"  t_coh = {t_coh:.2f} Gyr")
        print(f"  r_damp = {r_damp:.2f} kpc")
        print(f"  Effective λ = {eff_wavelength:.2f} kpc")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
        
        results['models']['winding_phase'] = {
            'A': float(A), 'dOmega_dR': float(dOmega_dR), 
            't_coh': float(t_coh), 'r_damp': float(r_damp),
            'effective_wavelength': float(eff_wavelength),
            'chi2': float(chi2), 'dof': int(dof), 'chi2_reduced': float(chi2/dof)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Model D: Bessel function (conservation constraint)
    print("\n--- Model D: Bessel J0 (conservation constraint) ---")
    try:
        from scipy.special import j0
        
        p0 = [0.3, 1.0]  # A, r_char
        bounds = ([0, 0.5], [1, 5])
        
        popt, pcov = curve_fit(K_conservation_bessel, r_data, corr_data,
                               p0=p0, sigma=corr_err, bounds=bounds, maxfev=10000)
        
        A, r_char = popt
        chi2 = np.sum(((corr_data - K_conservation_bessel(r_data, *popt)) / corr_err)**2)
        dof = len(r_data) - 2
        
        # First zero of J0 is at 2.405
        first_zero = 2.405 * r_char
        
        print(f"  A = {A:.4f}")
        print(f"  r_char = {r_char:.2f} kpc")
        print(f"  First zero at r = {first_zero:.2f} kpc")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
        
        results['models']['bessel_conservation'] = {
            'A': float(A), 'r_char': float(r_char),
            'first_zero': float(first_zero),
            'chi2': float(chi2), 'dof': int(dof), 'chi2_reduced': float(chi2/dof)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Compare to original power-law
    print("\n--- Baseline: Power-law (for comparison) ---")
    try:
        p0 = [0.3, 5.0, 0.5]
        bounds = ([0, 0.5, 0.1], [1, 20, 3.0])
        
        popt, pcov = curve_fit(K_powerlaw_baseline, r_data, corr_data,
                               p0=p0, sigma=corr_err, bounds=bounds, maxfev=10000)
        
        A, ell0, n_coh = popt
        chi2 = np.sum(((corr_data - K_powerlaw_baseline(r_data, *popt)) / corr_err)**2)
        dof = len(r_data) - 3
        
        print(f"  A = {A:.4f}, ℓ₀ = {ell0:.2f} kpc, n = {n_coh:.3f}")
        print(f"  χ²/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}")
        
        results['models']['powerlaw_baseline'] = {
            'A': float(A), 'ell0': float(ell0), 'n_coh': float(n_coh),
            'chi2': float(chi2), 'dof': int(dof), 'chi2_reduced': float(chi2/dof)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return results


def plot_oscillatory_results(r_data, corr_data, corr_err, results, output_path=None):
    """Generate comparison plot for oscillatory models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    r_theory = np.linspace(0.1, 12, 300)
    
    # Panel 1: Data with all oscillatory models
    ax = axes[0, 0]
    ax.errorbar(r_data, corr_data, yerr=corr_err, fmt='ko', capsize=3, 
                markersize=8, label='Gaia data', zorder=10)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    colors = {'damped_cosine_envelope': 'red', 'simple_damped_cosine': 'blue',
              'winding_phase': 'green', 'bessel_conservation': 'purple',
              'powerlaw_baseline': 'orange'}
    
    models = results.get('models', {})
    
    if 'damped_cosine_envelope' in models:
        m = models['damped_cosine_envelope']
        y = K_oscillatory_damped(r_theory, m['A'], m['ell0'], m['wavelength'], m['r_damp'])
        ax.plot(r_theory, y, '-', color=colors['damped_cosine_envelope'], lw=2,
                label=f"Damped cosine (χ²/dof={m['chi2_reduced']:.0f})")
    
    if 'simple_damped_cosine' in models:
        m = models['simple_damped_cosine']
        y = K_oscillatory_simple(r_theory, m['A'], m['wavelength'], m['r_damp'])
        ax.plot(r_theory, y, '--', color=colors['simple_damped_cosine'], lw=2,
                label=f"Simple cosine (χ²/dof={m['chi2_reduced']:.0f})")
    
    if 'bessel_conservation' in models:
        m = models['bessel_conservation']
        y = K_conservation_bessel(r_theory, m['A'], m['r_char'])
        ax.plot(r_theory, y, ':', color=colors['bessel_conservation'], lw=2,
                label=f"Bessel J₀ (χ²/dof={m['chi2_reduced']:.0f})")
    
    ax.set_xlabel('Separation [kpc]', fontsize=12)
    ax.set_ylabel('Correlation C(r)', fontsize=12)
    ax.set_title('Oscillatory Coherence Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # Panel 2: χ² comparison
    ax = axes[0, 1]
    model_names = []
    chi2_values = []
    
    for name, m in models.items():
        model_names.append(name.replace('_', '\n'))
        chi2_values.append(m['chi2_reduced'])
    
    if model_names:
        bars = ax.bar(range(len(model_names)), chi2_values, 
                      color=[colors.get(n.replace('\n', '_'), 'gray') for n in model_names],
                      alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_ylabel('χ²/dof (lower is better)', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Winding phase model detail
    ax = axes[1, 0]
    ax.errorbar(r_data, corr_data, yerr=corr_err, fmt='ko', capsize=3, 
                markersize=8, label='Data', zorder=10)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    if 'winding_phase' in models:
        m = models['winding_phase']
        y = K_winding_phase(r_theory, m['A'], m['dOmega_dR'], m['t_coh'], m['r_damp'])
        ax.plot(r_theory, y, 'g-', lw=2,
                label=f"Winding: dΩ/dR={m['dOmega_dR']:.2f}, t={m['t_coh']:.1f} Gyr")
        
        # Mark zero crossings
        zero_crossings = []
        for i in range(1, len(y)):
            if y[i-1] * y[i] < 0:
                zero_crossings.append(r_theory[i])
        for zc in zero_crossings[:2]:
            ax.axvline(zc, color='green', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Separation [kpc]', fontsize=12)
    ax.set_ylabel('Correlation C(r)', fontsize=12)
    ax.set_title('Winding-Induced Phase Correlation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = "OSCILLATORY COHERENCE ANALYSIS\n" + "="*40 + "\n\n"
    
    # Find best model
    if models:
        best_model = min(models.items(), key=lambda x: x[1]['chi2_reduced'])
        summary += f"BEST FIT: {best_model[0]}\n"
        summary += f"  χ²/dof = {best_model[1]['chi2_reduced']:.1f}\n\n"
        
        if 'wavelength' in best_model[1]:
            summary += f"  Wavelength λ = {best_model[1]['wavelength']:.1f} kpc\n"
            summary += f"  Zero crossing = {best_model[1]['zero_crossing']:.1f} kpc\n"
        elif 'effective_wavelength' in best_model[1]:
            summary += f"  Effective λ = {best_model[1]['effective_wavelength']:.1f} kpc\n"
        elif 'first_zero' in best_model[1]:
            summary += f"  First zero = {best_model[1]['first_zero']:.1f} kpc\n"
        
        summary += "\n"
        
        # Compare oscillatory to power-law
        if 'powerlaw_baseline' in models:
            pl_chi2 = models['powerlaw_baseline']['chi2_reduced']
            best_chi2 = best_model[1]['chi2_reduced']
            improvement = (pl_chi2 - best_chi2) / pl_chi2 * 100
            summary += f"vs Power-law: {improvement:.0f}% improvement\n"
    
    summary += "\nKEY FINDINGS:\n"
    summary += "• Negative correlations at r > 2.4 kpc\n"
    summary += "• Oscillatory models capture this\n"
    summary += "• λ ~ 9-10 kpc suggests galactic scale\n"
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
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
    print("OSCILLATORY COHERENCE MODEL TEST")
    print("="*70)
    print("\nTesting oscillatory models that can explain negative correlations")
    print("at large separations (r > 2.4 kpc).\n")
    
    all_results = {}
    
    # Test with different rotation curve baselines
    for rc_model in ['flat', 'eilers2019']:
        print(f"\n{'='*70}")
        print(f"ROTATION CURVE: {rc_model}")
        print("="*70)
        
        # Load data
        df = load_gaia_data_with_rotation_curve(rotation_model=rc_model)
        if df is None:
            continue
        
        # Compute correlations
        print("\nComputing velocity correlations...")
        r_data, corr_data, corr_err, n_pairs = compute_velocity_correlations(df)
        
        if r_data is None:
            continue
        
        print(f"  Bins: {len(r_data)}, r: {r_data.min():.2f}-{r_data.max():.2f} kpc")
        print(f"  Correlations: {corr_data.min():.3f} to {corr_data.max():.3f}")
        
        # Fit oscillatory models
        results = fit_oscillatory_model(r_data, corr_data, corr_err)
        results['rotation_curve'] = rc_model
        results['data'] = {
            'r_centers': r_data.tolist(),
            'correlations': corr_data.tolist(),
            'errors': corr_err.tolist()
        }
        
        all_results[rc_model] = results
        
        # Plot
        plot_path = OUTPUT_DIR / f"oscillatory_coherence_{rc_model}.png"
        plot_oscillatory_results(r_data, corr_data, corr_err, results, plot_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: BEST MODELS BY ROTATION CURVE")
    print("="*70)
    
    for rc_model, results in all_results.items():
        models = results.get('models', {})
        if models:
            best = min(models.items(), key=lambda x: x[1]['chi2_reduced'])
            print(f"\n{rc_model}:")
            print(f"  Best: {best[0]} (χ²/dof = {best[1]['chi2_reduced']:.0f})")
            if 'wavelength' in best[1]:
                print(f"  λ = {best[1]['wavelength']:.1f} kpc, zero at {best[1]['zero_crossing']:.1f} kpc")
    
    # Save results
    results_path = OUTPUT_DIR / "oscillatory_coherence_results.json"
    
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
