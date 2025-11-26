#!/usr/bin/env python3
"""
Extended Analysis: Conservation Integrals & Radius-Dependent Wavelength
========================================================================

Following the discovery of oscillatory coherence (λ ≈ 10 kpc), this script
performs deeper analysis:

1. Conservation integral test:
   - Does ∫ C(r) r dr ≈ 0? (1D conservation)
   - Does ∫ C(r) r² dr ≈ 0? (2D conservation)

2. Radius-dependent wavelength:
   - If λ comes from differential rotation, λ ∝ R²/t_coh
   - Inner disk should have λ_inner < λ_outer

3. First-principles λ prediction from MW parameters

4. Theoretical prediction vs observation comparison
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.integrate import simpson
import json
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ========== MODEL FUNCTIONS ==========

def K_oscillatory_simple(r, A, wavelength, r_damp):
    """Simple damped cosine."""
    r = np.asarray(r)
    return A * np.cos(2 * np.pi * r / wavelength) * np.exp(-r / r_damp)


def K_oscillatory_damped(r, A, ell0, wavelength, r_damp):
    """Damped cosine with √(1+r/ℓ) envelope."""
    r = np.asarray(r)
    envelope = 1.0 / np.sqrt(1 + r / ell0)
    oscillation = np.cos(2 * np.pi * r / wavelength)
    damping = np.exp(-r / r_damp)
    return A * envelope * oscillation * damping


# ========== CONSERVATION INTEGRALS ==========

def test_conservation_integrals(r_data, corr_data, corr_err=None):
    """
    Test if velocity perturbations satisfy conservation constraints.
    
    If total momentum/energy is conserved, correlation function must satisfy
    sum rules that force positive and negative correlations to balance.
    
    1D: ∫ C(r) r dr = 0  (momentum conservation in 1D)
    2D: ∫ C(r) r² dr = 0 (2D radial integral with area element)
    
    Also test modified integrals that account for finite survey volume.
    """
    print("\n" + "="*70)
    print("CONSERVATION INTEGRAL ANALYSIS")
    print("="*70)
    
    results = {}
    
    # Sort by radius
    idx = np.argsort(r_data)
    r = r_data[idx]
    C = corr_data[idx]
    
    # Compute various integrals
    # 1D integral: ∫ C(r) r dr
    integrand_1d = C * r
    integral_1d = simpson(integrand_1d, x=r)
    
    # 2D integral: ∫ C(r) r² dr (for 2D system with 2πr element)
    integrand_2d = C * r**2
    integral_2d = simpson(integrand_2d, x=r)
    
    # Weighted by 1/r (tests if inner matters more)
    integrand_inv = C  # Just C, since ∫ C(r) dr
    integral_0d = simpson(integrand_inv, x=r)
    
    # Cumulative integrals
    cumul_1d = np.array([simpson(integrand_1d[:i+1], x=r[:i+1]) for i in range(len(r))])
    cumul_2d = np.array([simpson(integrand_2d[:i+1], x=r[:i+1]) for i in range(len(r))])
    
    print(f"\n  Raw integrals (over data range {r.min():.2f} - {r.max():.2f} kpc):")
    print(f"    ∫ C(r) dr      = {integral_0d:.4f}")
    print(f"    ∫ C(r) r dr    = {integral_1d:.4f}  [1D conservation]")
    print(f"    ∫ C(r) r² dr   = {integral_2d:.4f}  [2D conservation]")
    
    # Positive vs negative contribution balance
    pos_contrib_1d = simpson(np.maximum(integrand_1d, 0), x=r)
    neg_contrib_1d = simpson(np.minimum(integrand_1d, 0), x=r)
    balance_1d = (pos_contrib_1d + neg_contrib_1d) / (pos_contrib_1d - neg_contrib_1d)
    
    pos_contrib_2d = simpson(np.maximum(integrand_2d, 0), x=r)
    neg_contrib_2d = simpson(np.minimum(integrand_2d, 0), x=r)
    balance_2d = (pos_contrib_2d + neg_contrib_2d) / (pos_contrib_2d - neg_contrib_2d)
    
    print(f"\n  Positive/negative balance:")
    print(f"    1D: pos = {pos_contrib_1d:.4f}, neg = {neg_contrib_1d:.4f}, ratio = {balance_1d:.3f}")
    print(f"    2D: pos = {pos_contrib_2d:.4f}, neg = {neg_contrib_2d:.4f}, ratio = {balance_2d:.3f}")
    
    # Interpretation
    print(f"\n  Interpretation:")
    if abs(integral_1d) < 0.1 * max(pos_contrib_1d, -neg_contrib_1d):
        print(f"    ✅ 1D integral near zero — suggests 1D conservation operating")
    elif integral_1d < 0:
        print(f"    ⚠️ 1D integral negative — negative correlations dominate")
    else:
        print(f"    ⚠️ 1D integral positive — positive correlations dominate")
    
    results['integral_0d'] = float(integral_0d)
    results['integral_1d'] = float(integral_1d)
    results['integral_2d'] = float(integral_2d)
    results['balance_1d'] = float(balance_1d)
    results['balance_2d'] = float(balance_2d)
    results['cumulative_1d'] = cumul_1d.tolist()
    results['cumulative_2d'] = cumul_2d.tolist()
    
    return results, r, cumul_1d, cumul_2d


# ========== RADIUS-DEPENDENT WAVELENGTH ==========

def load_gaia_by_radius(inner_cut=8.0, rotation_model='flat'):
    """Load Gaia data split into inner and outer samples."""
    gaia_path = ROOT / "data" / "gaia" / "gaia_processed_corrected.csv"
    
    if not gaia_path.exists():
        gaia_path = ROOT / "data" / "gaia" / "gaia_processed.csv"
    
    if not gaia_path.exists():
        return None, None
    
    print(f"Loading Gaia data...")
    df = pd.read_csv(gaia_path)
    
    # Map columns
    if 'R_cyl' in df.columns:
        df['R_kpc'] = df['R_cyl']
    if 'z' in df.columns:
        df['z_kpc'] = df['z']
    if 'v_phi' in df.columns:
        df['v_obs'] = df['v_phi']
    elif 'v_tangential' in df.columns:
        df['v_obs'] = df['v_tangential']
    
    # Baseline
    v_flat = 220.0
    df['delta_v'] = df['v_obs'] - v_flat
    
    # Filter to disk
    disk_mask = (df['R_kpc'] >= 4) & (df['R_kpc'] <= 16) & (np.abs(df['z_kpc']) < 1.0)
    df_disk = df[disk_mask].copy()
    
    # Split by radius
    df_inner = df_disk[df_disk['R_kpc'] < inner_cut].copy()
    df_outer = df_disk[df_disk['R_kpc'] >= inner_cut].copy()
    
    print(f"  Inner (R < {inner_cut} kpc): {len(df_inner):,} stars")
    print(f"  Outer (R >= {inner_cut} kpc): {len(df_outer):,} stars")
    
    return df_inner, df_outer


def compute_correlations_subsample(df, n_sample=15000, r_bins=None, seed=42):
    """Compute correlation function for a subsample."""
    np.random.seed(seed)
    
    if r_bins is None:
        r_bins = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0])
    
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
    
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    delta_v = df['delta_v'].values
    
    sigma_v = np.std(delta_v)
    delta_v_norm = delta_v / sigma_v
    
    coords = np.column_stack([R, z])
    tree = cKDTree(coords)
    
    r_max = r_bins[-1] * 1.1
    pairs = tree.query_pairs(r=r_max, output_type='ndarray')
    
    if len(pairs) == 0:
        return None, None, None
    
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    
    dR = R[i_idx] - R[j_idx]
    dz = z[i_idx] - z[j_idx]
    separations = np.sqrt(dR**2 + dz**2)
    products = delta_v_norm[i_idx] * delta_v_norm[j_idx]
    
    r_centers = []
    correlations = []
    correlation_errs = []
    
    for i in range(len(r_bins) - 1):
        r_lo, r_hi = r_bins[i], r_bins[i + 1]
        mask = (separations >= r_lo) & (separations < r_hi)
        
        n_pairs = np.sum(mask)
        if n_pairs < 50:
            continue
        
        corr = np.mean(products[mask])
        corr_err = np.std(products[mask]) / np.sqrt(n_pairs)
        
        r_centers.append(np.sqrt(r_lo * r_hi))
        correlations.append(corr)
        correlation_errs.append(corr_err)
    
    return np.array(r_centers), np.array(correlations), np.array(correlation_errs)


def fit_wavelength(r_data, corr_data, corr_err):
    """Fit oscillatory model and extract wavelength."""
    try:
        p0 = [0.3, 9.0, 20.0]
        bounds = ([0, 3, 2], [1, 25, 100])
        
        popt, pcov = curve_fit(K_oscillatory_simple, r_data, corr_data,
                               p0=p0, sigma=corr_err, bounds=bounds, maxfev=5000)
        
        A, wavelength, r_damp = popt
        wavelength_err = np.sqrt(pcov[1, 1])
        chi2 = np.sum(((corr_data - K_oscillatory_simple(r_data, *popt)) / corr_err)**2)
        dof = len(r_data) - 3
        
        return {
            'wavelength': float(wavelength),
            'wavelength_err': float(wavelength_err),
            'A': float(A),
            'r_damp': float(r_damp),
            'chi2': float(chi2),
            'dof': int(dof),
            'chi2_reduced': float(chi2 / dof) if dof > 0 else np.inf
        }
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None


def test_radius_dependent_wavelength():
    """
    Test if wavelength varies with galactocentric radius.
    
    Prediction: If λ comes from differential rotation,
    λ(R) ∝ R²/(v_c × t_coh)
    
    So λ_outer / λ_inner ≈ (R_outer / R_inner)²
    """
    print("\n" + "="*70)
    print("RADIUS-DEPENDENT WAVELENGTH ANALYSIS")
    print("="*70)
    print("\nPrediction: λ ∝ R² (from differential rotation)")
    print("  If true: λ_outer > λ_inner")
    
    results = {}
    
    # Load data split by radius
    df_inner, df_outer = load_gaia_by_radius(inner_cut=8.0)
    
    if df_inner is None:
        print("ERROR: Could not load data")
        return results
    
    # Mean radii
    R_inner_mean = df_inner['R_kpc'].mean()
    R_outer_mean = df_outer['R_kpc'].mean()
    
    print(f"\n  Mean radius - inner: {R_inner_mean:.2f} kpc")
    print(f"  Mean radius - outer: {R_outer_mean:.2f} kpc")
    
    # Theoretical prediction for λ ratio
    lambda_ratio_pred = (R_outer_mean / R_inner_mean)**2
    print(f"\n  Predicted λ_outer/λ_inner = ({R_outer_mean:.2f}/{R_inner_mean:.2f})² = {lambda_ratio_pred:.2f}")
    
    # Compute correlations for each sample
    print("\n  Computing correlations (inner sample)...")
    r_inner, corr_inner, err_inner = compute_correlations_subsample(df_inner, n_sample=12000)
    
    print("  Computing correlations (outer sample)...")
    r_outer, corr_outer, err_outer = compute_correlations_subsample(df_outer, n_sample=12000)
    
    if r_inner is None or r_outer is None:
        print("ERROR: Correlation computation failed")
        return results
    
    # Fit wavelengths
    print("\n  Fitting inner sample:")
    fit_inner = fit_wavelength(r_inner, corr_inner, err_inner)
    if fit_inner:
        print(f"    λ_inner = {fit_inner['wavelength']:.2f} ± {fit_inner['wavelength_err']:.2f} kpc")
        print(f"    χ²/dof = {fit_inner['chi2_reduced']:.0f}")
    
    print("\n  Fitting outer sample:")
    fit_outer = fit_wavelength(r_outer, corr_outer, err_outer)
    if fit_outer:
        print(f"    λ_outer = {fit_outer['wavelength']:.2f} ± {fit_outer['wavelength_err']:.2f} kpc")
        print(f"    χ²/dof = {fit_outer['chi2_reduced']:.0f}")
    
    # Compare
    if fit_inner and fit_outer:
        lambda_ratio_obs = fit_outer['wavelength'] / fit_inner['wavelength']
        
        print(f"\n  Observed λ_outer/λ_inner = {lambda_ratio_obs:.2f}")
        print(f"  Predicted from R² scaling = {lambda_ratio_pred:.2f}")
        
        ratio_error = abs(lambda_ratio_obs - lambda_ratio_pred) / lambda_ratio_pred * 100
        print(f"  Discrepancy: {ratio_error:.0f}%")
        
        if abs(lambda_ratio_obs - lambda_ratio_pred) < 0.5:
            print(f"\n  ✅ CONSISTENT with differential rotation origin")
        elif lambda_ratio_obs > 1.2:
            print(f"\n  ✅ λ_outer > λ_inner as predicted")
        else:
            print(f"\n  ⚠️ Ratio does not match R² prediction")
        
        results['R_inner_mean'] = float(R_inner_mean)
        results['R_outer_mean'] = float(R_outer_mean)
        results['lambda_inner'] = fit_inner
        results['lambda_outer'] = fit_outer
        results['lambda_ratio_observed'] = float(lambda_ratio_obs)
        results['lambda_ratio_predicted'] = float(lambda_ratio_pred)
        
        # Store data for plotting
        results['data_inner'] = {
            'r': r_inner.tolist(),
            'corr': corr_inner.tolist(),
            'err': err_inner.tolist()
        }
        results['data_outer'] = {
            'r': r_outer.tolist(),
            'corr': corr_outer.tolist(),
            'err': err_outer.tolist()
        }
    
    return results


# ========== FIRST-PRINCIPLES λ PREDICTION ==========

def predict_lambda_from_MW():
    """
    Derive λ from first principles using Milky Way rotation curve.
    
    For flat rotation: v_c = const
    Ω(R) = v_c / R
    dΩ/dR = -v_c / R²
    
    Phase accumulation: Δφ = |dΩ/dR| × Δr × t_coh
    
    Coherence oscillates with wavelength:
    λ = 2π / (|dΩ/dR| × t_coh)
    
    Substituting:
    λ = 2π R² / (v_c × t_coh)
    """
    print("\n" + "="*70)
    print("FIRST-PRINCIPLES λ PREDICTION")
    print("="*70)
    
    # MW parameters
    R0 = 8.122  # kpc (solar galactocentric radius)
    v_c = 220.0  # km/s (circular velocity)
    
    # Convert v_c to kpc/Gyr: 1 km/s = 1.022 kpc/Gyr
    v_c_kpc_Gyr = v_c * 1.022
    
    print(f"\n  Milky Way parameters:")
    print(f"    R₀ = {R0:.2f} kpc")
    print(f"    v_c = {v_c} km/s = {v_c_kpc_Gyr:.1f} kpc/Gyr")
    
    # Differential rotation at solar circle
    dOmega_dR = -v_c_kpc_Gyr / R0**2  # rad/kpc/Gyr
    print(f"\n  Differential rotation at R₀:")
    print(f"    |dΩ/dR| = {abs(dOmega_dR):.3f} rad/kpc/Gyr")
    
    # Orbital period at solar circle
    T_orbit = 2 * np.pi * R0 / v_c_kpc_Gyr  # Gyr
    print(f"    T_orbit = {T_orbit:.3f} Gyr = {T_orbit*1000:.0f} Myr")
    
    # Prediction: λ as function of t_coh
    print(f"\n  λ prediction: λ = 2π/(|dΩ/dR| × t_coh)")
    
    t_coh_values = [0.2, 0.5, 1.0, 2.0, 3.0]  # Gyr
    
    print(f"\n  t_coh [Gyr] | λ_pred [kpc] | # orbits")
    print(f"  " + "-"*40)
    
    predictions = {}
    for t_coh in t_coh_values:
        lambda_pred = 2 * np.pi / (abs(dOmega_dR) * t_coh)
        n_orbits = t_coh / T_orbit
        print(f"    {t_coh:.1f}        | {lambda_pred:.1f}          | {n_orbits:.1f}")
        predictions[f't_coh_{t_coh}'] = {
            't_coh_Gyr': t_coh,
            'lambda_kpc': float(lambda_pred),
            'n_orbits': float(n_orbits)
        }
    
    # Invert: given observed λ ≈ 10 kpc, what t_coh?
    lambda_obs = 10.2  # kpc from our fit
    t_coh_inferred = 2 * np.pi / (abs(dOmega_dR) * lambda_obs)
    n_orbits_inferred = t_coh_inferred / T_orbit
    
    print(f"\n  ★ Inverting observed λ = {lambda_obs:.1f} kpc:")
    print(f"    t_coh = {t_coh_inferred:.2f} Gyr = {t_coh_inferred*1000:.0f} Myr")
    print(f"    = {n_orbits_inferred:.2f} orbital periods")
    
    # This should connect to the winding gate!
    print(f"\n  Comparison to winding gate theory:")
    print(f"    Winding N_crit = v_c/σ_v ≈ 220/30 ≈ 7 orbits")
    print(f"    Inferred from λ: {n_orbits_inferred:.1f} orbits")
    
    if n_orbits_inferred < 3:
        print(f"    ⚠️ Shorter than expected — suggests additional decoherence")
    elif n_orbits_inferred > 5:
        print(f"    ✅ In range of winding gate prediction")
    
    # Epicyclic correction
    kappa_over_Omega = np.sqrt(2)  # For flat rotation
    f_epicyclic = 1 / kappa_over_Omega
    
    print(f"\n  Epicyclic correction factor:")
    print(f"    κ/Ω = √2 for flat rotation")
    print(f"    f_epicyclic = Ω/κ = {f_epicyclic:.3f}")
    print(f"    Corrected t_coh = {t_coh_inferred / f_epicyclic:.2f} Gyr")
    
    return {
        'R0_kpc': R0,
        'v_c_kms': v_c,
        'dOmega_dR': float(abs(dOmega_dR)),
        'T_orbit_Gyr': float(T_orbit),
        'lambda_observed': lambda_obs,
        't_coh_inferred': float(t_coh_inferred),
        'n_orbits_inferred': float(n_orbits_inferred),
        'f_epicyclic': float(f_epicyclic),
        'predictions': predictions
    }


# ========== VISUALIZATION ==========

def plot_extended_analysis(cons_results, r_cons, cumul_1d, cumul_2d,
                           radius_results, theory_results, output_path=None):
    """Generate comprehensive extended analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Conservation integrals
    ax = axes[0, 0]
    ax.plot(r_cons, cumul_1d, 'b-', lw=2, label='∫ C(r) r dr')
    ax.plot(r_cons, cumul_2d / 10, 'r--', lw=2, label='∫ C(r) r² dr / 10')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Upper integration limit [kpc]')
    ax.set_ylabel('Cumulative integral')
    ax.set_title('Conservation Integrals', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Inner vs outer correlations
    ax = axes[0, 1]
    if radius_results and 'data_inner' in radius_results:
        d_in = radius_results['data_inner']
        d_out = radius_results['data_outer']
        
        ax.errorbar(d_in['r'], d_in['corr'], yerr=d_in['err'], 
                    fmt='bo-', capsize=3, label=f"Inner (R < 8 kpc)")
        ax.errorbar(d_out['r'], d_out['corr'], yerr=d_out['err'],
                    fmt='rs-', capsize=3, label=f"Outer (R ≥ 8 kpc)")
        
        # Add fits
        r_theory = np.linspace(0.1, 7, 100)
        if radius_results.get('lambda_inner'):
            fit_in = radius_results['lambda_inner']
            y_in = K_oscillatory_simple(r_theory, fit_in['A'], fit_in['wavelength'], fit_in['r_damp'])
            ax.plot(r_theory, y_in, 'b--', alpha=0.5, lw=1.5)
        if radius_results.get('lambda_outer'):
            fit_out = radius_results['lambda_outer']
            y_out = K_oscillatory_simple(r_theory, fit_out['A'], fit_out['wavelength'], fit_out['r_damp'])
            ax.plot(r_theory, y_out, 'r--', alpha=0.5, lw=1.5)
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Separation [kpc]')
    ax.set_ylabel('Correlation')
    ax.set_title('Inner vs Outer Disk', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: λ prediction vs t_coh
    ax = axes[1, 0]
    if theory_results and 'predictions' in theory_results:
        t_values = []
        lambda_values = []
        for key, pred in theory_results['predictions'].items():
            t_values.append(pred['t_coh_Gyr'])
            lambda_values.append(pred['lambda_kpc'])
        
        ax.plot(t_values, lambda_values, 'ko-', lw=2, markersize=8, label='Theory: λ = 2π/(|dΩ/dR|×t)')
        
        # Mark observed λ
        lambda_obs = theory_results.get('lambda_observed', 10.2)
        t_inferred = theory_results.get('t_coh_inferred', 0.6)
        ax.axhline(lambda_obs, color='red', linestyle='--', lw=2, label=f'Observed λ = {lambda_obs:.1f} kpc')
        ax.axvline(t_inferred, color='green', linestyle=':', lw=2, label=f'Inferred t_coh = {t_inferred:.2f} Gyr')
        ax.plot(t_inferred, lambda_obs, 'r*', markersize=15, zorder=10)
    
    ax.set_xlabel('Coherence time t_coh [Gyr]')
    ax.set_ylabel('Wavelength λ [kpc]')
    ax.set_title('First-Principles λ Prediction', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = "EXTENDED ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
    
    # Conservation
    summary += "CONSERVATION INTEGRALS:\n"
    if cons_results:
        i1d = cons_results.get('integral_1d', 0)
        summary += f"  ∫ C(r) r dr = {i1d:.3f}\n"
        if abs(i1d) < 0.1:
            summary += "  → Near conservation\n"
        else:
            summary += "  → Imbalanced\n"
    summary += "\n"
    
    # Radius dependence
    summary += "RADIUS-DEPENDENT λ:\n"
    if radius_results:
        if radius_results.get('lambda_inner') and radius_results.get('lambda_outer'):
            l_in = radius_results['lambda_inner']['wavelength']
            l_out = radius_results['lambda_outer']['wavelength']
            ratio_obs = radius_results.get('lambda_ratio_observed', 0)
            ratio_pred = radius_results.get('lambda_ratio_predicted', 0)
            summary += f"  λ_inner = {l_in:.1f} kpc\n"
            summary += f"  λ_outer = {l_out:.1f} kpc\n"
            summary += f"  Ratio obs/pred = {ratio_obs:.2f}/{ratio_pred:.2f}\n"
    summary += "\n"
    
    # Theory
    summary += "FIRST PRINCIPLES:\n"
    if theory_results:
        t_inf = theory_results.get('t_coh_inferred', 0)
        n_orb = theory_results.get('n_orbits_inferred', 0)
        summary += f"  t_coh = {t_inf:.2f} Gyr\n"
        summary += f"  = {n_orb:.1f} orbits\n"
    
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
    print("EXTENDED COHERENCE ANALYSIS")
    print("="*70)
    
    all_results = {}
    
    # Load previous correlation data
    prev_results_path = OUTPUT_DIR / "oscillatory_coherence_results.json"
    if prev_results_path.exists():
        with open(prev_results_path) as f:
            prev_results = json.load(f)
        
        if 'flat' in prev_results and 'data' in prev_results['flat']:
            r_data = np.array(prev_results['flat']['data']['r_centers'])
            corr_data = np.array(prev_results['flat']['data']['correlations'])
            corr_err = np.array(prev_results['flat']['data']['errors'])
            
            # 1. Conservation integrals
            cons_results, r_cons, cumul_1d, cumul_2d = test_conservation_integrals(
                r_data, corr_data, corr_err)
            all_results['conservation'] = cons_results
    else:
        print("WARNING: Previous results not found, skipping conservation analysis")
        cons_results = {}
        r_cons = np.array([])
        cumul_1d = np.array([])
        cumul_2d = np.array([])
    
    # 2. Radius-dependent wavelength
    radius_results = test_radius_dependent_wavelength()
    all_results['radius_dependent'] = radius_results
    
    # 3. First-principles prediction
    theory_results = predict_lambda_from_MW()
    all_results['theory'] = theory_results
    
    # Plot
    plot_path = OUTPUT_DIR / "extended_analysis.png"
    plot_extended_analysis(cons_results, r_cons, cumul_1d, cumul_2d,
                           radius_results, theory_results, plot_path)
    
    # Save results
    results_path = OUTPUT_DIR / "extended_analysis_results.json"
    
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
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
