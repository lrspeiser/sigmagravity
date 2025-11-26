#!/usr/bin/env python3
"""
Gaia DR3 Velocity Correlation Test for Σ-Gravity
=================================================

This script tests the Σ-Gravity prediction that velocity residuals should be
spatially correlated according to the coherence kernel:

    ⟨δv(R) δv(R')⟩ ∝ K_coh(|R-R'|; ℓ₀, n_coh)
    
where K_coh(r) = (ℓ₀/(ℓ₀+r))^n_coh with ℓ₀ ≈ 5 kpc, n_coh ≈ 0.5

Null hypothesis (ΛCDM): Velocity residuals are uncorrelated beyond ~100 pc
                        (scale of dark matter substructure)

Σ-Gravity prediction:   Power-law correlations extending to ~5 kpc

This is a critical falsifiable test mentioned in the paper §2.8:
"Velocity correlations in Gaia DR3 should match the power-law coherence form
(ℓ₀/(ℓ₀ + |R-R'|))^n_coh with ℓ₀ = 5 kpc"

See README.md for reference.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ========== COHERENCE KERNEL ==========

def K_coh(r, ell0=5.0, n_coh=0.5):
    """
    Σ-Gravity coherence kernel.
    
    Parameters
    ----------
    r : array-like
        Separation in kpc
    ell0 : float
        Coherence length scale (default 5 kpc for galaxies)
    n_coh : float
        Coherence exponent (default 0.5)
    
    Returns
    -------
    K : array-like
        Coherence kernel value
    """
    r = np.asarray(r)
    return (ell0 / (ell0 + r)) ** n_coh


def K_coh_parametric(r, A, ell0, n_coh):
    """
    Parametric coherence kernel for fitting.
    
    C(r) = A * (ℓ₀/(ℓ₀+r))^n_coh
    """
    return A * (ell0 / (ell0 + r)) ** n_coh


def K_null(r, sigma=0.0):
    """
    ΛCDM null hypothesis: No correlations beyond local neighborhood.
    Returns constant (random noise floor).
    """
    return np.full_like(r, sigma, dtype=float)


# ========== DATA LOADING ==========

def load_predicted_gaia_data(filepath=None):
    """
    Load Gaia data with velocity predictions.
    
    Returns DataFrame with:
    - R_kpc, z_kpc: Galactic coordinates
    - v_obs_kms: Observed velocity
    - v_baryon_kms: Baryonic (Newtonian) prediction
    - v_model_kms: Σ-Gravity prediction
    - delta_v_baryon: Residual from baryonic prediction (what we test)
    """
    if filepath is None:
        filepath = ROOT / "data" / "gaia" / "outputs" / "mw_gaia_full_coverage_predicted.csv"
    
    filepath = Path(filepath)
    if not filepath.exists():
        # Try alternative locations
        alternatives = [
            ROOT / "data" / "gaia" / "outputs" / "mw_gaia_144k_plus_ac_predicted.csv",
            ROOT / "data" / "gaia" / "outputs" / "mw_gaia_144k_predicted.csv",
        ]
        for alt in alternatives:
            if alt.exists():
                filepath = alt
                break
        else:
            raise FileNotFoundError(f"No predicted Gaia data found. Tried: {filepath}, {alternatives}")
    
    print(f"Loading Gaia data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Calculate velocity residuals from baryonic prediction
    df['delta_v_baryon'] = df['v_obs_kms'] - df['v_baryon_kms']
    
    # Also calculate residuals from Σ-Gravity prediction for comparison
    df['delta_v_sigma'] = df['v_obs_kms'] - df['v_model_kms']
    
    # Filter to high-quality stars (small velocity errors)
    print(f"  Total stars: {len(df)}")
    
    # Radial selection for disk stars
    disk_mask = (df['R_kpc'] >= 4) & (df['R_kpc'] <= 16) & (np.abs(df['z_kpc']) < 1.0)
    df_disk = df[disk_mask].copy()
    print(f"  Disk stars (4 < R < 16 kpc, |z| < 1 kpc): {len(df_disk)}")
    
    return df_disk


def load_raw_gaia_data(filepath=None):
    """
    Load raw Gaia data (observed velocities only, no predictions).
    
    For use when predictions need to be computed.
    """
    if filepath is None:
        filepath = ROOT / "data" / "gaia" / "mw" / "gaia_mw_real.csv"
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Gaia data not found: {filepath}")
    
    print(f"Loading raw Gaia data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Total stars: {len(df)}")
    
    return df


def load_large_gaia_data(filepath=None, v_flat=220.0):
    """
    Load the large 1.8M star Gaia dataset.
    
    Since this doesn't have baryonic predictions pre-computed, we use
    a simple model: v_expected = v_flat (flat rotation curve approximation).
    
    The velocity residuals then become:
    delta_v = v_phi - v_flat
    
    For the correlation test, this is a reasonable first approximation
    since we're looking for *correlations*, not absolute values.
    
    Parameters
    ----------
    filepath : Path or None
        Path to gaia_processed_corrected.csv
    v_flat : float
        Assumed flat rotation curve velocity (km/s)
    
    Returns
    -------
    df : DataFrame
        With R_kpc, z_kpc, v_obs_kms, v_baryon_kms, delta_v_baryon
    """
    if filepath is None:
        filepath = ROOT / "data" / "gaia" / "gaia_processed_corrected.csv"
    
    filepath = Path(filepath)
    if not filepath.exists():
        # Try alternatives
        alternatives = [
            ROOT / "data" / "gaia" / "gaia_processed.csv",
            ROOT / "data" / "gaia" / "gaia_large_sample_raw.csv",
        ]
        for alt in alternatives:
            if alt.exists():
                filepath = alt
                break
        else:
            raise FileNotFoundError(f"Large Gaia data not found: {filepath}")
    
    print(f"Loading large Gaia dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Total stars: {len(df):,}")
    
    # Map columns to expected format
    df_out = pd.DataFrame()
    df_out['R_kpc'] = df['R_cyl'] if 'R_cyl' in df.columns else df['R_kpc']
    df_out['z_kpc'] = df['z'] if 'z' in df.columns else df['z_kpc']
    df_out['v_obs_kms'] = df['v_phi'] if 'v_phi' in df.columns else df['v_obs_kms']
    
    # Simple flat rotation curve model for v_baryon
    # This is an approximation - the real MW rotation curve is ~flat at 220 km/s
    df_out['v_baryon_kms'] = v_flat
    
    # Compute residual
    df_out['delta_v_baryon'] = df_out['v_obs_kms'] - df_out['v_baryon_kms']
    
    print(f"  Using flat rotation curve model: v_baryon = {v_flat} km/s")
    
    # Filter to disk stars
    disk_mask = (df_out['R_kpc'] >= 4) & (df_out['R_kpc'] <= 16) & (np.abs(df_out['z_kpc']) < 1.0)
    df_disk = df_out[disk_mask].copy()
    print(f"  Disk stars (4 < R < 16 kpc, |z| < 1 kpc): {len(df_disk):,}")
    
    return df_disk


# ========== CORRELATION ANALYSIS ==========

def compute_pair_separations(R, z, max_pairs=500000, seed=42):
    """
    Compute 3D separations between star pairs.
    
    For large samples, subsample pairs randomly.
    
    Parameters
    ----------
    R : array-like
        Galactocentric radii (kpc)
    z : array-like  
        Heights above disk (kpc)
    max_pairs : int
        Maximum number of pairs to compute (for memory)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    separations : array
        3D separations in kpc
    i_indices, j_indices : arrays
        Indices of star pairs
    """
    n = len(R)
    total_pairs = n * (n - 1) // 2
    print(f"  Total possible pairs: {total_pairs:,}")
    
    if total_pairs <= max_pairs:
        # Compute all pairs
        i_indices, j_indices = np.triu_indices(n, k=1)
    else:
        # Subsample pairs
        np.random.seed(seed)
        n_pairs = max_pairs
        print(f"  Subsampling to {n_pairs:,} pairs")
        
        # Random pairs
        i_indices = np.random.randint(0, n, n_pairs)
        j_indices = np.random.randint(0, n, n_pairs)
        # Avoid self-pairs
        mask = i_indices != j_indices
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
    
    # Compute 3D separations
    # Assume stars are in the disk plane with azimuth uniformly distributed
    # For radial correlation, approximate separation as sqrt((R1-R2)^2 + (z1-z2)^2)
    dR = np.abs(R[i_indices] - R[j_indices])
    dz = np.abs(z[i_indices] - z[j_indices])
    
    # For a disk, the in-plane angular separation matters
    # Without phi coordinates, we approximate with radial separation only
    separations = np.sqrt(dR**2 + dz**2)
    
    return separations, i_indices, j_indices


def radial_detrend(delta_v, R, degree=2):
    """
    Remove radial trends from velocity residuals.
    
    This is important because the baryonic model may have systematic errors
    that vary with radius, creating artificial anti-correlations at large
    separations.
    
    Parameters
    ----------
    delta_v : array
        Velocity residuals
    R : array
        Galactocentric radii
    degree : int
        Polynomial degree for detrending
    
    Returns
    -------
    delta_v_detrended : array
        Residuals with radial trend removed
    """
    # Fit polynomial to δv(R)
    coeffs = np.polyfit(R, delta_v, degree)
    trend = np.polyval(coeffs, R)
    delta_v_detrended = delta_v - trend
    
    print(f"  Radial detrending (degree {degree}):")
    print(f"    Original std: {np.std(delta_v):.2f} km/s")
    print(f"    Detrended std: {np.std(delta_v_detrended):.2f} km/s")
    
    return delta_v_detrended


def compute_velocity_correlation(delta_v, separations, i_idx, j_idx, 
                                  r_bins=None, min_pairs_per_bin=50,
                                  detrend=True, R=None, radial_detrend_degree=2):
    """
    Compute velocity-velocity correlation function vs separation.
    
    C(r) = ⟨δv(i) δv(j)⟩ / σ_v²
    
    where the average is over all pairs with separation in [r, r+dr].
    
    Parameters
    ----------
    delta_v : array
        Velocity residuals
    separations : array
        Pair separations (kpc)
    i_idx, j_idx : arrays
        Pair indices
    r_bins : array or None
        Bin edges for separation (kpc)
    min_pairs_per_bin : int
        Minimum pairs required per bin
    detrend : bool
        If True, subtract mean before computing correlations.
        This isolates fluctuation correlations from systematic offsets.
    R : array or None
        Galactocentric radii (needed for radial detrending)
    radial_detrend_degree : int
        Polynomial degree for radial detrending (0 = mean only)
    
    Returns
    -------
    r_centers : array
        Bin centers (kpc)
    correlation : array
        Correlation function C(r)
    correlation_err : array
        Standard error on correlation
    n_pairs : array
        Number of pairs per bin
    """
    if r_bins is None:
        # Log-spaced bins from 0.1 to 10 kpc
        r_bins = np.logspace(-1, 1.0, 25)
    
    # Detrend to isolate fluctuations
    if detrend:
        if R is not None and radial_detrend_degree > 0:
            # Radial detrending - removes systematic R-dependent trends
            delta_v = radial_detrend(delta_v.copy(), R, degree=radial_detrend_degree)
        else:
            # Simple mean subtraction
            delta_v = delta_v - np.mean(delta_v)
            print(f"  Mean subtraction: {np.mean(delta_v):.2f} km/s")
    
    # Normalize residuals
    sigma_v = np.std(delta_v)
    delta_v_norm = delta_v / sigma_v
    
    # Products for each pair
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
        if n_pairs < min_pairs_per_bin:
            continue
        
        # Mean correlation in this bin
        corr = np.mean(products[mask])
        corr_err = np.std(products[mask]) / np.sqrt(n_pairs)
        
        r_centers.append(np.sqrt(r_lo * r_hi))  # Geometric mean
        correlations.append(corr)
        correlation_errs.append(corr_err)
        n_pairs_list.append(n_pairs)
    
    return (np.array(r_centers), np.array(correlations), 
            np.array(correlation_errs), np.array(n_pairs_list))


def compute_radial_velocity_correlation(df, r_sep_bins=None, 
                                         delta_R_max=1.0,
                                         delta_z_max=0.3):
    """
    Compute velocity correlation between stars at different radii.
    
    This is a simpler test: compare velocities of stars in narrow radial shells.
    Stars in nearby radial shells should have correlated residuals if Σ-Gravity
    is correct.
    
    Parameters
    ----------
    df : DataFrame
        With R_kpc, z_kpc, delta_v_baryon
    r_sep_bins : array
        Radial separation bins (kpc)
    delta_R_max : float
        Max radial width of each shell
    delta_z_max : float
        Max height for disk selection
    
    Returns
    -------
    r_sep : array
        Radial separations
    correlation : array
        Correlation values
    correlation_err : array
        Errors
    """
    if r_sep_bins is None:
        r_sep_bins = np.linspace(0, 8, 40)
    
    # Select thin disk stars
    disk = df[np.abs(df['z_kpc']) < delta_z_max].copy()
    
    # Group by radius bins
    R_bins = np.arange(4, 16, delta_R_max)
    
    results = []
    
    for i, R1_lo in enumerate(R_bins[:-1]):
        R1_hi = R1_lo + delta_R_max
        shell1 = disk[(disk['R_kpc'] >= R1_lo) & (disk['R_kpc'] < R1_hi)]
        
        if len(shell1) < 100:
            continue
        
        mean_v1 = shell1['delta_v_baryon'].mean()
        std_v1 = shell1['delta_v_baryon'].std()
        n1 = len(shell1)
        
        for j, R2_lo in enumerate(R_bins[i+1:], start=i+1):
            R2_hi = R2_lo + delta_R_max
            shell2 = disk[(disk['R_kpc'] >= R2_lo) & (disk['R_kpc'] < R2_hi)]
            
            if len(shell2) < 100:
                continue
            
            mean_v2 = shell2['delta_v_baryon'].mean()
            std_v2 = shell2['delta_v_baryon'].std()
            n2 = len(shell2)
            
            # Radial separation
            r_sep = (R2_lo + R2_hi) / 2 - (R1_lo + R1_hi) / 2
            
            # Correlation of shell means (binned estimator)
            # This tests if distant shells have correlated residuals
            results.append({
                'R1': (R1_lo + R1_hi) / 2,
                'R2': (R2_lo + R2_hi) / 2,
                'r_sep': r_sep,
                'mean_v1': mean_v1,
                'mean_v2': mean_v2,
                'n1': n1,
                'n2': n2,
            })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Compute correlation at each separation
    # Use sign of mean residuals to detect coherent patterns
    r_sep_unique = np.sort(df_results['r_sep'].unique())
    
    correlations = []
    r_centers = []
    
    for r in r_sep_unique:
        subset = df_results[df_results['r_sep'] == r]
        if len(subset) < 3:
            continue
        
        # Product of residual signs/magnitudes
        products = subset['mean_v1'].values * subset['mean_v2'].values
        corr = np.mean(products) / (df['delta_v_baryon'].std()**2)
        
        correlations.append(corr)
        r_centers.append(r)
    
    return np.array(r_centers), np.array(correlations), np.ones(len(correlations)) * 0.1


# ========== FITTING AND ANALYSIS ==========

def fit_coherence_kernel(r_data, corr_data, corr_err=None):
    """
    Fit the Σ-Gravity coherence kernel to measured correlations.
    
    Returns best-fit parameters and uncertainties.
    """
    # Initial guesses based on theory
    p0 = [0.5, 5.0, 0.5]  # A, ℓ₀, n_coh
    bounds = ([0, 0.5, 0.1], [10, 20, 2.0])
    
    try:
        if corr_err is not None and np.all(corr_err > 0):
            popt, pcov = curve_fit(K_coh_parametric, r_data, corr_data, 
                                   p0=p0, sigma=corr_err, bounds=bounds,
                                   maxfev=10000)
        else:
            popt, pcov = curve_fit(K_coh_parametric, r_data, corr_data, 
                                   p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"  Warning: Fit failed ({e})")
        return None, None


def compute_chi2(r_data, corr_data, corr_err, model_func, *params):
    """
    Compute χ² for a model fit.
    """
    model = model_func(r_data, *params)
    if corr_err is None or np.any(corr_err <= 0):
        corr_err = np.ones_like(corr_data) * np.std(corr_data)
    chi2 = np.sum(((corr_data - model) / corr_err)**2)
    return chi2


# ========== VISUALIZATION ==========

def plot_velocity_correlation(r_data, corr_data, corr_err, n_pairs,
                               fit_params=None, output_path=None):
    """
    Plot measured correlation function vs Σ-Gravity prediction.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ========== Panel 1: Correlation function vs separation ==========
    ax = axes[0, 0]
    ax.errorbar(r_data, corr_data, yerr=corr_err, fmt='o', 
                color='black', markersize=6, capsize=3, 
                label='Gaia DR3 data', zorder=5)
    
    # Theoretical predictions
    r_theory = np.linspace(0.1, 10, 200)
    ax.plot(r_theory, K_coh(r_theory, ell0=5.0, n_coh=0.5), 
            'r-', linewidth=2, label=r'Σ-Gravity: $\ell_0=5$ kpc, $n_{\rm coh}=0.5$')
    ax.plot(r_theory, K_coh(r_theory, ell0=2.0, n_coh=0.5), 
            'r--', linewidth=1.5, alpha=0.7, label=r'$\ell_0=2$ kpc')
    ax.plot(r_theory, K_coh(r_theory, ell0=10.0, n_coh=0.5), 
            'r:', linewidth=1.5, alpha=0.7, label=r'$\ell_0=10$ kpc')
    
    # ΛCDM null: no correlation
    ax.axhline(0, color='blue', linestyle='--', linewidth=2, 
               label=r'$\Lambda$CDM null: $C(r) = 0$')
    
    # Best fit if available
    if fit_params is not None:
        A, ell0, n_coh = fit_params
        ax.plot(r_theory, K_coh_parametric(r_theory, A, ell0, n_coh),
                'g-', linewidth=2, alpha=0.8,
                label=f'Best fit: $\\ell_0={ell0:.1f}$ kpc, $n={{}}={n_coh:.2f}$')
    
    ax.set_xlabel('Separation |R - R\'| [kpc]', fontsize=12)
    ax.set_ylabel(r'Correlation $C(r) = \langle\delta v \delta v\'\rangle / \sigma_v^2$', fontsize=12)
    ax.set_title('Velocity Correlation Function', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # ========== Panel 2: Log-log plot ==========
    ax = axes[0, 1]
    mask = corr_data > 0
    if np.sum(mask) > 3:
        ax.errorbar(r_data[mask], corr_data[mask], yerr=corr_err[mask],
                    fmt='o', color='black', markersize=6, capsize=3, label='Data')
        ax.plot(r_theory, K_coh(r_theory, 5.0, 0.5), 'r-', linewidth=2,
                label=r'$(\ell_0/(\ell_0+r))^{0.5}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Separation [kpc]', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Log-Log Scale (Power Law Test)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
    else:
        ax.text(0.5, 0.5, 'Insufficient positive correlations\nfor log-log plot',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
    
    # ========== Panel 3: Number of pairs per bin ==========
    ax = axes[1, 0]
    ax.bar(r_data, n_pairs, width=np.diff(r_data, prepend=0), 
           alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Separation [kpc]', fontsize=12)
    ax.set_ylabel('Number of pairs', fontsize=12)
    ax.set_title('Sample Size per Bin', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # ========== Panel 4: Model comparison ==========
    ax = axes[1, 1]
    
    # Compute chi2 for different models
    chi2_null = compute_chi2(r_data, corr_data, corr_err, K_null, 0.0)
    chi2_sg_5 = compute_chi2(r_data, corr_data, corr_err, K_coh, 5.0, 0.5)
    chi2_sg_2 = compute_chi2(r_data, corr_data, corr_err, K_coh, 2.0, 0.5)
    
    models = [r'$\Lambda$CDM null', r'$\Sigma$-G ($\ell_0$=5)', r'$\Sigma$-G ($\ell_0$=2)']
    chi2s = [chi2_null, chi2_sg_5, chi2_sg_2]
    colors = ['blue', 'red', 'orange']
    
    bars = ax.bar(models, chi2s, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(r'$\chi^2$ (lower is better)', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, chi2 in zip(bars, chi2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{chi2:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {output_path}")
    
    plt.close()
    return fig


def print_summary(r_data, corr_data, corr_err, n_pairs, fit_params=None):
    """
    Print analysis summary.
    """
    print("\n" + "=" * 70)
    print("GAIA VELOCITY CORRELATION TEST - SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Data Summary:")
    print(f"   Separation range: {r_data.min():.2f} - {r_data.max():.2f} kpc")
    print(f"   Number of bins: {len(r_data)}")
    print(f"   Total pairs: {n_pairs.sum():,}")
    
    print(f"\n📈 Correlation Results:")
    print(f"   Max correlation: {corr_data.max():.4f} at r = {r_data[np.argmax(corr_data)]:.2f} kpc")
    print(f"   Mean correlation (r < 2 kpc): {corr_data[r_data < 2].mean():.4f}")
    print(f"   Mean correlation (r > 5 kpc): {corr_data[r_data > 5].mean():.4f}")
    
    # χ² comparison
    chi2_null = compute_chi2(r_data, corr_data, corr_err, K_null, 0.0)
    chi2_sg = compute_chi2(r_data, corr_data, corr_err, K_coh, 5.0, 0.5)
    dof = len(r_data) - 1
    
    print(f"\n🔬 Model Comparison (χ²):")
    print(f"   ΛCDM null (C=0): χ² = {chi2_null:.1f} (χ²/dof = {chi2_null/dof:.2f})")
    print(f"   Σ-Gravity (ℓ₀=5 kpc): χ² = {chi2_sg:.1f} (χ²/dof = {chi2_sg/dof:.2f})")
    print(f"   Δχ² = {chi2_null - chi2_sg:.1f} (positive favors Σ-Gravity)")
    
    if fit_params is not None:
        A, ell0, n_coh = fit_params
        chi2_fit = compute_chi2(r_data, corr_data, corr_err, K_coh_parametric, A, ell0, n_coh)
        print(f"\n📐 Best Fit Parameters:")
        print(f"   Amplitude A = {A:.3f}")
        print(f"   Coherence length ℓ₀ = {ell0:.2f} kpc")
        print(f"   Exponent n_coh = {n_coh:.3f}")
        print(f"   χ² (best fit) = {chi2_fit:.1f}")
        
        # Compare to theoretical prediction
        print(f"\n📋 Comparison to Theory:")
        print(f"   Theory predicts: ℓ₀ ≈ 5 kpc, n_coh ≈ 0.5")
        print(f"   Fitted values: ℓ₀ = {ell0:.2f} kpc, n_coh = {n_coh:.3f}")
        if 2 < ell0 < 10 and 0.3 < n_coh < 0.7:
            print(f"   ✅ CONSISTENT with Σ-Gravity prediction!")
        else:
            print(f"   ⚠️  Parameters differ from theoretical prediction")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if chi2_sg < chi2_null - 10:
        print("""
✅ SIGNIFICANT CORRELATION DETECTED

The velocity residuals show spatial correlations inconsistent with 
ΛCDM's null hypothesis (C=0 beyond ~100 pc). The correlation pattern
appears consistent with Σ-Gravity's coherence kernel.

This result supports the non-local gravitational coupling hypothesis.
Follow-up with larger samples is recommended.
""")
    elif chi2_sg < chi2_null:
        print("""
⚠️  WEAK CORRELATION DETECTED

Some correlation signal is present, but not statistically decisive.
The current sample may be insufficient to distinguish models.

Recommendation: Analyze larger Gaia sample or refine selection criteria.
""")
    else:
        print("""
❓ INCONCLUSIVE / NULL CONSISTENT

No significant correlation detected beyond random fluctuations.
This is consistent with ΛCDM's null hypothesis.

Possible interpretations:
1. Σ-Gravity's coherence length is smaller than expected
2. Sample selection/systematics masking the signal
3. The coherence effect is weaker than predicted
""")


# ========== MAIN EXECUTION ==========

def run_correlation_test(data_file=None, output_dir=None, max_pairs=500000,
                          method='pair_wise', use_large_sample=False):
    """
    Run the full velocity correlation analysis.
    
    Parameters
    ----------
    data_file : str or Path
        Path to predicted Gaia data
    output_dir : str or Path
        Output directory for plots/results
    max_pairs : int
        Maximum pairs to compute (memory limit)
    method : str
        'pair_wise' - direct pair correlation
        'radial' - shell-to-shell correlation
    use_large_sample : bool
        If True, use the 1.8M star dataset with flat rotation curve model
    """
    print("\n" + "=" * 70)
    print("GAIA DR3 VELOCITY CORRELATION TEST FOR Σ-GRAVITY")
    print("=" * 70)
    print("""
This test checks whether velocity residuals (v_obs - v_baryon) are
spatially correlated according to Σ-Gravity's coherence kernel:

    C(r) = ⟨δv(R) δv(R')⟩ ∝ (ℓ₀/(ℓ₀+r))^n_coh

Theory predicts: ℓ₀ ≈ 5 kpc, n_coh ≈ 0.5
ΛCDM predicts: C(r) ≈ 0 for r > 100 pc
""")
    
    # Setup output
    if output_dir is None:
        output_dir = ROOT / "GravityWaveTest" / "outputs" / "velocity_correlation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading Gaia data...")
    if use_large_sample:
        df = load_large_gaia_data(data_file)
    else:
        df = load_predicted_gaia_data(data_file)
    print(f"  Using {len(df):,} stars")
    
    # Compute correlations
    print("\n[2/4] Computing pair correlations...")
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    delta_v = df['delta_v_baryon'].values
    
    # Print residual statistics
    print(f"  Velocity residual stats:")
    print(f"    Mean: {np.mean(delta_v):.2f} km/s")
    print(f"    Std:  {np.std(delta_v):.2f} km/s")
    print(f"    Range: {np.min(delta_v):.1f} to {np.max(delta_v):.1f} km/s")
    
    if method == 'pair_wise':
        # Direct pair-wise correlation
        separations, i_idx, j_idx = compute_pair_separations(R, z, max_pairs=max_pairs)
        
        # Log-spaced bins for power-law detection
        r_bins = np.logspace(-0.5, 1.0, 30)
        
        r_data, corr_data, corr_err, n_pairs = compute_velocity_correlation(
            delta_v, separations, i_idx, j_idx, r_bins=r_bins,
            detrend=True, R=R, radial_detrend_degree=2
        )
    else:
        # Radial shell correlation
        r_data, corr_data, corr_err = compute_radial_velocity_correlation(df)
        n_pairs = np.ones_like(r_data) * 1000  # Placeholder
    
    if len(r_data) == 0:
        print("ERROR: No valid correlation bins computed!")
        return None
    
    print(f"  Computed {len(r_data)} separation bins")
    
    # Fit coherence kernel
    print("\n[3/4] Fitting coherence kernel...")
    fit_params, fit_errors = fit_coherence_kernel(r_data, corr_data, corr_err)
    
    if fit_params is not None:
        print(f"  Best fit: A={fit_params[0]:.3f}, ℓ₀={fit_params[1]:.2f} kpc, n={fit_params[2]:.3f}")
    
    # Generate plots
    print("\n[4/4] Generating plots...")
    plot_path = output_dir / "velocity_correlation_test.png"
    plot_velocity_correlation(r_data, corr_data, corr_err, n_pairs,
                               fit_params=fit_params, output_path=plot_path)
    
    # Print summary
    print_summary(r_data, corr_data, corr_err, n_pairs, fit_params)
    
    # Save results
    results_df = pd.DataFrame({
        'r_kpc': r_data,
        'correlation': corr_data,
        'correlation_err': corr_err,
        'n_pairs': n_pairs
    })
    results_path = output_dir / "velocity_correlation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved results: {results_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return {
        'r_data': r_data,
        'corr_data': corr_data,
        'corr_err': corr_err,
        'n_pairs': n_pairs,
        'fit_params': fit_params,
        'fit_errors': fit_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gaia DR3 Velocity Correlation Test for Σ-Gravity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gaia_velocity_correlation_test.py
  python gaia_velocity_correlation_test.py --max-pairs 1000000
  python gaia_velocity_correlation_test.py --method radial

This test checks the critical Σ-Gravity prediction that velocity residuals
should be correlated according to the coherence kernel K_coh(r).
        """
    )
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to predicted Gaia data CSV')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--max-pairs', type=int, default=500000,
                        help='Maximum pairs to compute (default: 500000)')
    parser.add_argument('--method', choices=['pair_wise', 'radial'], 
                        default='pair_wise',
                        help='Correlation method (default: pair_wise)')
    parser.add_argument('--large-sample', action='store_true',
                        help='Use the 1.8M star dataset (with flat RC model)')
    
    args = parser.parse_args()
    
    results = run_correlation_test(
        data_file=args.data_file,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        method=args.method,
        use_large_sample=args.large_sample
    )
    
    return results


if __name__ == "__main__":
    main()
