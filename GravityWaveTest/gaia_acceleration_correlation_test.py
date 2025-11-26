#!/usr/bin/env python3
"""
Gaia DR3 Acceleration Correlation Test for Σ-Gravity
=====================================================

MOTIVATION:
-----------
The velocity correlation test showed ℓ₀ ≈ 0.5 kpc, not the predicted 5 kpc.
But Σ-Gravity predicts enhanced *gravitational acceleration*, not velocities:

    g_eff = g_baryon × [1 + K(R)]

Velocities integrate over orbital history and suffer from phase mixing.
Accelerations are instantaneous probes of the local gravitational field.

This test uses the Jeans equation to extract accelerations from Gaia kinematics:

    a_R = -σ_R² × [d ln ν/dR + d ln σ_R²/dR + (1 - σ_φ²/σ_R²)/R]

We then test for spatial correlations in:
    δa = a_obs - a_Newton

If Σ-Gravity is correct, ⟨δa(R) δa(R')⟩ should follow K_coh(|R-R'|) with ℓ₀ ≈ 5 kpc.

KEY DIFFERENCE FROM VELOCITY TEST:
----------------------------------
- Velocities: probe time-averaged orbital dynamics
- Accelerations: probe instantaneous gravitational field
- If coherence acts on potential/acceleration, velocity correlations may be
  washed out while acceleration correlations survive.

See README.md for API key setup.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# Try CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy available - GPU acceleration enabled")
except ImportError:
    HAS_CUPY = False
    cp = None
    print("CuPy not available - using CPU")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ========== PHYSICAL CONSTANTS ==========
G_GRAV = 4.302e-6  # kpc (km/s)² / M_sun


# ========== COHERENCE KERNEL ==========

def K_coh(r, ell0=5.0, n_coh=0.5):
    """Σ-Gravity coherence kernel."""
    r = np.asarray(r)
    return (ell0 / (ell0 + r)) ** n_coh


def K_coh_parametric(r, A, ell0, n_coh):
    """Parametric coherence kernel for fitting."""
    return A * (ell0 / (ell0 + r)) ** n_coh


# ========== DATA LOADING ==========

def load_gaia_kinematics(filepath=None):
    """
    Load Gaia data with full kinematics for Jeans analysis.
    
    Returns DataFrame with:
    - R_kpc, z_kpc, phi: cylindrical coordinates
    - v_R, v_phi, v_z: velocity components (if available)
    - For Jeans equation we need velocity dispersions in radial bins
    """
    if filepath is None:
        filepath = ROOT / "data" / "gaia" / "gaia_processed_corrected.csv"
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Gaia data not found: {filepath}")
    
    print(f"Loading Gaia kinematics from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Total stars: {len(df):,}")
    
    # Map columns
    df_out = pd.DataFrame()
    df_out['R_kpc'] = df['R_cyl'] if 'R_cyl' in df.columns else df['R_kpc']
    df_out['z_kpc'] = df['z'] if 'z' in df.columns else df['z_kpc']
    df_out['phi'] = df['phi'] if 'phi' in df.columns else np.zeros(len(df))
    
    # Velocities
    df_out['v_phi'] = df['v_phi'] if 'v_phi' in df.columns else df['v_obs_kms']
    
    # We need v_R (radial velocity in cylindrical coords)
    # Check if available, otherwise estimate from proper motions
    if 'v_rad' in df.columns:
        # This is typically line-of-sight velocity, need to transform
        df_out['v_R'] = df['v_rad']  # Approximation for disk stars
    else:
        # Without radial velocities, we can only do partial Jeans analysis
        df_out['v_R'] = np.nan
    
    # Cartesian for pair separations
    df_out['x_kpc'] = df_out['R_kpc'] * np.cos(df_out['phi'])
    df_out['y_kpc'] = df_out['R_kpc'] * np.sin(df_out['phi'])
    
    # Filter to disk
    disk_mask = (df_out['R_kpc'] >= 4) & (df_out['R_kpc'] <= 16) & (np.abs(df_out['z_kpc']) < 0.5)
    df_disk = df_out[disk_mask].copy().reset_index(drop=True)
    print(f"  Thin disk stars (4 < R < 16 kpc, |z| < 0.5 kpc): {len(df_disk):,}")
    
    return df_disk


# ========== JEANS EQUATION ANALYSIS ==========

def compute_radial_profiles(df, R_bins=None, min_stars_per_bin=100):
    """
    Compute radial profiles of density and velocity dispersions.
    
    Returns profiles needed for Jeans equation:
    - ν(R): stellar number density
    - σ_φ(R): azimuthal velocity dispersion
    - σ_R(R): radial velocity dispersion (if v_R available)
    """
    if R_bins is None:
        R_bins = np.linspace(4, 16, 49)  # 0.25 kpc bins
    
    R = df['R_kpc'].values
    v_phi = df['v_phi'].values
    
    R_centers = []
    nu_profile = []
    sigma_phi_profile = []
    sigma_R_profile = []
    mean_v_phi = []
    n_stars = []
    
    for i in range(len(R_bins) - 1):
        R_lo, R_hi = R_bins[i], R_bins[i + 1]
        mask = (R >= R_lo) & (R < R_hi)
        n = np.sum(mask)
        
        if n < min_stars_per_bin:
            continue
        
        R_mid = (R_lo + R_hi) / 2
        
        # Number density (proportional to counts / annular area)
        area = np.pi * (R_hi**2 - R_lo**2)
        nu = n / area
        
        # Azimuthal velocity dispersion
        v_phi_bin = v_phi[mask]
        sigma_phi = np.std(v_phi_bin)
        v_phi_mean = np.mean(v_phi_bin)
        
        # Radial velocity dispersion (if available)
        if 'v_R' in df.columns and not df['v_R'].isna().all():
            v_R_bin = df.loc[mask, 'v_R'].dropna()
            if len(v_R_bin) > 10:
                sigma_R = np.std(v_R_bin)
            else:
                # Estimate from σ_φ using typical MW ratio σ_R/σ_φ ≈ 1.5
                sigma_R = sigma_phi * 1.5
        else:
            sigma_R = sigma_phi * 1.5  # Approximation
        
        R_centers.append(R_mid)
        nu_profile.append(nu)
        sigma_phi_profile.append(sigma_phi)
        sigma_R_profile.append(sigma_R)
        mean_v_phi.append(v_phi_mean)
        n_stars.append(n)
    
    return pd.DataFrame({
        'R_kpc': R_centers,
        'nu': nu_profile,
        'sigma_phi': sigma_phi_profile,
        'sigma_R': sigma_R_profile,
        'v_phi_mean': mean_v_phi,
        'n_stars': n_stars
    })


def compute_jeans_acceleration(profiles, smooth_sigma=2):
    """
    Compute radial acceleration from Jeans equation.
    
    Axisymmetric Jeans equation in cylindrical coords (R, φ, z):
    
    a_R = -σ_R² × [d ln ν/dR + d ln σ_R²/dR + (1 - σ_φ²/σ_R²)/R]
    
    This gives the radial acceleration needed to support the observed
    velocity dispersion and density profiles.
    
    Parameters
    ----------
    profiles : DataFrame
        From compute_radial_profiles()
    smooth_sigma : float
        Gaussian smoothing for derivative estimation
    
    Returns
    -------
    R : array
        Radii (kpc)
    a_jeans : array
        Jeans-inferred radial acceleration (km²/s²/kpc)
    """
    R = np.array(profiles['R_kpc'])
    nu = np.array(profiles['nu'])
    sigma_R = np.array(profiles['sigma_R'])
    sigma_phi = np.array(profiles['sigma_phi'])
    
    # Smooth profiles before taking derivatives
    if smooth_sigma > 0:
        nu_smooth = gaussian_filter1d(nu, smooth_sigma)
        sigma_R_smooth = gaussian_filter1d(sigma_R, smooth_sigma)
        sigma_phi_smooth = gaussian_filter1d(sigma_phi, smooth_sigma)
    else:
        nu_smooth = nu
        sigma_R_smooth = sigma_R
        sigma_phi_smooth = sigma_phi
    
    # Compute logarithmic derivatives using finite differences
    dR = np.gradient(R)
    
    # d ln ν / dR
    d_ln_nu_dR = np.gradient(np.log(nu_smooth + 1e-10), R)
    
    # d ln σ_R² / dR = 2 × d ln σ_R / dR
    d_ln_sigR2_dR = 2 * np.gradient(np.log(sigma_R_smooth + 1e-10), R)
    
    # Anisotropy term: (1 - σ_φ²/σ_R²) / R
    beta = 1 - (sigma_phi_smooth / sigma_R_smooth)**2
    anisotropy_term = beta / R
    
    # Jeans acceleration
    sigma_R2 = sigma_R_smooth**2
    a_jeans = -sigma_R2 * (d_ln_nu_dR + d_ln_sigR2_dR + anisotropy_term)
    
    return R, a_jeans, {
        'd_ln_nu_dR': d_ln_nu_dR,
        'd_ln_sigR2_dR': d_ln_sigR2_dR,
        'anisotropy': anisotropy_term,
        'sigma_R': sigma_R_smooth,
        'sigma_phi': sigma_phi_smooth
    }


def compute_newtonian_acceleration(R, M_disk=5e10, R_d=2.5, M_bulge=1e10, a_bulge=0.5):
    """
    Compute expected Newtonian radial acceleration from baryonic mass model.
    
    Simple model:
    - Exponential disk: M_disk, scale length R_d
    - Hernquist bulge: M_bulge, scale radius a_bulge
    
    Returns a_Newton in km²/s²/kpc (negative = inward)
    """
    # Disk contribution (approximate for exponential disk)
    # For exact, would use Bessel functions - this is the flat RC approximation
    y = R / (2 * R_d)
    # Use disk circular velocity formula
    v_disk2 = G_GRAV * M_disk * R / (R + R_d)**2 * 0.5  # Simplified
    
    # More accurate: enclosed mass approximation
    M_enc_disk = M_disk * (1 - (1 + R/R_d) * np.exp(-R/R_d))
    a_disk = -G_GRAV * M_enc_disk / R**2
    
    # Bulge contribution (Hernquist profile)
    M_enc_bulge = M_bulge * R**2 / (R + a_bulge)**2
    a_bulge_comp = -G_GRAV * M_enc_bulge / R**2
    
    a_Newton = a_disk + a_bulge_comp
    
    return a_Newton


# ========== CORRELATION ANALYSIS ==========

def compute_acceleration_residuals(df, profiles, smooth_sigma=2):
    """
    Compute acceleration residuals δa = a_obs - a_Newton for each star.
    
    Assigns each star the Jeans-inferred acceleration at its radius,
    then subtracts the Newtonian prediction.
    """
    # Get Jeans accelerations on profile grid
    R_grid, a_jeans, diagnostics = compute_jeans_acceleration(profiles, smooth_sigma)
    
    # Get Newtonian accelerations
    a_newton = compute_newtonian_acceleration(R_grid)
    
    # Residuals on grid
    delta_a_grid = a_jeans - a_newton
    
    # Interpolate to each star's radius
    from scipy.interpolate import interp1d
    interp_func = interp1d(R_grid, delta_a_grid, kind='linear', 
                           bounds_error=False, fill_value=0)
    
    R_stars = df['R_kpc'].values
    delta_a_stars = interp_func(R_stars)
    
    return delta_a_stars, R_grid, a_jeans, a_newton, diagnostics


def compute_velocity_fluctuations(df, R_bins=None, v_flat=220.0):
    """
    Compute velocity fluctuations around the mean rotation at each radius.
    
    This is a better test: instead of using Jeans-derived accelerations
    (which are radially smooth), use individual stellar velocity deviations
    from the local mean.
    
    δv_i = v_phi_i - <v_phi>(R_i)
    
    This probes fluctuations in the gravitational potential at fixed R.
    """
    if R_bins is None:
        R_bins = np.linspace(4, 16, 121)  # Fine bins: 0.1 kpc
    
    R = df['R_kpc'].values
    v_phi = df['v_phi'].values
    
    # Compute mean velocity in each bin
    R_centers = (R_bins[:-1] + R_bins[1:]) / 2
    v_mean = np.zeros(len(R_bins) - 1)
    v_std = np.zeros(len(R_bins) - 1)
    
    for i in range(len(R_bins) - 1):
        mask = (R >= R_bins[i]) & (R < R_bins[i + 1])
        if np.sum(mask) > 10:
            v_mean[i] = np.mean(v_phi[mask])
            v_std[i] = np.std(v_phi[mask])
        else:
            v_mean[i] = np.nan
            v_std[i] = np.nan
    
    # Interpolate to get expected velocity at each star's radius
    from scipy.interpolate import interp1d
    v_interp = interp1d(R_centers, v_mean, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    
    v_expected = v_interp(R)
    
    # Fluctuations around local mean
    delta_v = v_phi - v_expected
    
    # Also compute local dispersion for normalization
    sigma_interp = interp1d(R_centers, v_std, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
    sigma_local = sigma_interp(R)
    
    # Normalize by local dispersion
    delta_v_norm = delta_v / np.where(sigma_local > 0, sigma_local, 1)
    
    return delta_v_norm, R_centers, v_mean, v_std


def compute_acceleration_correlation_gpu(x, y, z, delta_a, r_bins, batch_size=2000):
    """
    GPU-accelerated acceleration correlation computation.
    
    C(r) = ⟨δa(R) δa(R')⟩ / σ_a²
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy not available")
    
    n = len(x)
    n_bins = len(r_bins) - 1
    j_chunk_size = min(100000, n)
    
    print(f"  GPU correlation: {n:,} stars, {n_bins} bins")
    print(f"  Batch sizes: i={batch_size}, j={j_chunk_size}")
    
    # Normalize residuals
    sigma_a = np.std(delta_a)
    delta_a_norm = delta_a / sigma_a
    
    # Prepare arrays
    x_np = x.astype(np.float32)
    y_np = y.astype(np.float32)
    z_np = z.astype(np.float32)
    a_np = delta_a_norm.astype(np.float32)
    r_bins_gpu = cp.asarray(r_bins, dtype=cp.float32)
    
    # Accumulators
    sum_prod = cp.zeros(n_bins, dtype=cp.float64)
    sum_prod2 = cp.zeros(n_bins, dtype=cp.float64)
    counts = cp.zeros(n_bins, dtype=cp.int64)
    
    max_sep = r_bins[-1]
    max_sep2 = max_sep ** 2
    
    n_i_batches = (n + batch_size - 1) // batch_size
    last_report = 0
    
    print(f"  Processing {n_i_batches} i-batches...")
    
    for i_batch in range(n_i_batches):
        i_start = i_batch * batch_size
        i_end = min(i_start + batch_size, n)
        
        x_i = cp.asarray(x_np[i_start:i_end, None], dtype=cp.float32)
        y_i = cp.asarray(y_np[i_start:i_end, None], dtype=cp.float32)
        z_i = cp.asarray(z_np[i_start:i_end, None], dtype=cp.float32)
        a_i = cp.asarray(a_np[i_start:i_end, None], dtype=cp.float32)
        
        global_i = cp.arange(i_start, i_end)[:, None]
        
        j_min = i_start
        n_j_chunks = ((n - j_min) + j_chunk_size - 1) // j_chunk_size
        
        for j_chunk in range(n_j_chunks):
            j_start = j_min + j_chunk * j_chunk_size
            j_end = min(j_start + j_chunk_size, n)
            
            x_j = cp.asarray(x_np[j_start:j_end], dtype=cp.float32)
            y_j = cp.asarray(y_np[j_start:j_end], dtype=cp.float32)
            z_j = cp.asarray(z_np[j_start:j_end], dtype=cp.float32)
            a_j = cp.asarray(a_np[j_start:j_end], dtype=cp.float32)
            
            dx = x_i - x_j
            dy = y_i - y_j
            dz = z_i - z_j
            r2 = dx**2 + dy**2 + dz**2
            
            products = a_i * a_j
            
            global_j = cp.arange(j_start, j_end)
            upper_tri_mask = global_j > global_i
            
            valid = (r2 <= max_sep2) & (r2 > 0) & upper_tri_mask
            
            r = cp.sqrt(r2)
            bin_indices = cp.searchsorted(r_bins_gpu, r, side='right') - 1
            
            for b in range(n_bins):
                bin_mask = valid & (bin_indices == b)
                cnt = cp.sum(bin_mask)
                if cnt > 0:
                    p = products[bin_mask]
                    sum_prod[b] += cp.sum(p)
                    sum_prod2[b] += cp.sum(p**2)
                    counts[b] += cnt
            
            del x_j, y_j, z_j, a_j, dx, dy, dz, r2, products, r, bin_indices, valid, global_j
        
        del x_i, y_i, z_i, a_i, global_i
        cp.get_default_memory_pool().free_all_blocks()
        
        progress = (i_batch + 1) / n_i_batches * 100
        if progress >= last_report + 10:
            print(f"    Progress: {progress:.0f}%")
            last_report = int(progress // 10) * 10
    
    # Results
    sum_prod_cpu = cp.asnumpy(sum_prod)
    sum_prod2_cpu = cp.asnumpy(sum_prod2)
    counts_cpu = cp.asnumpy(counts)
    
    r_centers = []
    correlations = []
    correlation_errs = []
    n_pairs_list = []
    
    for b in range(n_bins):
        if counts_cpu[b] < 50:
            continue
        
        n_pairs = counts_cpu[b]
        mean_corr = sum_prod_cpu[b] / n_pairs
        var_corr = sum_prod2_cpu[b] / n_pairs - mean_corr**2
        std_err = np.sqrt(var_corr / n_pairs) if var_corr > 0 else 0.001
        
        r_lo, r_hi = r_bins[b], r_bins[b+1]
        r_centers.append(np.sqrt(r_lo * r_hi))
        correlations.append(mean_corr)
        correlation_errs.append(std_err)
        n_pairs_list.append(n_pairs)
    
    print(f"  Total pairs processed: {counts_cpu.sum():,}")
    
    return (np.array(r_centers), np.array(correlations), 
            np.array(correlation_errs), np.array(n_pairs_list))


def compute_acceleration_correlation_cpu(df, delta_a, r_bins, max_pairs=500000):
    """
    CPU fallback for acceleration correlation.
    Uses random pair sampling.
    """
    n = len(df)
    x = df['x_kpc'].values
    y = df['y_kpc'].values
    z = df['z_kpc'].values
    
    # Random pairs
    np.random.seed(42)
    i_idx = np.random.randint(0, n, max_pairs)
    j_idx = np.random.randint(0, n, max_pairs)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    # Separations
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dz = z[i_idx] - z[j_idx]
    separations = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Normalize
    sigma_a = np.std(delta_a)
    delta_a_norm = delta_a / sigma_a
    
    products = delta_a_norm[i_idx] * delta_a_norm[j_idx]
    
    # Bin
    r_centers = []
    correlations = []
    correlation_errs = []
    n_pairs_list = []
    
    for i in range(len(r_bins) - 1):
        r_lo, r_hi = r_bins[i], r_bins[i + 1]
        bin_mask = (separations >= r_lo) & (separations < r_hi)
        n_pairs = np.sum(bin_mask)
        
        if n_pairs < 50:
            continue
        
        corr = np.mean(products[bin_mask])
        corr_err = np.std(products[bin_mask]) / np.sqrt(n_pairs)
        
        r_centers.append(np.sqrt(r_lo * r_hi))
        correlations.append(corr)
        correlation_errs.append(corr_err)
        n_pairs_list.append(n_pairs)
    
    return (np.array(r_centers), np.array(correlations), 
            np.array(correlation_errs), np.array(n_pairs_list))


# ========== VISUALIZATION ==========

def plot_acceleration_analysis(R_grid, a_jeans, a_newton, diagnostics,
                                r_data, corr_data, corr_err, n_pairs,
                                fit_params=None, output_path=None):
    """
    Comprehensive plot of acceleration analysis and correlations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel 1: Jeans vs Newtonian acceleration
    ax = axes[0, 0]
    ax.plot(R_grid, -a_jeans, 'b-', linewidth=2, label='Jeans (observed)')
    ax.plot(R_grid, -a_newton, 'r--', linewidth=2, label='Newtonian (baryonic)')
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel(r'$|a_R|$ [km²/s²/kpc]', fontsize=12)
    ax.set_title('Radial Acceleration Profile', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Panel 2: Acceleration residual
    ax = axes[0, 1]
    delta_a = a_jeans - a_newton
    ax.plot(R_grid, delta_a, 'g-', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel(r'$\delta a = a_{Jeans} - a_{Newton}$ [km²/s²/kpc]', fontsize=12)
    ax.set_title('Acceleration Residual (Dark Acceleration)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Jeans equation components
    ax = axes[0, 2]
    ax.plot(R_grid, diagnostics['d_ln_nu_dR'], label=r'd ln ν/dR')
    ax.plot(R_grid, diagnostics['d_ln_sigR2_dR'], label=r'd ln σ_R²/dR')
    ax.plot(R_grid, diagnostics['anisotropy'], label=r'(1-σ_φ²/σ_R²)/R')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Component value [1/kpc]', fontsize=12)
    ax.set_title('Jeans Equation Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Acceleration correlation function
    ax = axes[1, 0]
    ax.errorbar(r_data, corr_data, yerr=corr_err, fmt='o', 
                color='black', markersize=6, capsize=3, label='Data')
    
    r_theory = np.linspace(0.1, 10, 200)
    ax.plot(r_theory, K_coh(r_theory, ell0=5.0, n_coh=0.5), 
            'r-', linewidth=2, label=r'Σ-Gravity: $\ell_0=5$ kpc')
    ax.plot(r_theory, K_coh(r_theory, ell0=0.5, n_coh=0.5), 
            'orange', linewidth=2, linestyle='--', label=r'$\ell_0=0.5$ kpc')
    ax.axhline(0, color='blue', linestyle='--', linewidth=2, label='ΛCDM null')
    
    if fit_params is not None:
        A, ell0, n_coh = fit_params
        ax.plot(r_theory, K_coh_parametric(r_theory, A, ell0, n_coh),
                'g-', linewidth=2, alpha=0.8,
                label=f'Best fit: $\\ell_0$={ell0:.2f} kpc')
    
    ax.set_xlabel('Separation [kpc]', fontsize=12)
    ax.set_ylabel(r'$C(r) = \langle\delta a \delta a\'\rangle / \sigma_a^2$', fontsize=12)
    ax.set_title('Acceleration Correlation Function', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    # Panel 5: Log-log correlation
    ax = axes[1, 1]
    mask = corr_data > 0
    if np.sum(mask) > 3:
        ax.errorbar(r_data[mask], corr_data[mask], yerr=corr_err[mask],
                    fmt='o', color='black', markersize=6, capsize=3)
        ax.plot(r_theory, K_coh(r_theory, 5.0, 0.5), 'r-', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('Separation [kpc]', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Log-Log (Power Law Test)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 6: Pair counts
    ax = axes[1, 2]
    ax.bar(r_data, n_pairs, width=np.diff(r_data, prepend=0), 
           alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Separation [kpc]', fontsize=12)
    ax.set_ylabel('Number of pairs', fontsize=12)
    ax.set_title('Sample Size', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {output_path}")
    
    plt.close()
    return fig


# ========== MAIN EXECUTION ==========

def run_acceleration_test(data_file=None, output_dir=None, use_gpu=True, 
                          subsample_n=None, max_sep=10.0, batch_size=2000):
    """
    Run the full acceleration correlation analysis.
    """
    print("\n" + "=" * 70)
    print("GAIA DR3 ACCELERATION CORRELATION TEST FOR Σ-GRAVITY")
    print("=" * 70)
    print("""
This test uses the Jeans equation to extract accelerations from kinematics:

    a_R = -σ_R² × [d ln ν/dR + d ln σ_R²/dR + (1 - σ_φ²/σ_R²)/R]

We test for spatial correlations in acceleration residuals:
    
    δa = a_Jeans - a_Newton

Theory predicts: ⟨δa δa'⟩ ∝ K_coh(r) with ℓ₀ ≈ 5 kpc
(This is a more direct test than velocity correlations)
""")
    
    # Setup
    if output_dir is None:
        output_dir = ROOT / "GravityWaveTest" / "outputs" / "acceleration_correlation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading Gaia kinematics...")
    df = load_gaia_kinematics(data_file)
    
    if subsample_n and len(df) > subsample_n:
        print(f"  Subsampling to {subsample_n:,} stars")
        df = df.sample(n=subsample_n, random_state=42).reset_index(drop=True)
    print(f"  Using {len(df):,} stars")
    
    # Compute radial profiles
    print("\n[2/5] Computing radial profiles...")
    profiles = compute_radial_profiles(df)
    print(f"  Computed {len(profiles)} radial bins")
    print(f"  σ_φ range: {profiles['sigma_phi'].min():.1f} - {profiles['sigma_phi'].max():.1f} km/s")
    print(f"  σ_R range: {profiles['sigma_R'].min():.1f} - {profiles['sigma_R'].max():.1f} km/s")
    
    # Compute acceleration residuals
    print("\n[3/5] Computing Jeans accelerations and residuals...")
    delta_a, R_grid, a_jeans, a_newton, diagnostics = compute_acceleration_residuals(
        df, profiles, smooth_sigma=2
    )
    
    print(f"  Acceleration residual stats:")
    print(f"    Mean: {np.nanmean(delta_a):.2f} km²/s²/kpc")
    print(f"    Std:  {np.nanstd(delta_a):.2f} km²/s²/kpc")
    
    # Remove NaN values
    valid_mask = ~np.isnan(delta_a)
    df_valid = df[valid_mask].reset_index(drop=True)
    delta_a_valid = delta_a[valid_mask]
    print(f"  Valid stars (no NaN): {len(df_valid):,}")
    
    # Compute correlations
    print("\n[4/5] Computing acceleration correlations...")
    r_bins = np.logspace(-0.5, np.log10(max_sep), 30)
    
    if use_gpu and HAS_CUPY:
        x = df_valid['x_kpc'].values
        y = df_valid['y_kpc'].values
        z = df_valid['z_kpc'].values
        r_data, corr_data, corr_err, n_pairs = compute_acceleration_correlation_gpu(
            x, y, z, delta_a_valid, r_bins, batch_size=batch_size
        )
    else:
        r_data, corr_data, corr_err, n_pairs = compute_acceleration_correlation_cpu(
            df_valid, delta_a_valid, r_bins
        )
    
    print(f"  Computed {len(r_data)} separation bins")
    
    # Fit coherence kernel
    print("\n[5/5] Fitting and plotting...")
    try:
        p0 = [0.5, 5.0, 0.5]
        bounds = ([0, 0.1, 0.1], [10, 20, 2.0])
        popt, _ = curve_fit(K_coh_parametric, r_data, corr_data, 
                           p0=p0, sigma=corr_err, bounds=bounds, maxfev=10000)
        fit_params = popt
        print(f"  Best fit: A={popt[0]:.3f}, ℓ₀={popt[1]:.2f} kpc, n={popt[2]:.3f}")
    except Exception as e:
        print(f"  Fit failed: {e}")
        fit_params = None
    
    # Plot
    plot_path = output_dir / "acceleration_correlation_test.png"
    plot_acceleration_analysis(
        R_grid, a_jeans, a_newton, diagnostics,
        r_data, corr_data, corr_err, n_pairs,
        fit_params=fit_params, output_path=plot_path
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("ACCELERATION CORRELATION TEST - SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Data Summary:")
    print(f"   Stars analyzed: {len(df_valid):,}")
    print(f"   Total pairs: {n_pairs.sum():,}")
    
    print(f"\n📈 Correlation Results:")
    print(f"   Max correlation: {corr_data.max():.4f} at r = {r_data[np.argmax(corr_data)]:.2f} kpc")
    if len(r_data[r_data < 2]) > 0:
        print(f"   Mean correlation (r < 2 kpc): {corr_data[r_data < 2].mean():.4f}")
    if len(r_data[r_data > 5]) > 0:
        print(f"   Mean correlation (r > 5 kpc): {corr_data[r_data > 5].mean():.4f}")
    
    if fit_params is not None:
        print(f"\n📐 Best Fit Parameters:")
        print(f"   Amplitude A = {fit_params[0]:.3f}")
        print(f"   Coherence length ℓ₀ = {fit_params[1]:.2f} kpc")
        print(f"   Exponent n_coh = {fit_params[2]:.3f}")
        
        if 2 < fit_params[1] < 10:
            print(f"\n   ✅ ℓ₀ in expected range (2-10 kpc)!")
        else:
            print(f"\n   ⚠️  ℓ₀ differs from theoretical prediction (5 kpc)")
    
    # Save results
    results_df = pd.DataFrame({
        'r_kpc': r_data,
        'correlation': corr_data,
        'correlation_err': corr_err,
        'n_pairs': n_pairs
    })
    results_path = output_dir / "acceleration_correlation_results.csv"
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
        'R_grid': R_grid,
        'a_jeans': a_jeans,
        'a_newton': a_newton
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gaia DR3 Acceleration Correlation Test for Σ-Gravity"
    )
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--subsample-n', type=int, default=None)
    parser.add_argument('--max-sep', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=2000)
    parser.add_argument('--no-gpu', action='store_true')
    
    args = parser.parse_args()
    
    run_acceleration_test(
        data_file=args.data_file,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        subsample_n=args.subsample_n,
        max_sep=args.max_sep,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
