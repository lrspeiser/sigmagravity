#!/usr/bin/env python3
"""
Gaia DR3 Velocity Correlation Analysis for Σ-Gravity Testing
=============================================================

Tests the prediction: ξ_v(Δr) ∝ (ℓ₀ / (ℓ₀ + Δr))^n_coh
with ℓ₀ ≈ 5 kpc and n_coh ≈ 0.5 from SPARC calibration.

Usage:
    python gaia_analysis.py              # Quick synthetic test
    python gaia_analysis.py --gaia       # Real Gaia DR3 data (requires astroquery)
    python gaia_analysis.py --n 200000   # Specify number of stars

Requirements:
    pip install numpy scipy matplotlib astropy astroquery
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import argparse
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SigmaGravityParams:
    """Σ-Gravity theoretical parameters from SPARC"""
    ell_0: float = 4.993    # Coherence length [kpc]
    n_coh: float = 0.5      # Power-law exponent
    sigma_v: float = 35.0   # Velocity dispersion [km/s]

@dataclass
class GalactocentricParams:
    """Galactocentric coordinate system"""
    R_sun: float = 8.122    # Solar radius [kpc]
    z_sun: float = 0.0208   # Solar height [kpc]
    v_sun: tuple = (11.1, 245.8, 7.8)  # Solar motion [km/s]

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def model_standard(r, A0, ell):
    """Standard exponential: ξ = A₀ exp(-r/ℓ)"""
    return A0 * np.exp(-r / ell)

def model_sigma_gravity(r, A0, ell_0, n_coh):
    """Σ-Gravity power-law: ξ = A₀ (ℓ₀/(ℓ₀+r))^n"""
    return A0 * (ell_0 / (ell_0 + r)) ** n_coh

def model_sigma_fixed(r, A0, ell_0):
    """Σ-Gravity with n_coh=0.5 fixed"""
    return A0 * (ell_0 / (ell_0 + r)) ** 0.5

# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(n_stars=10000, ell_0=5.0, n_coh=0.5, 
                           sigma_v=35.0, seed=42):
    """
    Generate synthetic stellar data with Σ-Gravity-like correlations.
    
    Uses Fourier method to generate velocity field with power spectrum:
        P_v(k) ∝ 1/(1 + (k*ell_0)^2)^(n_coh + 1)
    
    This produces correlation function: ξ_v(r) ∝ (ell_0/(ell_0+r))^n_coh
    """
    np.random.seed(seed)
    
    # Disk annulus (4-12 kpc)
    R = np.sqrt(np.random.uniform(4**2, 12**2, n_stars))
    phi = np.random.uniform(0, 2*np.pi, n_stars)
    z = np.random.normal(0, 0.3, n_stars)
    
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    
    # Create velocity field on grid using Fourier method
    L = 30.0  # Domain size [kpc]
    grid_size = 256  # Higher resolution for better correlations
    
    # Build power spectrum in Fourier space
    kx = np.fft.fftfreq(grid_size, d=L/grid_size) * 2 * np.pi
    ky = np.fft.fftfreq(grid_size, d=L/grid_size) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1e-10  # Avoid division by zero
    
    # Power spectrum for Σ-Gravity: P(k) ∝ 1/(1 + (k*ell_0)^2)^(n_coh + 1)
    # This is the Fourier transform of the Matérn-like covariance
    P_v = 1.0 / (1.0 + (K * ell_0)**2)**(n_coh + 1)
    P_v[0, 0] = 0  # Zero mean
    
    # Generate random Fourier coefficients with correct power
    amplitude = np.sqrt(P_v)
    phase = np.random.uniform(0, 2*np.pi, (grid_size, grid_size))
    v_k = amplitude * np.exp(1j * phase)
    
    # Ensure real output via Hermitian symmetry
    v_k = (v_k + np.conj(v_k[::-1, ::-1])) / 2
    
    # Transform to real space
    v_field = np.real(np.fft.ifft2(v_k))
    
    # Normalize to desired variance
    v_field = v_field / v_field.std() * sigma_v * 0.8
    
    # Grid coordinates
    x_grid = np.linspace(-L/2, L/2, grid_size)
    y_grid = np.linspace(-L/2, L/2, grid_size)
    
    # Interpolate to star positions
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((x_grid, y_grid), v_field.T,
                                     bounds_error=False, fill_value=0)
    delta_v = interp(np.column_stack([x, y]))
    
    # Add measurement noise (smaller fraction)
    delta_v += np.random.normal(0, sigma_v * 0.2, n_stars)
    
    return {
        'x': x, 'y': y, 'z': z, 'R': R, 'phi': phi,
        'delta_v_phi': delta_v,
        'n_stars': n_stars
    }

# =============================================================================
# GAIA DATA RETRIEVAL
# =============================================================================

def query_gaia(n_stars=100000, b_max=25.0, cache_file='gaia_cache.npz'):
    """
    Query Gaia DR3 for disk stars with radial velocities.
    
    Parameters
    ----------
    n_stars : int
        Maximum number of stars to retrieve
    b_max : float
        Maximum absolute galactic latitude [deg]
    cache_file : str
        File to cache downloaded data
        
    Returns
    -------
    dict : Galactocentric coordinates and velocity residuals
    """
    if os.path.exists(cache_file):
        print(f"Loading cached Gaia data from {cache_file}")
        data = np.load(cache_file)
        return dict(data)
    
    try:
        from astroquery.gaia import Gaia
        import astropy.units as u
        from astropy.coordinates import SkyCoord, Galactocentric
        import astropy.coordinates as coord
    except ImportError:
        raise ImportError(
            "Install astropy and astroquery:\n"
            "  pip install astropy astroquery"
        )
    
    query = f"""
    SELECT TOP {n_stars}
        source_id, ra, dec, l, b,
        parallax, parallax_error,
        pmra, pmdec, 
        radial_velocity, radial_velocity_error
    FROM gaiadr3.gaia_source
    WHERE parallax_over_error > 5
        AND radial_velocity IS NOT NULL
        AND ABS(b) < {b_max}
        AND ruwe < 1.4
        AND radial_velocity_error < 5
        AND parallax > 0.1
    """
    
    print(f"Querying Gaia DR3 for {n_stars} stars...")
    print("This may take several minutes...")
    
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    
    print(f"Retrieved {len(results)} stars")
    
    # Convert to arrays
    ra = np.array(results['ra'])
    dec = np.array(results['dec'])
    parallax = np.array(results['parallax'])
    pmra = np.array(results['pmra'])
    pmdec = np.array(results['pmdec'])
    rv = np.array(results['radial_velocity'])
    
    # Transform to Galactocentric
    gc_params = GalactocentricParams()
    
    print("Transforming to Galactocentric coordinates...")
    
    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=(1.0 / parallax) * u.kpc,
        pm_ra_cosdec=pmra * u.mas/u.yr,
        pm_dec=pmdec * u.mas/u.yr,
        radial_velocity=rv * u.km/u.s,
        frame='icrs'
    )
    
    gc_frame = Galactocentric(
        galcen_distance=gc_params.R_sun * u.kpc,
        z_sun=gc_params.z_sun * u.kpc,
        galcen_v_sun=coord.CartesianDifferential(
            list(gc_params.v_sun) * u.km/u.s
        )
    )
    
    gc = coords.transform_to(gc_frame)
    
    # Extract coordinates
    x = gc.cartesian.x.to(u.kpc).value
    y = gc.cartesian.y.to(u.kpc).value
    z_gc = gc.cartesian.z.to(u.kpc).value
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    
    # Extract velocities
    vx = gc.velocity.d_x.to(u.km/u.s).value
    vy = gc.velocity.d_y.to(u.km/u.s).value
    vz = gc.velocity.d_z.to(u.km/u.s).value
    
    # Convert to cylindrical
    v_R = (x * vx + y * vy) / R
    v_phi = (x * vy - y * vx) / R
    
    # Compute velocity residuals (subtract mean rotation curve)
    print("Computing velocity residuals...")
    R_bins = np.linspace(3, 15, 40)
    v_phi_median = []
    R_centers = []
    
    for i in range(len(R_bins) - 1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            v_phi_median.append(np.median(v_phi[mask]))
            R_centers.append(0.5 * (R_bins[i] + R_bins[i+1]))
    
    v_phi_median = np.array(v_phi_median)
    R_centers = np.array(R_centers)
    
    # Interpolate expected rotation
    v_phi_expected = np.interp(R, R_centers, v_phi_median)
    delta_v_phi = v_phi - v_phi_expected
    
    # Similarly for v_R
    v_R_median = []
    for i in range(len(R_bins) - 1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            v_R_median.append(np.median(v_R[mask]))
    v_R_median = np.array(v_R_median)
    v_R_expected = np.interp(R, R_centers, v_R_median)
    delta_v_R = v_R - v_R_expected
    
    data = {
        'x': x, 'y': y, 'z': z_gc, 'R': R, 'phi': phi,
        'v_phi': v_phi, 'v_R': v_R, 'v_z': vz,
        'delta_v_phi': delta_v_phi,
        'delta_v_R': delta_v_R,
        'n_stars': len(x)
    }
    
    # Cache the data
    np.savez(cache_file, **data)
    print(f"Cached {len(x)} stars to {cache_file}")
    
    return data

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlation(data, sep_bins=None, max_sample=10000, 
                       velocity_component='phi'):
    """
    Compute velocity correlation function from stellar data.
    
    Parameters
    ----------
    data : dict
        Dictionary with positions and velocity residuals
    sep_bins : array
        Separation bin edges [kpc]
    max_sample : int
        Maximum stars to use (subsamples if needed)
    velocity_component : str
        'phi' for azimuthal, 'R' for radial
        
    Returns
    -------
    dict : Correlation function results
    """
    x = data['x']
    y = data['y']
    z = data['z']
    R = data['R']
    
    if velocity_component == 'phi':
        dv = data['delta_v_phi']
    elif velocity_component == 'R':
        dv = data.get('delta_v_R', data['delta_v_phi'])
    else:
        dv = data['delta_v_phi']
    
    # Quality cuts
    mask = (R > 4) & (R < 12) & (np.abs(z) < 1) & np.isfinite(dv)
    x, y, z, R, dv = x[mask], y[mask], z[mask], R[mask], dv[mask]
    
    n = len(x)
    print(f"Computing correlations for {n} stars after quality cuts")
    
    # Subsample for computational efficiency
    if n > max_sample:
        idx = np.random.choice(n, max_sample, replace=False)
        x, y, z, R, dv = x[idx], y[idx], z[idx], R[idx], dv[idx]
        n = max_sample
        print(f"Subsampled to {n} stars for efficiency")
    
    # Default separation bins (log-spaced from 100 pc to 8 kpc)
    if sep_bins is None:
        sep_bins = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 
                            2.0, 3.0, 4.0, 5.0, 6.5, 8.0])
    
    n_bins = len(sep_bins) - 1
    xi_sum = np.zeros(n_bins)
    xi_sq_sum = np.zeros(n_bins)
    n_pairs = np.zeros(n_bins, dtype=int)
    
    # Compute pairwise correlations
    print("Computing pair correlations...")
    for i in range(n):
        if i % 1000 == 0 and i > 0:
            print(f"  Progress: {i}/{n} ({100*i/n:.0f}%)")
        
        # Separations from star i to all stars j > i
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dz_arr = z[i] - z[i+1:]
        sep = np.sqrt(dx**2 + dy**2 + dz_arr**2)
        
        # Velocity products
        dv_prod = dv[i] * dv[i+1:]
        
        # Bin by separation
        for b in range(n_bins):
            mask = (sep >= sep_bins[b]) & (sep < sep_bins[b+1])
            xi_sum[b] += np.sum(dv_prod[mask])
            xi_sq_sum[b] += np.sum(dv_prod[mask]**2)
            n_pairs[b] += np.sum(mask)
    
    # Compute statistics
    sep_centers = np.sqrt(sep_bins[:-1] * sep_bins[1:])  # Geometric mean
    valid = n_pairs > 50
    
    xi_mean = np.where(valid, xi_sum / n_pairs, np.nan)
    xi_var = np.where(valid, xi_sq_sum / n_pairs - xi_mean**2, np.nan)
    xi_err = np.where(valid, np.sqrt(np.abs(xi_var) / n_pairs), np.nan)
    
    print(f"\nCorrelation computed at {np.sum(valid)} valid bins")
    print(f"Total pairs analyzed: {n_pairs.sum():,}")
    
    return {
        'separation': sep_centers,
        'xi_v': xi_mean,
        'xi_v_err': xi_err,
        'n_pairs': n_pairs,
        'valid': valid
    }

# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_models(corr):
    """
    Fit standard and Σ-Gravity models to correlation data.
    
    Parameters
    ----------
    corr : dict
        Output from compute_correlation()
        
    Returns
    -------
    dict : Fit results for each model
    """
    
    mask = corr['valid'] & (corr['xi_v'] > 0)
    r = corr['separation'][mask]
    xi = corr['xi_v'][mask]
    err = np.maximum(corr['xi_v_err'][mask], 1.0)  # Minimum error floor
    
    if len(r) < 4:
        print("Warning: Too few valid data points for fitting")
        return {'r': r, 'xi': xi, 'err': err}
    
    results = {'r': r, 'xi': xi, 'err': err}
    
    # --- Model A: Standard exponential ---
    try:
        popt, pcov = curve_fit(
            model_standard, r, xi, 
            p0=[xi[0], 1.0],
            sigma=err, 
            bounds=([0, 0.05], [1e5, 20]),
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov))
        chi2 = np.sum(((xi - model_standard(r, *popt)) / err)**2)
        bic = chi2 + 2 * np.log(len(r))
        
        results['standard'] = {
            'A0': popt[0], 'A0_err': perr[0],
            'ell': popt[1], 'ell_err': perr[1],
            'chi2': chi2, 'bic': bic, 'dof': len(r) - 2
        }
    except Exception as e:
        print(f"Standard model fit failed: {e}")
        results['standard'] = None
    
    # --- Model B: Σ-Gravity with free n_coh ---
    try:
        popt, pcov = curve_fit(
            model_sigma_gravity, r, xi, 
            p0=[xi[0], 5.0, 0.5],
            sigma=err, 
            bounds=([0, 0.1, 0.1], [1e5, 20, 2.0]),
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov))
        chi2 = np.sum(((xi - model_sigma_gravity(r, *popt)) / err)**2)
        bic = chi2 + 3 * np.log(len(r))
        
        results['sigma_free'] = {
            'A0': popt[0], 'A0_err': perr[0],
            'ell_0': popt[1], 'ell_0_err': perr[1],
            'n_coh': popt[2], 'n_coh_err': perr[2],
            'chi2': chi2, 'bic': bic, 'dof': len(r) - 3
        }
    except Exception as e:
        print(f"Σ-Gravity (free) fit failed: {e}")
        results['sigma_free'] = None
    
    # --- Model B': Σ-Gravity with n_coh=0.5 fixed ---
    try:
        popt, pcov = curve_fit(
            model_sigma_fixed, r, xi,
            p0=[xi[0], 5.0],
            sigma=err, 
            bounds=([0, 0.1], [1e5, 20]),
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov))
        chi2 = np.sum(((xi - model_sigma_fixed(r, *popt)) / err)**2)
        bic = chi2 + 2 * np.log(len(r))
        
        results['sigma_fixed'] = {
            'A0': popt[0], 'A0_err': perr[0],
            'ell_0': popt[1], 'ell_0_err': perr[1],
            'n_coh': 0.5,
            'chi2': chi2, 'bic': bic, 'dof': len(r) - 2
        }
    except Exception as e:
        print(f"Σ-Gravity (fixed) fit failed: {e}")
        results['sigma_fixed'] = None
    
    return results

# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(corr, fits, output_path='correlation_results.png'):
    """
    Create diagnostic plots of correlation function and fits.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    r = corr['separation']
    xi = corr['xi_v']
    err = corr['xi_v_err']
    valid = corr['valid']
    
    r_model = np.linspace(0.1, 8, 100)
    
    # ===================
    # Left: Linear scale
    # ===================
    ax = axes[0]
    ax.errorbar(r[valid], xi[valid], yerr=err[valid], fmt='ko', 
               capsize=3, markersize=6, label='Gaia data')
    
    if fits.get('standard'):
        f = fits['standard']
        ax.plot(r_model, model_standard(r_model, f['A0'], f['ell']),
               'r--', lw=2, label=f"Standard: ℓ={f['ell']:.2f} kpc")
    
    if fits.get('sigma_fixed'):
        f = fits['sigma_fixed']
        ax.plot(r_model, model_sigma_fixed(r_model, f['A0'], f['ell_0']),
               'b-', lw=2, label=f"Σ-Gravity: ℓ₀={f['ell_0']:.2f} kpc")
    
    ax.set_xlabel('Separation Δr [kpc]', fontsize=12)
    ax.set_ylabel('ξ_v [km²/s²]', fontsize=12)
    ax.set_title('Velocity Correlation Function', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    
    # ===================
    # Right: Log-log scale
    # ===================
    ax = axes[1]
    pos = valid & (xi > 0)
    ax.errorbar(r[pos], xi[pos], yerr=err[pos], fmt='ko', 
               capsize=3, markersize=6)
    
    if fits.get('standard'):
        f = fits['standard']
        ax.plot(r_model, model_standard(r_model, f['A0'], f['ell']),
               'r--', lw=2, label='Standard (exponential)')
    
    if fits.get('sigma_fixed'):
        f = fits['sigma_fixed']
        ax.plot(r_model, model_sigma_fixed(r_model, f['A0'], f['ell_0']),
               'b-', lw=2, label='Σ-Gravity (power-law)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Separation Δr [kpc]', fontsize=12)
    ax.set_ylabel('ξ_v [km²/s²]', fontsize=12)
    ax.set_title('Log-Log Scale (Power Law Test)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add slope = -0.5 guide line
    if np.any(pos):
        y_ref = xi[pos][0]
        x_ref = r[pos][0]
        ax.plot([x_ref, 4*x_ref], [y_ref, y_ref * (x_ref/(4*x_ref))**0.5], 
               'k--', alpha=0.4, lw=1)
        ax.text(2*x_ref, 0.7*y_ref, 'slope = -0.5', fontsize=9, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return fig

def print_results(fits):
    """Print formatted fit results and model comparison."""
    
    print("\n" + "="*60)
    print("FIT RESULTS")
    print("="*60)
    
    if fits.get('standard'):
        f = fits['standard']
        print(f"\nStandard Model (exponential decay):")
        print(f"  A₀ = {f['A0']:.1f} ± {f['A0_err']:.1f} km²/s²")
        print(f"  ℓ  = {f['ell']:.3f} ± {f['ell_err']:.3f} kpc")
        print(f"  χ²/dof = {f['chi2']/f['dof']:.2f}")
        print(f"  BIC = {f['bic']:.1f}")
    
    if fits.get('sigma_free'):
        f = fits['sigma_free']
        print(f"\nΣ-Gravity Model (free n_coh):")
        print(f"  A₀    = {f['A0']:.1f} ± {f['A0_err']:.1f} km²/s²")
        print(f"  ℓ₀    = {f['ell_0']:.3f} ± {f['ell_0_err']:.3f} kpc  [theory: 5.0 kpc]")
        print(f"  n_coh = {f['n_coh']:.3f} ± {f['n_coh_err']:.3f}  [theory: 0.5]")
        print(f"  χ²/dof = {f['chi2']/f['dof']:.2f}")
        print(f"  BIC = {f['bic']:.1f}")
    
    if fits.get('sigma_fixed'):
        f = fits['sigma_fixed']
        print(f"\nΣ-Gravity Model (n_coh=0.5 fixed):")
        print(f"  A₀ = {f['A0']:.1f} ± {f['A0_err']:.1f} km²/s²")
        print(f"  ℓ₀ = {f['ell_0']:.3f} ± {f['ell_0_err']:.3f} kpc  [theory: 5.0 kpc]")
        print(f"  χ²/dof = {f['chi2']/f['dof']:.2f}")
        print(f"  BIC = {f['bic']:.1f}")
    
    # Model comparison
    if fits.get('standard') and fits.get('sigma_fixed'):
        delta_bic = fits['standard']['bic'] - fits['sigma_fixed']['bic']
        
        print("\n" + "-"*60)
        print("MODEL COMPARISON")
        print("-"*60)
        print(f"\nΔBIC (Standard - Σ-Gravity) = {delta_bic:.1f}")
        
        if delta_bic > 10:
            verdict = "VERY STRONG evidence for Σ-Gravity"
            symbol = "★★★"
        elif delta_bic > 6:
            verdict = "STRONG evidence for Σ-Gravity"
            symbol = "★★"
        elif delta_bic > 2:
            verdict = "Positive evidence for Σ-Gravity"
            symbol = "★"
        elif delta_bic > -2:
            verdict = "Inconclusive"
            symbol = "?"
        else:
            verdict = "Standard model preferred"
            symbol = "✗"
        
        print(f"\n{symbol} {verdict} {symbol}")
        
        # Parameter consistency check
        if fits.get('sigma_free'):
            f = fits['sigma_free']
            ell_0_match = 3.0 < f['ell_0'] < 8.0
            n_coh_match = 0.3 < f['n_coh'] < 0.8
            
            print(f"\nParameter consistency with SPARC:")
            print(f"  ℓ₀ in [3, 8] kpc: {'✓' if ell_0_match else '✗'} ({f['ell_0']:.2f} kpc)")
            print(f"  n_coh in [0.3, 0.8]: {'✓' if n_coh_match else '✗'} ({f['n_coh']:.2f})")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gaia DR3 Velocity Correlation Analysis for Σ-Gravity Testing'
    )
    parser.add_argument('--gaia', action='store_true', 
                       help='Use real Gaia DR3 data (requires astroquery)')
    parser.add_argument('--n', type=int, default=10000, 
                       help='Number of stars to analyze')
    parser.add_argument('--output', type=str, default='correlation_results.png',
                       help='Output plot filename')
    parser.add_argument('--cache', type=str, default='gaia_cache.npz',
                       help='Gaia data cache file')
    args = parser.parse_args()
    
    print("="*60)
    print("Σ-GRAVITY VELOCITY CORRELATION ANALYSIS")
    print("="*60)
    print("\nTesting prediction: ξ_v(Δr) ∝ (ℓ₀/(ℓ₀+Δr))^0.5")
    print("with ℓ₀ ≈ 5 kpc from SPARC calibration\n")
    
    # =====================
    # Step 1: Get data
    # =====================
    if args.gaia:
        print("DATA SOURCE: Gaia DR3")
        print("-" * 40)
        data = query_gaia(n_stars=args.n, cache_file=args.cache)
    else:
        print("DATA SOURCE: Synthetic (for testing)")
        print("Run with --gaia for real Gaia DR3 data")
        print("-" * 40)
        data = generate_synthetic_data(n_stars=args.n)
    
    print(f"Total stars available: {data['n_stars']}")
    
    # =====================
    # Step 2: Compute correlation
    # =====================
    print("\n" + "-"*40)
    print("COMPUTING CORRELATION FUNCTION")
    print("-"*40)
    corr = compute_correlation(data)
    
    # Print correlation values
    print("\nMeasured correlation function:")
    print("  Δr [kpc]    ξ_v [km²/s²]    N_pairs")
    for i in range(len(corr['separation'])):
        if corr['valid'][i]:
            print(f"  {corr['separation'][i]:6.2f}      {corr['xi_v'][i]:8.1f}        {corr['n_pairs'][i]:6d}")
    
    # =====================
    # Step 3: Fit models
    # =====================
    print("\n" + "-"*40)
    print("FITTING MODELS")
    print("-"*40)
    fits = fit_models(corr)
    
    # =====================
    # Step 4: Print results
    # =====================
    print_results(fits)
    
    # =====================
    # Step 5: Create plots
    # =====================
    print("\n" + "-"*40)
    print("CREATING PLOTS")
    print("-"*40)
    plot_results(corr, fits, args.output)
    
    # =====================
    # Summary
    # =====================
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput: {args.output}")
    
    if not args.gaia:
        print("\nTo analyze real Gaia DR3 data:")
        print("  python gaia_analysis.py --gaia --n 200000")
    
    return corr, fits

if __name__ == '__main__':
    main()
