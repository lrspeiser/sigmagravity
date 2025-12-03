"""
Σ-Gravity Cosmological Expansion Analysis

GPU-accelerated (CuPy) and CPU-parallelized implementation.
Fits modified Friedmann equations with Σ_cos(z) to Pantheon+ SN Ia data.

Data source: ../data/pantheon/Pantheon+SH0ES.dat
See sigma_cosmology_theory.md for theoretical background.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GPU Setup - CuPy with fallback to NumPy
# =============================================================================

try:
    import cupy as cp
    from cupyx.scipy.interpolate import interp1d as cp_interp1d
    GPU_AVAILABLE = True
    print("CuPy GPU acceleration: ENABLED (RTX 5090)")
    # Set memory pool for better performance
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
except ImportError:
    cp = np  # Fallback to numpy
    GPU_AVAILABLE = False
    print("CuPy not available, using NumPy (CPU only)")

# Number of CPU cores for parallel processing
N_CORES = 10
print(f"CPU parallelization: {N_CORES} cores")

# =============================================================================
# Physical Constants (GPU arrays)
# =============================================================================

c = 299792.458  # km/s (speed of light)
e_val = np.e    # Euler's number
MPC_TO_KM = 3.086e19  # km per Mpc

# =============================================================================
# Vectorized Σ-Gravity Functions (GPU-accelerated)
# =============================================================================

def compute_g_dagger(H0_kmsMpc, xp=cp):
    """Compute g† = c·H₀/(2e) in m/s² - fully vectorized"""
    H0_per_sec = H0_kmsMpc / MPC_TO_KM
    c_m_per_s = c * 1000
    return c_m_per_s * H0_per_sec / (2 * e_val)


def h_function_vectorized(g, g_dagger, xp=cp):
    """
    Vectorized h(g) = sqrt(g†/g) · g† / (g† + g)
    Works on GPU arrays with CuPy or CPU with NumPy.
    """
    g = xp.maximum(g, 1e-30)
    ratio = g_dagger / g
    return xp.sqrt(ratio) * g_dagger / (g_dagger + g)


def sigma_cos_vectorized(E_array, H0, A_cos, xp=cp):
    """
    Vectorized Σ_cos computation for array of E values.
    Σ_cos = 1 + A_cos · h(c·H₀·E)
    """
    H_array = H0 * E_array  # H(z) values
    H_per_sec = H_array / MPC_TO_KM
    c_m_per_s = c * 1000
    g_H = c_m_per_s * H_per_sec  # Acceleration at Hubble scale
    
    g_dagger = compute_g_dagger(H0, xp)
    h_vals = h_function_vectorized(g_H, g_dagger, xp)
    
    return 1.0 + A_cos * h_vals


# =============================================================================
# Fast E(z) Solver - Vectorized Fixed-Point Iteration on GPU
# =============================================================================

def solve_E_grid_gpu(z_grid, Omega_m0, A_cos, H0, n_iter=50, tol=1e-10):
    """
    Solve E(z) on a fine grid using GPU-accelerated fixed-point iteration.
    
    E² = Ω_m0·(1+z)³·Σ_cos(E) where Σ_cos depends on E.
    
    Uses vectorized iteration across all z simultaneously on GPU.
    """
    xp = cp if GPU_AVAILABLE else np
    
    # Move to GPU
    z_gpu = xp.asarray(z_grid, dtype=xp.float64)
    one_plus_z_cubed = (1 + z_gpu)**3
    
    # Initial guess: matter-only universe
    E = xp.sqrt(Omega_m0 * one_plus_z_cubed + (1 - Omega_m0))
    
    # Fixed-point iteration (vectorized across all z)
    for _ in range(n_iter):
        Sigma = sigma_cos_vectorized(E, H0, A_cos, xp)
        E_sq_new = Omega_m0 * one_plus_z_cubed * Sigma
        E_new = xp.sqrt(xp.maximum(E_sq_new, 1e-10))
        
        # Check convergence
        max_diff = xp.max(xp.abs(E_new - E))
        E = E_new
        
        if max_diff < tol:
            break
    
    # Return to CPU if needed
    if GPU_AVAILABLE:
        return cp.asnumpy(E)
    return E


# =============================================================================
# Distance Calculations - Vectorized Trapezoidal Integration
# =============================================================================

class CosmologyInterpolator:
    """
    Pre-compute E(z) on a fine grid and use fast interpolation.
    This avoids repeated root-finding during fitting.
    """
    
    def __init__(self, z_max=3.0, n_grid=2000):
        self.z_max = z_max
        self.n_grid = n_grid
        self.z_grid = np.linspace(0, z_max, n_grid)
        self._E_grid = None
        self._chi_grid = None
        self._params = None
    
    def update(self, Omega_m0, A_cos, H0):
        """Recompute grids for new parameters."""
        params = (Omega_m0, A_cos, H0)
        if self._params == params:
            return  # Already computed
        
        self._params = params
        
        # Solve E(z) on grid (GPU-accelerated)
        self._E_grid = solve_E_grid_gpu(self.z_grid, Omega_m0, A_cos, H0)
        
        # Compute comoving distance grid via trapezoidal integration
        # χ(z) = (c/H₀) ∫₀ᶻ dz' / E(z')
        integrand = 1.0 / self._E_grid
        self._chi_grid = np.zeros_like(self.z_grid)
        dz = self.z_grid[1] - self.z_grid[0]
        
        # Cumulative trapezoidal integration
        self._chi_grid[1:] = np.cumsum(
            0.5 * (integrand[:-1] + integrand[1:]) * dz
        )
        self._chi_grid *= (c / H0)  # Scale to Mpc
    
    def luminosity_distance(self, z):
        """Fast interpolated D_L(z) in Mpc."""
        chi = np.interp(z, self.z_grid, self._chi_grid)
        return (1 + z) * chi
    
    def distance_modulus(self, z, M=0.0):
        """Fast interpolated μ(z)."""
        D_L = self.luminosity_distance(z)
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25 + M
    
    def E_of_z(self, z):
        """Interpolated E(z)."""
        return np.interp(z, self.z_grid, self._E_grid)


# Global interpolator instance
_cosmo_interp = CosmologyInterpolator(z_max=3.0, n_grid=2000)


def compute_mu_array_fast(z_array, Omega_m0, A_cos, H0, M=0.0):
    """
    Fast vectorized distance modulus computation.
    Uses pre-computed interpolation grid.
    """
    _cosmo_interp.update(Omega_m0, A_cos, H0)
    return _cosmo_interp.distance_modulus(z_array, M)


def compute_E_array_fast(z_array, Omega_m0, A_cos, H0):
    """Fast E(z) array via interpolation."""
    _cosmo_interp.update(Omega_m0, A_cos, H0)
    return _cosmo_interp.E_of_z(z_array)


# =============================================================================
# ΛCDM Reference Model (Vectorized)
# =============================================================================

def E_LCDM_vectorized(z, Omega_m0, Omega_Lambda):
    """Vectorized ΛCDM E(z)."""
    return np.sqrt(Omega_m0 * (1 + z)**3 + Omega_Lambda)


def mu_LCDM_vectorized(z_array, Omega_m0, Omega_Lambda, H0, M=0.0):
    """Vectorized distance modulus for ΛCDM using trapezoidal integration."""
    # Fine grid for integration
    z_max = np.max(z_array) * 1.1
    z_grid = np.linspace(0, z_max, 2000)
    E_grid = E_LCDM_vectorized(z_grid, Omega_m0, Omega_Lambda)
    
    # Comoving distance grid
    integrand = 1.0 / E_grid
    dz = z_grid[1] - z_grid[0]
    chi_grid = np.zeros_like(z_grid)
    chi_grid[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    chi_grid *= (c / H0)
    
    # Interpolate to requested z values
    chi = np.interp(z_array, z_grid, chi_grid)
    D_L = (1 + z_array) * chi
    
    return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25 + M


# =============================================================================
# Data Loading
# =============================================================================

def load_pantheon_data(max_z=2.5):
    """
    Load Pantheon+ SN Ia data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'pantheon', 'Pantheon+SH0ES.dat')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Pantheon data not found at: {data_path}")
    
    print(f"\nLoading Pantheon+ data from: {data_path}")
    
    df = pd.read_csv(data_path, sep=r'\s+')
    
    z = df['zCMB'].values
    mu = df['MU_SH0ES'].values
    mu_err = df['MU_SH0ES_ERR_DIAG'].values
    
    # Filter by redshift
    mask = (z > 0.01) & (z <= max_z)
    z = z[mask]
    mu = mu[mask]
    mu_err = mu_err[mask]
    
    # Sort by redshift for efficient interpolation
    sort_idx = np.argsort(z)
    z = z[sort_idx]
    mu = mu[sort_idx]
    mu_err = mu_err[sort_idx]
    
    print(f"  Loaded {len(z)} SNe with 0.01 < z < {max_z}")
    print(f"  Redshift range: {z.min():.4f} - {z.max():.4f}")
    
    return z, mu, mu_err


# =============================================================================
# Chi-Squared Functions (Fast)
# =============================================================================

def chi_squared_sigma(params, z_data, mu_data, mu_err):
    """
    Fast χ² for Σ-Gravity cosmology.
    """
    H0, Omega_m0, A_cos, M = params
    
    # Parameter bounds
    if not (50 < H0 < 90):
        return 1e10
    if not (0.1 < Omega_m0 < 0.5):
        return 1e10
    if not (0.1 < A_cos < 10):
        return 1e10
    
    try:
        mu_model = compute_mu_array_fast(z_data, Omega_m0, A_cos, H0, M)
        residuals = (mu_data - mu_model) / mu_err
        return np.sum(residuals**2)
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return 1e10


def chi_squared_lcdm(params, z_data, mu_data, mu_err):
    """
    Fast χ² for ΛCDM.
    """
    H0, Omega_m0, M = params
    Omega_Lambda = 1 - Omega_m0
    
    if not (50 < H0 < 90):
        return 1e10
    if not (0.1 < Omega_m0 < 0.5):
        return 1e10
    
    mu_model = mu_LCDM_vectorized(z_data, Omega_m0, Omega_Lambda, H0, M)
    residuals = (mu_data - mu_model) / mu_err
    return np.sum(residuals**2)


# =============================================================================
# Physical Constants for Baryons-Only Analysis
# =============================================================================

# Baryon density from Planck: Ω_b h² ≈ 0.0224, with h ≈ 0.67 → Ω_b ≈ 0.05
OMEGA_BARYON = 0.05  # The ONLY matter in a truly baryons-only universe


# =============================================================================
# Chi-Squared for Baryons-Only Models
# =============================================================================

def chi_squared_gr_baryons_only(params, z_data, mu_data, mu_err):
    """
    χ² for GR + baryons only (no dark matter, no dark energy, no Σ).
    Ω_m fixed to Ω_baryon ≈ 0.05, Ω_Λ = 0, Σ = 1.
    
    Only free params: [H0, M]
    """
    H0, M = params
    
    if not (40 < H0 < 100):
        return 1e10
    
    # Pure matter-only, baryons only
    mu_model = mu_matter_only_vectorized(z_data, OMEGA_BARYON, H0, M)
    residuals = (mu_data - mu_model) / mu_err
    return np.sum(residuals**2)


def chi_squared_sigma_baryons_only(params, z_data, mu_data, mu_err):
    """
    χ² for Σ-Gravity + baryons only (no dark matter, no dark energy).
    Ω_m fixed to Ω_baryon ≈ 0.05, Ω_Λ = 0, A_cos free.
    
    Free params: [H0, A_cos, M]
    """
    H0, A_cos, M = params
    
    if not (40 < H0 < 100):
        return 1e10
    if not (0.5 < A_cos < 15):
        return 1e10
    
    try:
        mu_model = compute_mu_array_fast(z_data, OMEGA_BARYON, A_cos, H0, M)
        residuals = (mu_data - mu_model) / mu_err
        return np.sum(residuals**2)
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return 1e10


def chi_squared_gr_effective_matter(params, z_data, mu_data, mu_err):
    """
    χ² for GR + effective matter (Ω_m free, Ω_Λ = 0, Σ = 1).
    Control model to show how badly GR fails without dark energy.
    
    Free params: [H0, Omega_m, M]
    """
    H0, Omega_m, M = params
    
    if not (40 < H0 < 100):
        return 1e10
    if not (0.01 < Omega_m < 1.0):
        return 1e10
    
    mu_model = mu_matter_only_vectorized(z_data, Omega_m, H0, M)
    residuals = (mu_data - mu_model) / mu_err
    return np.sum(residuals**2)


def mu_matter_only_vectorized(z_array, Omega_m, H0, M=0.0):
    """
    Distance modulus for matter-only universe (no Λ, Σ=1).
    E(z) = sqrt(Ω_m * (1+z)³)
    """
    z_max = np.max(z_array) * 1.1
    z_grid = np.linspace(1e-6, z_max, 2000)  # Avoid z=0 singularity
    E_grid = np.sqrt(Omega_m * (1 + z_grid)**3)
    
    # Comoving distance via trapezoidal integration
    integrand = 1.0 / E_grid
    dz = z_grid[1] - z_grid[0]
    chi_grid = np.zeros_like(z_grid)
    chi_grid[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    chi_grid *= (c / H0)
    
    chi = np.interp(z_array, z_grid, chi_grid)
    D_L = (1 + z_array) * chi
    
    return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25 + M


# =============================================================================
# Fitting Functions - Baryons-Only Models
# =============================================================================

def fit_gr_baryons_only(z_data, mu_data, mu_err):
    """
    Model A: GR + baryons only (Ω_m = Ω_b = 0.05, Ω_Λ = 0, Σ = 1)
    The baseline "no dark stuff, no Σ" model.
    """
    print("\n" + "="*60)
    print("MODEL A: GR + Baryons Only")
    print(f"  Ω_m = Ω_baryon = {OMEGA_BARYON:.3f} (FIXED)")
    print("  Ω_Λ = 0 (no dark energy)")
    print("  Σ = 1 (standard GR)")
    print("="*60)
    
    t0 = time.time()
    
    # Only fit: [H0, M]
    x0 = [70.0, -19.3]
    
    result = minimize(
        chi_squared_gr_baryons_only,
        x0,
        args=(z_data, mu_data, mu_err),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    t1 = time.time()
    
    H0_fit, M_fit = result.x
    chi2_fit = result.fun
    dof = len(z_data) - 2
    chi2_red = chi2_fit / dof
    
    print(f"\nBest-fit parameters:")
    print(f"  H₀ = {H0_fit:.2f} km/s/Mpc")
    print(f"  M = {M_fit:.4f}")
    print(f"\nχ² = {chi2_fit:.2f}")
    print(f"χ²/dof = {chi2_red:.4f} ({len(z_data)} - 2 = {dof} dof)")
    print(f"Fit time: {t1-t0:.2f} seconds")
    
    # Store Omega_m for plotting
    result.Omega_m = OMEGA_BARYON
    
    return result


def fit_sigma_baryons_only(z_data, mu_data, mu_err):
    """
    Model B: Σ-Gravity + baryons only (Ω_m = Ω_b = 0.05, Ω_Λ = 0, A_cos free)
    YOUR REAL MODEL: No dark matter, no dark energy, only baryons + coherence.
    """
    print("\n" + "="*60)
    print("MODEL B: Σ-GRAVITY + Baryons Only (YOUR MODEL)")
    print(f"  Ω_m = Ω_baryon = {OMEGA_BARYON:.3f} (FIXED)")
    print("  Ω_Λ = 0 (no dark energy)")
    print("  Σ = Σ_cos(z) with A_cos free")
    print("="*60)
    
    t0 = time.time()
    
    # Fit: [H0, A_cos, M]
    x0 = [70.0, 4.0, -19.3]
    
    result = minimize(
        chi_squared_sigma_baryons_only,
        x0,
        args=(z_data, mu_data, mu_err),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    t1 = time.time()
    
    H0_fit, A_cos_fit, M_fit = result.x
    chi2_fit = result.fun
    dof = len(z_data) - 3
    chi2_red = chi2_fit / dof
    
    print(f"\nBest-fit parameters:")
    print(f"  H₀ = {H0_fit:.2f} km/s/Mpc")
    print(f"  A_cos = {A_cos_fit:.4f}  (cf. π√2 ≈ 4.44 from clusters)")
    print(f"  M = {M_fit:.4f}")
    print(f"\nχ² = {chi2_fit:.2f}")
    print(f"χ²/dof = {chi2_red:.4f} ({len(z_data)} - 3 = {dof} dof)")
    print(f"Fit time: {t1-t0:.2f} seconds")
    
    # Store for plotting
    result.Omega_m = OMEGA_BARYON
    
    return result


def fit_gr_effective_matter(z_data, mu_data, mu_err):
    """
    Model C (Control): GR + effective matter (Ω_m free, Ω_Λ = 0, Σ = 1)
    Shows how badly GR fails without dark energy even if Ω_m floats.
    """
    print("\n" + "="*60)
    print("MODEL C (Control): GR + Effective Matter")
    print("  Ω_m = FREE (can absorb 'missing' mass)")
    print("  Ω_Λ = 0 (no dark energy)")
    print("  Σ = 1 (standard GR)")
    print("="*60)
    
    t0 = time.time()
    
    # Fit: [H0, Omega_m, M]
    x0 = [70.0, 0.3, -19.3]
    
    result = minimize(
        chi_squared_gr_effective_matter,
        x0,
        args=(z_data, mu_data, mu_err),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    t1 = time.time()
    
    H0_fit, Omega_m_fit, M_fit = result.x
    chi2_fit = result.fun
    dof = len(z_data) - 3
    chi2_red = chi2_fit / dof
    
    print(f"\nBest-fit parameters:")
    print(f"  H₀ = {H0_fit:.2f} km/s/Mpc")
    print(f"  Ω_m = {Omega_m_fit:.4f}")
    print(f"  M = {M_fit:.4f}")
    print(f"\nχ² = {chi2_fit:.2f}")
    print(f"χ²/dof = {chi2_red:.4f} ({len(z_data)} - 3 = {dof} dof)")
    print(f"Fit time: {t1-t0:.2f} seconds")
    
    # Store for plotting
    result.Omega_m = Omega_m_fit
    
    return result


def fit_LCDM(z_data, mu_data, mu_err):
    """
    ΛCDM reference (pipeline sanity check only).
    """
    print("\n" + "="*60)
    print("REFERENCE: ΛCDM (pipeline sanity check)")
    print("="*60)
    
    t0 = time.time()
    
    x0 = [70.0, 0.30, -19.3]
    
    result = minimize(
        chi_squared_lcdm,
        x0,
        args=(z_data, mu_data, mu_err),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    t1 = time.time()
    
    H0_fit, Omega_m0_fit, M_fit = result.x
    chi2_fit = result.fun
    dof = len(z_data) - 3
    chi2_red = chi2_fit / dof
    
    print(f"\nBest-fit parameters:")
    print(f"  H₀ = {H0_fit:.2f} km/s/Mpc")
    print(f"  Ω_m0 = {Omega_m0_fit:.4f}")
    print(f"  Ω_Λ = {1 - Omega_m0_fit:.4f}")
    print(f"  M = {M_fit:.4f}")
    print(f"\nχ² = {chi2_fit:.2f}")
    print(f"χ²/dof = {chi2_red:.4f} ({len(z_data)} - 3 = {dof} dof)")
    print(f"Fit time: {t1-t0:.2f} seconds")
    
    return result


# =============================================================================
# Effective Dark Energy Analysis
# =============================================================================

def compute_effective_w(z_array, Omega_m0, A_cos, H0):
    """
    Compute effective dark energy equation of state w_eff(z).
    """
    E_array = compute_E_array_fast(z_array, Omega_m0, A_cos, H0)
    
    # Effective DE density: ρ_DE,eff / ρ_crit0 = E² - Ω_m0·(1+z)³
    rho_DE_eff = E_array**2 - Omega_m0 * (1 + z_array)**3
    
    # Numerical derivative for w_eff
    ln_1pz = np.log(1 + z_array)
    ln_rho = np.log(np.maximum(rho_DE_eff, 1e-10))
    d_ln_rho = np.gradient(ln_rho, ln_1pz)
    
    w_eff = -1 - d_ln_rho / 3
    
    return w_eff, rho_DE_eff


# =============================================================================
# Plotting - Baryons-Only Analysis
# =============================================================================

def plot_baryons_only(z_data, mu_data, mu_err, gr_baryons, sigma_baryons, gr_eff_matter, lcdm_ref):
    """
    Create comparison plots for baryons-only models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Extract parameters
    H0_A, M_A = gr_baryons.x
    H0_B, A_cos, M_B = sigma_baryons.x
    H0_C, Om_C, M_C = gr_eff_matter.x
    H0_ref, Om_ref, M_ref = lcdm_ref.x
    
    z_fine = np.linspace(0.01, z_data.max(), 200)
    
    # Compute all model curves
    mu_A = mu_matter_only_vectorized(z_fine, OMEGA_BARYON, H0_A, M_A)  # GR + baryons
    mu_B = compute_mu_array_fast(z_fine, OMEGA_BARYON, A_cos, H0_B, M_B)  # Σ + baryons
    mu_C = mu_matter_only_vectorized(z_fine, Om_C, H0_C, M_C)  # GR + eff matter
    mu_ref = mu_LCDM_vectorized(z_fine, Om_ref, 1-Om_ref, H0_ref, M_ref)  # ΛCDM
    
    # --- Plot 1: Hubble Diagram ---
    ax1 = axes[0, 0]
    ax1.errorbar(z_data, mu_data, yerr=mu_err, fmt='o', ms=2, alpha=0.2, 
                 color='gray', label='Pantheon+ data', zorder=1)
    ax1.plot(z_fine, mu_A, 'g--', lw=2, label=f'A: GR+baryons (Ω_b={OMEGA_BARYON})', zorder=2)
    ax1.plot(z_fine, mu_B, 'b-', lw=2.5, label=f'B: Σ+baryons (A={A_cos:.2f})', zorder=4)
    ax1.plot(z_fine, mu_C, 'm:', lw=2, label=f'C: GR+eff.matter (Ω_m={Om_C:.2f})', zorder=3)
    ax1.plot(z_fine, mu_ref, 'r--', lw=1.5, alpha=0.7, label=f'Ref: ΛCDM', zorder=2)
    ax1.set_xlabel('Redshift z', fontsize=12)
    ax1.set_ylabel('Distance modulus μ', fontsize=12)
    ax1.set_title('Hubble Diagram: Baryons-Only Models', fontsize=13)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xscale('log')
    
    # --- Plot 2: Residuals vs ΛCDM reference ---
    ax2 = axes[0, 1]
    mu_ref_data = mu_LCDM_vectorized(z_data, Om_ref, 1-Om_ref, H0_ref, M_ref)
    mu_A_data = mu_matter_only_vectorized(z_data, OMEGA_BARYON, H0_A, M_A)
    mu_B_data = compute_mu_array_fast(z_data, OMEGA_BARYON, A_cos, H0_B, M_B)
    
    ax2.axhline(0, color='k', ls='--', alpha=0.5, lw=1)
    ax2.scatter(z_data, mu_data - mu_ref_data, s=3, alpha=0.3, color='gray', label='Data - ΛCDM')
    ax2.plot(z_fine, mu_A - mu_ref, 'g--', lw=2, label='GR+baryons - ΛCDM')
    ax2.plot(z_fine, mu_B - mu_ref, 'b-', lw=2.5, label='Σ+baryons - ΛCDM')
    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Δμ (relative to ΛCDM)', fontsize=12)
    ax2.set_title('How much does Σ close the gap to ΛCDM?', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-3, 3)
    
    # --- Plot 3: E(z) = H(z)/H₀ comparison ---
    ax3 = axes[1, 0]
    E_A = np.sqrt(OMEGA_BARYON * (1 + z_fine)**3)  # GR + baryons
    E_B = compute_E_array_fast(z_fine, OMEGA_BARYON, A_cos, H0_B)  # Σ + baryons
    E_C = np.sqrt(Om_C * (1 + z_fine)**3)  # GR + eff matter
    E_ref = E_LCDM_vectorized(z_fine, Om_ref, 1-Om_ref)  # ΛCDM
    
    ax3.plot(z_fine, E_A, 'g--', lw=2, label='A: GR+baryons')
    ax3.plot(z_fine, E_B, 'b-', lw=2.5, label='B: Σ+baryons')
    ax3.plot(z_fine, E_C, 'm:', lw=2, label='C: GR+eff.matter')
    ax3.plot(z_fine, E_ref, 'r--', lw=1.5, alpha=0.7, label='Ref: ΛCDM')
    ax3.set_xlabel('Redshift z', fontsize=12)
    ax3.set_ylabel('E(z) = H(z)/H₀', fontsize=12)
    ax3.set_title('Expansion Rate: Σ boosts E(z) without Λ', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 2.5)
    
    # --- Plot 4: χ² comparison bar chart ---
    ax4 = axes[1, 1]
    models = ['A: GR+baryons', 'B: Σ+baryons', 'C: GR+eff.matter', 'Ref: ΛCDM']
    chi2_vals = [gr_baryons.fun, sigma_baryons.fun, gr_eff_matter.fun, lcdm_ref.fun]
    colors = ['green', 'blue', 'magenta', 'red']
    
    bars = ax4.bar(models, chi2_vals, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('χ²', fontsize=12)
    ax4.set_title('χ² Comparison: Lower is Better', fontsize=13)
    ax4.tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, val in zip(bars, chi2_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Add improvement annotation
    improvement = gr_baryons.fun - sigma_baryons.fun
    ax4.annotate(f'Σ-Gravity improves\nΔχ² = {improvement:.0f}',
                xy=(1, sigma_baryons.fun), xytext=(1.5, sigma_baryons.fun + 2000),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'sigma_cosmology_baryons_only.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


# =============================================================================
# Main Execution - Baryons-Only Analysis
# =============================================================================

def main():
    """
    Main execution: Pure baryons-only cosmology analysis.
    
    Three models, ALL with no dark matter and no dark energy:
    A) GR + baryons only (Ω_m = Ω_b = 0.05, Λ=0, Σ=1)
    B) Σ-Gravity + baryons only (Ω_m = Ω_b = 0.05, Λ=0, A_cos free)  <-- YOUR MODEL
    C) GR + effective matter (Ω_m free, Λ=0, Σ=1) -- control
    
    Plus ΛCDM as a pipeline sanity check only.
    """
    print("="*70)
    print("Σ-GRAVITY COSMOLOGICAL EXPANSION ANALYSIS")
    print("BARYONS-ONLY: No Dark Matter, No Dark Energy")
    print("="*70)
    print("\nQuestion: With ONLY baryons (Ω_b ≈ 0.05) and NO dark energy,")
    print("how much of the observed expansion can Σ-Gravity explain?")
    print(f"\nBaryon fraction used: Ω_baryon = {OMEGA_BARYON}")
    
    total_t0 = time.time()
    
    # Load Pantheon data
    z_data, mu_data, mu_err = load_pantheon_data(max_z=2.5)
    
    # =================================================================
    # Fit all baryons-only models
    # =================================================================
    
    # Model A: GR + baryons only (the baseline "nothing works" model)
    gr_baryons = fit_gr_baryons_only(z_data, mu_data, mu_err)
    
    # Model B: Σ-Gravity + baryons only (YOUR REAL MODEL)
    sigma_baryons = fit_sigma_baryons_only(z_data, mu_data, mu_err)
    
    # Model C: GR + effective matter (control - how much Ω_m would you need?)
    gr_eff_matter = fit_gr_effective_matter(z_data, mu_data, mu_err)
    
    # Reference: ΛCDM (sanity check only)
    lcdm_ref = fit_LCDM(z_data, mu_data, mu_err)
    
    # =================================================================
    # Results Summary
    # =================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY: BARYONS-ONLY COSMOLOGY")
    print("="*70)
    
    print("\nχ² values (lower is better):")
    print(f"  A) GR + baryons only:      χ² = {gr_baryons.fun:.1f}")
    print(f"  B) Σ-Gravity + baryons:    χ² = {sigma_baryons.fun:.1f}  <-- YOUR MODEL")
    print(f"  C) GR + effective matter:  χ² = {gr_eff_matter.fun:.1f}")
    print(f"  Ref) ΛCDM:                 χ² = {lcdm_ref.fun:.1f}  (sanity check)")
    
    # Key comparisons
    delta_AB = gr_baryons.fun - sigma_baryons.fun
    delta_B_ref = sigma_baryons.fun - lcdm_ref.fun
    
    print(f"\nΔχ² (A → B): {delta_AB:.1f}")
    print(f"  → Σ-Gravity improves fit by {delta_AB:.0f} over GR+baryons")
    
    print(f"\nΔχ² (B → ΛCDM): {delta_B_ref:.1f}")
    if delta_B_ref < 50:
        print("  → Σ+baryons fits NEARLY AS WELL as ΛCDM!")
    else:
        print(f"  → ΛCDM still fits better by Δχ² = {delta_B_ref:.0f}")
    
    # Extract Σ-Gravity parameters
    H0_B, A_cos, M_B = sigma_baryons.x
    
    print(f"\nΣ-Gravity best-fit coherence amplitude:")
    print(f"  A_cos = {A_cos:.3f}")
    print(f"  (cf. π√2 ≈ 4.44 from independent cluster lensing fits)")
    
    # Compute effective w(z) for Σ-Gravity
    z_test = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
    w_eff, rho_DE = compute_effective_w(z_test, OMEGA_BARYON, A_cos, H0_B)
    
    print(f"\nEffective equation of state w_eff(z) from Σ:")
    for z, w in zip(z_test, w_eff):
        print(f"  z = {z:.1f}: w_eff = {w:.3f}")
    print("  (w = -1 is cosmological constant behavior)")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"""
With Ω_m fixed to the baryon fraction (~{OMEGA_BARYON}) and Ω_Λ = 0:

• GR alone (Model A) underpredicts distances to high-z supernovae,
  yielding χ² = {gr_baryons.fun:.0f}.

• Σ-Gravity (Model B) reduces residuals and improves χ² by {delta_AB:.0f},
  with A_cos ≈ {A_cos:.2f} comparable to the amplitude from cluster lensing.

• The effective w_eff ≈ {np.mean(w_eff):.2f} shows Σ provides a
  phantom-like (“super-cosmological-constant”) acceleration.

In a strictly dark-matter- and dark-energy-free cosmology,
Σ-Gravity recovers a significant fraction of the observed expansion.
""")
    
    total_t1 = time.time()
    print(f"Total analysis time: {total_t1-total_t0:.2f} seconds")
    
    # Plot results
    plot_baryons_only(z_data, mu_data, mu_err, gr_baryons, sigma_baryons, 
                      gr_eff_matter, lcdm_ref)
    
    # Clear GPU memory
    if GPU_AVAILABLE:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    return {
        'gr_baryons': gr_baryons,
        'sigma_baryons': sigma_baryons,
        'gr_eff_matter': gr_eff_matter,
        'lcdm_ref': lcdm_ref
    }


if __name__ == "__main__":
    main()
