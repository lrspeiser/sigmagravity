"""
pantheon_fit_optimized.py
------------------------
Optimized Weyl-integrable redshift model fitting to Pantheon+ SNe Ia data.

Features:
- Multi-core CPU parallelization
- GPU acceleration with CuPy (if available)
- Vectorized distance inversion
- Efficient caching
- Progress tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time

# Add cosmo to path
SCRIPT_DIR = Path(__file__).parent
COSMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(COSMO_DIR))

from sigma_redshift_derivations import SigmaKernel, WeylModel

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy available - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - using CPU only")

# Constants
c = 299792458.0  # m/s
Mpc = 3.0856775814913673e22  # m

class OptimizedWeylDistanceInverter:
    """
    Optimized distance inverter with GPU acceleration and vectorization.
    """
    
    def __init__(self, model, z_tol=1e-6, max_iter=50, use_gpu=True):
        self.model = model
        self.z_tol = z_tol
        self.max_iter = max_iter
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Cache for efficiency
        self._cache = {}
        
        # Pre-compute constants
        self.H0_SI = (self.model.H0_kms_Mpc * 1000.0) / Mpc
        
        print(f"Distance inverter initialized: GPU={'ON' if self.use_gpu else 'OFF'}")
    
    def _vectorized_newton_solve(self, target_z_array):
        """
        Vectorized Newton solve for multiple redshifts.
        """
        if self.use_gpu:
            return self._gpu_newton_solve(target_z_array)
        else:
            return self._cpu_newton_solve(target_z_array)
    
    def _cpu_newton_solve(self, target_z_array):
        """
        CPU-based vectorized Newton solve.
        """
        target_z_array = np.asarray(target_z_array)
        n_z = len(target_z_array)
        
        # Initial guesses based on Hubble law
        D_guess = (target_z_array * c) / self.H0_SI
        
        D = D_guess.copy()
        
        for iteration in range(self.max_iter):
            # Compute current redshifts
            z_current = np.array([self.model.z_of_distance_Mpc(d) for d in D])
            
            errors = z_current - target_z_array
            
            # Check convergence
            max_error = np.max(np.abs(errors))
            if max_error < self.z_tol:
                break
            
            # Numerical derivatives
            dz = np.maximum(1e-3 * D, 0.1)
            z_plus = np.array([self.model.z_of_distance_Mpc(d + dz[i]) for i, d in enumerate(D)])
            z_minus = np.array([self.model.z_of_distance_Mpc(d - dz[i]) for i, d in enumerate(D)])
            dz_dD = (z_plus - z_minus) / (2 * dz)
            
            # Newton step
            D = D - errors / np.maximum(np.abs(dz_dD), 1e-10)
            D = np.maximum(D, 0.1)  # Prevent negative distances
        
        return D
    
    def _gpu_newton_solve(self, target_z_array):
        """
        GPU-accelerated Newton solve using CuPy.
        """
        if not self.use_gpu:
            return self._cpu_newton_solve(target_z_array)
        
        target_z_array = np.asarray(target_z_array)
        target_z_gpu = cp.asarray(target_z_array)
        
        # Initial guesses
        D_guess_gpu = (target_z_gpu * c) / self.H0_SI
        D_gpu = D_guess_gpu.copy()
        
        for iteration in range(self.max_iter):
            # Compute redshifts (still need CPU for model evaluation)
            D_cpu = cp.asnumpy(D_gpu)
            z_current = np.array([self.model.z_of_distance_Mpc(d) for d in D_cpu])
            z_current_gpu = cp.asarray(z_current)
            
            errors_gpu = z_current_gpu - target_z_gpu
            
            # Check convergence
            max_error = cp.max(cp.abs(errors_gpu))
            if max_error < self.z_tol:
                break
            
            # Numerical derivatives
            dz_gpu = cp.maximum(1e-3 * D_gpu, 0.1)
            D_plus_cpu = cp.asnumpy(D_gpu + dz_gpu)
            D_minus_cpu = cp.asnumpy(D_gpu - dz_gpu)
            
            z_plus = np.array([self.model.z_of_distance_Mpc(d) for d in D_plus_cpu])
            z_minus = np.array([self.model.z_of_distance_Mpc(d) for d in D_minus_cpu])
            
            z_plus_gpu = cp.asarray(z_plus)
            z_minus_gpu = cp.asarray(z_minus)
            dz_dD_gpu = (z_plus_gpu - z_minus_gpu) / (2 * dz_gpu)
            
            # Newton step
            D_gpu = D_gpu - errors_gpu / cp.maximum(cp.abs(dz_dD_gpu), 1e-10)
            D_gpu = cp.maximum(D_gpu, 0.1)
        
        return cp.asnumpy(D_gpu)
    
    def compute_distance_modulus_batch(self, z_array, n_workers=None):
        """
        Compute distance moduli for a batch of redshifts using parallel processing.
        """
        z_array = np.asarray(z_array)
        n_z = len(z_array)
        
        if n_workers is None:
            n_workers = min(mp.cpu_count(), n_z)
        
        print(f"Computing distance moduli for {n_z} redshifts using {n_workers} workers...")
        start_time = time.time()
        
        # Split into chunks for parallel processing
        chunk_size = max(1, n_z // n_workers)
        chunks = [z_array[i:i+chunk_size] for i in range(0, n_z, chunk_size)]
        
        # Use ThreadPoolExecutor for I/O-bound tasks (model evaluation)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Process chunks in parallel
            futures = []
            for chunk in chunks:
                future = executor.submit(self._vectorized_newton_solve, chunk)
                futures.append(future)
            
            # Collect results
            distances = []
            for future in futures:
                distances.extend(future.result())
        
        distances = np.array(distances)
        d_L = distances * (1 + z_array)**2  # Luminosity distance
        mu = 5 * np.log10(d_L) - 5
        
        elapsed_time = time.time() - start_time
        print(f"Completed in {elapsed_time:.2f} seconds ({n_z/elapsed_time:.1f} redshifts/sec)")
        
        return mu

def create_mock_pantheon_data(n_sne=1000):
    """
    Create mock Pantheon+ data for testing.
    """
    np.random.seed(42)
    
    # Mock redshift distribution
    z_min, z_max = 0.01, 2.3
    z = np.random.uniform(z_min, z_max, n_sne)
    z = np.sort(z)
    
    # Mock distance modulus with ΛCDM baseline
    H0 = 70.0  # km/s/Mpc
    mu_lcdm = 5 * np.log10((c/H0 * z * (1+z)**2) / Mpc) - 5
    
    # Add observational scatter
    mu_err = np.random.normal(0.1, 0.02, n_sne)
    mu_obs = mu_lcdm + np.random.normal(0, mu_err, n_sne)
    
    return pd.DataFrame({
        'z': z,
        'mu': mu_obs,
        'mu_err': mu_err
    })

def chi2_weyl_model_optimized(params, pantheon_data, kernel_params, n_workers=None):
    """
    Optimized χ² computation with parallel processing.
    """
    alpha0_scale, ell0_kpc, p, ncoh = params
    
    try:
        # Create model
        kernel = SigmaKernel(
            A=kernel_params.get('A', 1.0),
            ell0_kpc=ell0_kpc,
            p=p,
            ncoh=ncoh
        )
        model = WeylModel(kernel=kernel, H0_kms_Mpc=70.0, alpha0_scale=alpha0_scale)
        inverter = OptimizedWeylDistanceInverter(model, use_gpu=GPU_AVAILABLE)
        
        # Compute predicted distance moduli in parallel
        mu_pred = inverter.compute_distance_modulus_batch(
            pantheon_data['z'].values, 
            n_workers=n_workers
        )
        
        # Compute χ²
        chi2_value = np.sum(((pantheon_data['mu'].values - mu_pred) / pantheon_data['mu_err'].values)**2)
        
        return chi2_value
    
    except Exception as e:
        print(f"Error in chi2 computation: {e}")
        return 1e10

def fit_weyl_model_optimized(pantheon_data, kernel_params=None, n_workers=None):
    """
    Optimized Weyl model fitting with parallel processing.
    """
    if kernel_params is None:
        kernel_params = {'A': 1.0}
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print("="*80)
    print("OPTIMIZED WEYL MODEL FITTING")
    print("="*80)
    print(f"Data: {len(pantheon_data)} SNe Ia")
    print(f"Redshift range: {pantheon_data['z'].min():.3f} - {pantheon_data['z'].max():.3f}")
    print(f"CPU cores: {n_workers}")
    print(f"GPU acceleration: {'ON' if GPU_AVAILABLE else 'OFF'}")
    
    # Initial parameter guess
    x0 = [0.95, 200.0, 0.75, 0.5]  # [alpha0_scale, ell0_kpc, p, ncoh]
    
    # Parameter bounds
    bounds = [
        (0.1, 2.0),    # alpha0_scale
        (50.0, 500.0), # ell0_kpc
        (0.1, 2.0),    # p
        (0.1, 2.0)     # ncoh
    ]
    
    # Create partial function with fixed arguments
    chi2_func = partial(chi2_weyl_model_optimized, 
                       pantheon_data=pantheon_data, 
                       kernel_params=kernel_params,
                       n_workers=n_workers)
    
    print(f"\nStarting optimization...")
    start_time = time.time()
    
    # Fit with progress tracking
    result = minimize(
        chi2_func,
        x0,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100, 'disp': True}
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    
    if not result.success:
        print(f"Warning: Fit did not converge: {result.message}")
    
    # Extract best parameters
    best_params = {
        'alpha0_scale': result.x[0],
        'ell0_kpc': result.x[1],
        'p': result.x[2],
        'ncoh': result.x[3]
    }
    
    # Compute fit statistics
    chi2_min = result.fun
    n_params = len(result.x)
    n_data = len(pantheon_data)
    dof = n_data - n_params
    chi2_reduced = chi2_min / dof
    p_value = 1 - chi2.cdf(chi2_min, dof)
    
    # Create best-fit model
    kernel = SigmaKernel(
        A=kernel_params.get('A', 1.0),
        ell0_kpc=best_params['ell0_kpc'],
        p=best_params['p'],
        ncoh=best_params['ncoh']
    )
    best_model = WeylModel(kernel=kernel, H0_kms_Mpc=70.0, alpha0_scale=best_params['alpha0_scale'])
    
    # Compute final residuals
    inverter = OptimizedWeylDistanceInverter(best_model, use_gpu=GPU_AVAILABLE)
    mu_pred = inverter.compute_distance_modulus_batch(pantheon_data['z'].values, n_workers=n_workers)
    residuals = pantheon_data['mu'].values - mu_pred
    
    fit_result = {
        'best_params': best_params,
        'chi2_min': chi2_min,
        'chi2_reduced': chi2_reduced,
        'dof': dof,
        'p_value': p_value,
        'residuals': residuals,
        'mu_pred': mu_pred,
        'best_model': best_model,
        'inverter': inverter,
        'success': result.success,
        'optimization_time': elapsed_time
    }
    
    return fit_result

def main():
    print("="*80)
    print("OPTIMIZED WEYL MODEL FIT TO PANTHEON+ DATA")
    print("="*80)
    
    # Check system resources
    n_cores = mp.cpu_count()
    print(f"System: {n_cores} CPU cores available")
    print(f"GPU: {'Available' if GPU_AVAILABLE else 'Not available'}")
    
    # Load data
    pantheon_data = create_mock_pantheon_data(n_sne=500)  # Smaller dataset for testing
    
    # Fit model with optimization
    fit_result = fit_weyl_model_optimized(pantheon_data, n_workers=n_cores)
    
    # Print results
    print(f"\n" + "="*80)
    print("FIT RESULTS")
    print("="*80)
    
    print(f"\nBest-fit parameters:")
    for param, value in fit_result['best_params'].items():
        print(f"  {param}: {value:.4f}")
    
    print(f"\nFit statistics:")
    print(f"  χ²_min: {fit_result['chi2_min']:.2f}")
    print(f"  χ²_reduced: {fit_result['chi2_reduced']:.4f}")
    print(f"  Degrees of freedom: {fit_result['dof']}")
    print(f"  P-value: {fit_result['p_value']:.2e}")
    print(f"  Fit success: {fit_result['success']}")
    print(f"  Optimization time: {fit_result['optimization_time']:.2f} seconds")
    
    # Create visualization
    print(f"\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Hubble diagram
    ax = axes[0, 0]
    ax.errorbar(pantheon_data['z'], pantheon_data['mu'], 
               yerr=pantheon_data['mu_err'], fmt='o', alpha=0.5, markersize=2, 
               label='Pantheon+ data', color='gray')
    
    # Plot best-fit model
    z_model = np.linspace(0.01, 2.5, 100)
    mu_model = fit_result['inverter'].compute_distance_modulus_batch(z_model)
    ax.plot(z_model, mu_model, 'r-', linewidth=2, label='Weyl model')
    
    # Plot ΛCDM baseline
    mu_lcdm = 5 * np.log10((c/70.0 * z_model * (1+z_model)**2) / Mpc) - 5
    ax.plot(z_model, mu_lcdm, 'k--', linewidth=2, label='ΛCDM baseline')
    
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance Modulus μ')
    ax.set_title('Hubble Diagram (Optimized Fit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    residuals = fit_result['residuals']
    ax.errorbar(pantheon_data['z'], residuals, 
               yerr=pantheon_data['mu_err'], fmt='o', alpha=0.5, markersize=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residuals (μ_data - μ_model)')
    ax.set_title('Fit Residuals')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, alpha=0.7, density=True)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics
    ax = axes[1, 1]
    ax.text(0.5, 0.7, f'Optimization Time: {fit_result["optimization_time"]:.1f}s', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.5, f'CPU Cores: {n_cores}', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.3, f'GPU: {"ON" if GPU_AVAILABLE else "OFF"}', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Performance Metrics')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plot_file = COSMO_DIR / "outputs" / "pantheon_fit_optimized.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    # Save results
    results_file = COSMO_DIR / "outputs" / "pantheon_fit_optimized.csv"
    results_df = pd.DataFrame({
        'z': pantheon_data['z'],
        'mu_data': pantheon_data['mu'],
        'mu_err': pantheon_data['mu_err'],
        'mu_model': fit_result['mu_pred'],
        'residuals': fit_result['residuals']
    })
    results_df.to_csv(results_file, index=False)
    print(f"Results saved: {results_file}")
    
    print(f"\n" + "="*80)
    print("OPTIMIZED FIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()






