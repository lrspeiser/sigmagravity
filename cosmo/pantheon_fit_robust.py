"""
pantheon_fit_robust.py
---------------------
Robust Weyl-integrable redshift model fitting with better numerical stability.

Features:
- Improved numerical stability
- Better parameter bounds
- Robust error handling
- Faster convergence
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

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy available - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - using CPU only")

# Add cosmo to path
SCRIPT_DIR = Path(__file__).parent
COSMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(COSMO_DIR))

from sigma_redshift_derivations import SigmaKernel, WeylModel

# Import the actual ImprovedWeylModel from the improved explorer
try:
    sys.path.insert(0, str(SCRIPT_DIR / "examples"))
    from explore_weyl_redshift_improved import ImprovedWeylModel
    print("✅ Using ImprovedWeylModel from explore_weyl_redshift_improved.py")
except ImportError:
    print("⚠️ ImprovedWeylModel not found, using local implementation")
    
    class ImprovedWeylModel(WeylModel):
        """
        Improved Weyl model with saturation and gradient coupling options.
        """
        
        def __init__(self, kernel, H0_kms_Mpc=70.0, alpha0_scale=1.0, 
                     saturation_eps=0.0, gradient_xi=0.0, **kwargs):
            super().__init__(kernel, H0_kms_Mpc, alpha0_scale, **kwargs)
            self.saturation_eps = saturation_eps  # 0 < eps < 1 for saturation
            self.gradient_xi = gradient_xi        # gradient coupling strength
        
        def alpha_effective(self, l_array):
            """
            Effective alpha with saturation and/or gradient coupling.
            """
            alpha0_base = self.alpha0_per_m()
            
            if self.saturation_eps > 0:
                # Saturation: alpha_eff(l) = alpha0 * [1 - eps * C(l)]
                C = self.kernel.C_along_line(l_array)
                return alpha0_base * (1 - self.saturation_eps * C)
            elif self.gradient_xi > 0:
                # Gradient coupling: Q_μ k^μ = alpha0 * [C + xi * ℓ0 * ∂_l C]
                C = self.kernel.C_along_line(l_array)
                dl = l_array[1] - l_array[0] if len(l_array) > 1 else l_array[0] * 0.01
                dC_dl = np.gradient(C, dl)
                ell0 = self.kernel.SI()['ell0']
                return alpha0_base * (C + self.gradient_xi * ell0 * dC_dl)
            else:
                # Standard case: alpha_eff = alpha0 * C(l)
                C = self.kernel.C_along_line(l_array)
                return alpha0_base * C
    
    def z_of_distance_Mpc(self, D_Mpc, n_steps=20000):
        """
        Improved redshift calculation with effective alpha.
        """
        L = D_Mpc * Mpc
        l = np.linspace(0.0, L, int(n_steps)+1)
        alpha_eff = self.alpha_effective(l)
        integral  = np.trapz(alpha_eff, l)
        return np.expm1(integral)

# Constants
c = 299792458.0  # m/s
Mpc = 3.0856775814913673e22  # m

class RobustWeylDistanceInverter:
    """
    Robust distance inverter with better numerical stability.
    """
    
    def __init__(self, model, z_tol=1e-5, max_iter=20):
        self.model = model
        self.z_tol = z_tol
        self.max_iter = max_iter
        
        # Pre-compute constants
        self.H0_SI = (self.model.H0_kms_Mpc * 1000.0) / Mpc
        
        # Cache for efficiency
        self._cache = {}
    
    def find_distance_from_redshift(self, target_z):
        """
        Robust Newton solve for r(z) with better numerical stability.
        """
        if target_z in self._cache:
            return self._cache[target_z]
        
        # Initial guess based on Hubble law (convert to Mpc)
        D_guess = (target_z * c) / self.H0_SI / Mpc
        
        D = D_guess
        for iteration in range(self.max_iter):
            try:
                z_current = self.model.z_of_distance_Mpc(D)
                error = z_current - target_z
                
                if abs(error) < self.z_tol:
                    self._cache[target_z] = D
                    return D
                
                # Numerical derivative with adaptive step size
                dz = max(1e-3 * D, 0.1)
                z_plus = self.model.z_of_distance_Mpc(D + dz)
                z_minus = self.model.z_of_distance_Mpc(D - dz)
                dz_dD = (z_plus - z_minus) / (2 * dz)
                
                if abs(dz_dD) < 1e-10:
                    break
                
                # Newton step with damping
                step = -error / dz_dD
                D_new = D + 0.5 * step  # Damping factor
                D_new = max(D_new, 0.1)  # Prevent negative distances
                
                # Check for reasonable bounds
                if D_new > 10000:  # 10 Gpc limit
                    break
                
                D = D_new
                
            except (OverflowError, ValueError):
                # Handle numerical issues gracefully
                break
        
        self._cache[target_z] = D
        return D
    
    def compute_distance_modulus(self, z_array, verbose=False):
        """
        Compute distance moduli with robust error handling.
        """
        z_array = np.asarray(z_array)
        distances = []
        n_z = len(z_array)
        
        for i, z in enumerate(z_array):
            if verbose and (i % max(1, n_z // 10) == 0):  # Print progress every 10%
                print(f"    Computing distance for z={z:.3f} ({i+1}/{n_z})")
            
            try:
                distance = self.find_distance_from_redshift(z)
                distances.append(distance)
            except Exception as e:
                # Fallback to Hubble law for problematic redshifts
                distance = (z * c) / self.H0_SI / Mpc
                distances.append(distance)
        
        distances = np.array(distances)
        d_L_Mpc = distances * (1 + z_array)**2  # Luminosity distance in Mpc
        
        # Handle potential overflow in log
        d_L_Mpc = np.clip(d_L_Mpc, 1e-10, 1e10)  # Clip to reasonable range
        mu = 5 * np.log10(d_L_Mpc) + 25  # μ with d_L in Mpc (since 1 Mpc = 10^6 pc)
        
        return mu
    
    def compute_distance_modulus_parallel(self, z_array, n_cores=None):
        """
        Compute distance modulus μ(z) using parallel processing.
        """
        if n_cores is None:
            n_cores = min(mp.cpu_count(), 8)  # Use up to 8 cores
        
        print(f"  Using {n_cores} CPU cores for parallel computation...")
        
        # Split the redshift array into chunks
        chunk_size = max(1, len(z_array) // n_cores)
        chunks = [z_array[i:i+chunk_size] for i in range(0, len(z_array), chunk_size)]
        
        # Create a partial function with the inverter instance
        compute_chunk = partial(self._compute_chunk_distances)
        
        # Use ProcessPoolExecutor for CPU-bound work
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunk_results = list(executor.map(compute_chunk, chunks))
        
        # Flatten results
        distances = []
        for chunk_distances in chunk_results:
            distances.extend(chunk_distances)
        
        distances = np.array(distances)
        d_L_Mpc = distances * (1 + z_array)**2  # Luminosity distance in Mpc
        
        # Handle potential overflow in log
        d_L_Mpc = np.clip(d_L_Mpc, 1e-10, 1e10)  # Clip to reasonable range
        mu = 5 * np.log10(d_L_Mpc) + 25  # μ with d_L in Mpc (since 1 Mpc = 10^6 pc)
        
        return mu
    
    def _compute_chunk_distances(self, z_chunk):
        """Helper function for parallel distance computation."""
        distances = []
        for z in z_chunk:
            try:
                distance = self.find_distance_from_redshift(z)
                distances.append(distance)
            except Exception as e:
                # Fallback to Hubble law for problematic redshifts
                distance = (z * c) / self.H0_SI / Mpc
                distances.append(distance)
        return distances

def load_pantheon_data():
    """
    Load Pantheon+ (SH0ES) and align the full STAT+SYS covariance to the filtered,
    z-sorted subset used in the fit. Returns (df, C_inv) where df has columns
    {z, mu, mu_err, i_orig} in sorted-by-z order, and C_inv is the matching (N,N).
    """
    import numpy as np, pandas as pd
    data_file = "data/pantheon/Pantheon+SH0ES.dat"
    cov_file  = "data/pantheon/Pantheon+SH0ES_STAT+SYS.cov"

    print("  Loading Pantheon+ data file...")
    df_raw = pd.read_csv(data_file, sep=r"\s+", comment="#", engine="python")
    df_raw["i_orig"] = np.arange(len(df_raw), dtype=int)  # remember original row

    # Columns per the SH0ES table
    z     = df_raw["zCMB"].to_numpy()
    mu    = df_raw["MU_SH0ES"].to_numpy()
    sigma = df_raw["MU_SH0ES_ERR_DIAG"].to_numpy()
    valid = np.isfinite(z) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)

    print(f"  Raw rows: {len(df_raw)} | Valid after cuts: {valid.sum()}")
    idx_valid = np.flatnonzero(valid)                    # indices in original order

    # Build a small dataframe, then sort by z (BUT keep i_orig so we can reorder C)
    df = df_raw.loc[idx_valid, ["zCMB","MU_SH0ES","MU_SH0ES_ERR_DIAG","i_orig"]].copy()
    df.columns = ["z","mu","mu_err","i_orig"]
    df.sort_values("z", inplace=True, ignore_index=True)

    # --- Covariance alignment ---
    print("  Loading Pantheon+ STAT+SYS covariance...")
    C_full = np.loadtxt(cov_file)
    
    # Check if covariance is flattened and reshape if needed
    if C_full.ndim == 1:
        # Assume it's a flattened symmetric matrix, need to determine size
        N_full = int(np.sqrt(len(C_full)))
        if N_full * N_full != len(C_full):
            print(f"  ⚠️ Cov matrix length {len(C_full)} is not a perfect square. Diagonal fallback.")
            return df, None
        C_full = C_full.reshape(N_full, N_full)
        print(f"  Reshaped covariance from {len(C_full.flatten())} elements to {C_full.shape}")
    else:
        N_full = C_full.shape[0]

    if N_full < idx_valid.max()+1:
        # Unlikely, but guard anyway
        print(f"  ⚠️ Cov matrix ({N_full}) smaller than max index in data ({idx_valid.max()+1}). Diagonal fallback.")
        return df, None

    # 1) subset to valid rows (original order)
    C_sub = C_full[np.ix_(idx_valid, idx_valid)]         # (N_valid, N_valid)

    # 2) reorder to z-sorted order
    # Build a map from original index -> position in the "valid" list
    pos_in_valid = {orig: j for j, orig in enumerate(idx_valid)}
    # For each sorted row, what is its position in C_sub?
    perm = np.array([pos_in_valid[o] for o in df["i_orig"].to_numpy()], dtype=int)
    C = C_sub[np.ix_(perm, perm)]                        # match df row order exactly

    # Numerical safety: cholesky try; fall back to inv if needed
    try:
        import scipy.linalg as sl
        L = sl.cho_factor(C, overwrite_a=False, check_finite=False)
        # Keep (L, lower/upper flag) so we can apply C^{-1} via solves later
        Cinvt = ("cho", L)
        print(f"  ✅ Full covariance aligned: N = {C.shape[0]}")
    except Exception as e:
        print(f"  ⚠️ Cholesky failed ({e}); using explicit inverse (this is OK at N~1700)")
        try:
            Cinv = np.linalg.inv(C)
            Cinvt = ("dense", Cinv)
            print(f"  ✅ Full covariance inverted: N = {Cinv.shape[0]}")
        except Exception as e2:
            print(f"  ❌ Could not invert covariance: {e2}. Diagonal fallback.")
            Cinvt = None

    return df, Cinvt

def create_mock_pantheon_data(n_sne=200):
    """
    Create smaller, more manageable mock Pantheon+ data as fallback.
    """
    np.random.seed(42)
    
    # Mock redshift distribution with more reasonable range
    z_min, z_max = 0.01, 1.5  # Reduced max redshift for stability
    z = np.random.uniform(z_min, z_max, n_sne)
    z = np.sort(z)
    
    # Mock distance modulus with ΛCDM baseline
    H0 = 70.0  # km/s/Mpc
    mu_lcdm = 5 * np.log10((c/H0 * z * (1+z)**2) / Mpc) - 5
    
    # Add observational scatter
    mu_err = np.random.normal(0.1, 0.02, n_sne)
    mu_obs = mu_lcdm + np.random.normal(0, mu_err, n_sne)
    
    # Create mock data with no covariance matrix
    pantheon = pd.DataFrame({
        'z': z,
        'mu': mu_obs,
        'mu_err': mu_err,
        'cov_matrix': [None] * n_sne,
        'inv_cov': [None] * n_sne
    })
    
    return pantheon

def chi2_weyl_model_robust(params, pantheon_data, kernel_params, Cinv_tag, verbose=False):
    """
    Robust χ² computation with full covariance matrix and μ₀ marginalization.
    """
    alpha0_scale, ell0_kpc, p, ncoh, saturation_eps, gradient_xi = params
    
    if verbose:
        print(f"  Testing params: α₀={alpha0_scale:.3f}, ℓ₀={ell0_kpc:.1f}, p={p:.3f}, n={ncoh:.3f}")
    
    try:
        # Create model with reasonable parameter bounds
        kernel = SigmaKernel(
            A=kernel_params.get('A', 1.0),
            ell0_kpc=np.clip(ell0_kpc, 50.0, 500.0),
            p=np.clip(p, 0.1, 2.0),
            ncoh=np.clip(ncoh, 0.1, 2.0)
        )
        model = ImprovedWeylModel(kernel=kernel, H0_kms_Mpc=70.0, alpha0_scale=np.clip(alpha0_scale, 0.1, 3.0),
                                  saturation_eps=np.clip(saturation_eps, 0.0, 0.9), 
                                  gradient_xi=np.clip(gradient_xi, 0.0, 1.0))
        inverter = RobustWeylDistanceInverter(model)
        
        # Compute predicted distance moduli (use parallel processing for large datasets)
        if len(pantheon_data) > 500:
            mu_pred = inverter.compute_distance_modulus_parallel(pantheon_data['z'].values)
        else:
            mu_pred = inverter.compute_distance_modulus(pantheon_data['z'].values, verbose=verbose)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(mu_pred)) or np.any(np.isinf(mu_pred)):
            return 1e10
        
        # Compute residuals
        residuals = pantheon_data['mu'].values - mu_pred
        
        # Gaussian priors (unchanged)
        prior_penalty = ((p - 0.75)/0.1)**2 + ((ncoh - 0.5)/0.1)**2

        if Cinv_tag is not None:
            ones = np.ones_like(residuals)
            if Cinv_tag[0] == "cho":
                import scipy.linalg as sl
                L = Cinv_tag[1]
                # Use solves to apply C^{-1} x without forming C^{-1}
                y_res = sl.cho_solve(L, residuals, check_finite=False)
                y_one = sl.cho_solve(L, ones,      check_finite=False)
                num   = ones @ y_res
                den   = ones @ y_one
                mu0_star = num / den
                chi2_value = residuals @ y_res - (num**2)/den
            else:
                Cinv = Cinv_tag[1]
                num  = ones @ (Cinv @ residuals)
                den  = ones @ (Cinv @ ones)
                mu0_star   = num / den
                chi2_value = residuals @ (Cinv @ residuals) - (num**2)/den

            if verbose:
                print(f"    → Full covariance χ² = {chi2_value:.2f}, μ₀* = {mu0_star:.3f}")
        else:
            # Diagonal fallback with μ-offset profiling
            w = 1.0 / pantheon_data['mu_err'].values**2
            delta_mu = np.sum(w*residuals)/np.sum(w)
            chi2_value = np.sum(((residuals - delta_mu)**2) * w)
            mu0_star = delta_mu
            if verbose:
                print(f"    → Diagonal χ² = {chi2_value:.2f}, Δμ = {delta_mu:.3f}")

        chi2_value += prior_penalty
        
        # Check for reasonable χ²
        if np.isnan(chi2_value) or np.isinf(chi2_value):
            if verbose:
                print(f"    → Numerical issue: χ² = {chi2_value}")
            return 1e10
        
        if verbose:
            print(f"    → Final χ² = {chi2_value:.2f} (including priors: {prior_penalty:.2f})")
        
        return chi2_value
    
    except Exception as e:
        if verbose:
            print(f"    → Exception in chi2: {e}")
        return 1e10

def fit_weyl_model_robust(pantheon_data, kernel_params=None, Cinv_tag=None):
    """
    Robust Weyl model fitting with better parameter bounds.
    """
    if kernel_params is None:
        kernel_params = {'A': 1.0}
    
    print("="*80)
    print("ROBUST WEYL MODEL FITTING")
    print("="*80)
    print(f"Data: {len(pantheon_data)} SNe Ia")
    print(f"Redshift range: {pantheon_data['z'].min():.3f} - {pantheon_data['z'].max():.3f}")
    
    # Initial parameter guess (closer to known good values)
    x0 = [1.5, 200.0, 0.75, 0.5, 0.1, 0.05]  # [alpha0_scale, ell0_kpc, p, ncoh, saturation_eps, gradient_xi]
    
    # FIX C: Parameter bounds with improved model parameters - avoid hitting limits
    bounds = [
        (0.1, 10.0),     # alpha0_scale (much wider range to avoid hitting bounds)
        (50.0, 500.0),   # ell0_kpc (reasonable range)
        (0.5, 1.0),      # p (tighter around galaxy/cluster calibrated values: 0.75 ± 0.1)
        (0.3, 0.7),      # ncoh (tighter around galaxy/cluster calibrated values: 0.5 ± 0.1)
        (0.0, 0.5),      # saturation_eps (saturation strength)
        (0.0, 0.5)       # gradient_xi (gradient coupling strength)
    ]
    
    print(f"\nStarting optimization with tighter bounds...")
    start_time = time.time()
    
    # Progress callback for optimization
    iteration_count = [0]
    def callback(xk):
        iteration_count[0] += 1
        if iteration_count[0] % 5 == 0:  # Print every 5 iterations
            print(f"  Iteration {iteration_count[0]}: α₀={xk[0]:.3f}, ℓ₀={xk[1]:.1f}, p={xk[2]:.3f}, n={xk[3]:.3f}, ε={xk[4]:.3f}, ξ={xk[5]:.3f}")
        elif iteration_count[0] % 1 == 0:  # Show progress every iteration
            print(f"  Iteration {iteration_count[0]}...", end='\r')
    
    # Fit with more robust optimizer
    result = minimize(
        chi2_weyl_model_robust,
        x0,
        args=(pantheon_data, kernel_params, Cinv_tag),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 50, 'ftol': 1e-6},
        callback=callback
    )
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
    if not result.success:
        print(f"Warning: Fit did not converge: {result.message}")
        print("Using best parameters found...")
    
    # Extract best parameters
    best_params = {
        'alpha0_scale': result.x[0],
        'ell0_kpc': result.x[1],
        'p': result.x[2],
        'ncoh': result.x[3],
        'saturation_eps': result.x[4],
        'gradient_xi': result.x[5]
    }
    
    # Compute fit statistics
    chi2_min = result.fun
    n_params = len(result.x)      # α0, ℓ0, p, ncoh, eps, xi
    n_data = len(pantheon_data)   # SNe count
    dof = n_data - n_params - 1   # minus 1 for profiled μ0
    
    if not np.isnan(chi2_min) and not np.isinf(chi2_min):
        chi2_reduced = chi2_min / dof
        p_value = 1 - chi2.cdf(chi2_min, dof)
    else:
        chi2_reduced = np.nan
        p_value = np.nan
    
    # Create best-fit model
    kernel = SigmaKernel(
        A=kernel_params.get('A', 1.0),
        ell0_kpc=best_params['ell0_kpc'],
        p=best_params['p'],
        ncoh=best_params['ncoh']
    )
    best_model = ImprovedWeylModel(kernel=kernel, H0_kms_Mpc=70.0, alpha0_scale=best_params['alpha0_scale'],
                                  saturation_eps=best_params['saturation_eps'],
                                  gradient_xi=best_params['gradient_xi'])
    
    # Compute final residuals with proper μ₀ profiling
    inverter = RobustWeylDistanceInverter(best_model)
    mu_pred = inverter.compute_distance_modulus(pantheon_data['z'].values)
    residuals = pantheon_data['mu'].values - mu_pred
    
    # Compute profiled residuals (same as used in χ²)
    if Cinv_tag is not None:
        # Full covariance case
        ones = np.ones_like(residuals)
        if Cinv_tag[0] == "cho":
            import scipy.linalg as sl
            L = Cinv_tag[1]
            y_res = sl.cho_solve(L, residuals, check_finite=False)
            y_one = sl.cho_solve(L, ones, check_finite=False)
            mu0_star = (ones @ y_res) / (ones @ y_one)
        else:
            Cinv = Cinv_tag[1]
            mu0_star = (ones @ (Cinv @ residuals)) / (ones @ (Cinv @ ones))
        residuals_profiled = residuals - mu0_star
        print(f"  Using full covariance with μ₀* = {mu0_star:.3f}")
    else:
        # Diagonal case
        w = 1.0 / pantheon_data['mu_err'].values**2
        delta_mu = np.sum(w * residuals) / np.sum(w)
        residuals_profiled = residuals - delta_mu
        print(f"  Using diagonal errors with Δμ = {delta_mu:.3f}")
    
    # Use profiled residuals for plotting
    residuals = residuals_profiled
    
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
    print("ROBUST WEYL MODEL FIT TO PANTHEON+ DATA")
    print("="*80)
    
    # Load real Pantheon+ data
    print("Loading Pantheon+ data...")
    pantheon_data, Cinv_tag = load_pantheon_data()
    
    # Fit model with robust optimization
    fit_result = fit_weyl_model_robust(pantheon_data, Cinv_tag=Cinv_tag)
    
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
               yerr=pantheon_data['mu_err'], fmt='o', alpha=0.5, markersize=3, 
               label='Pantheon+ data', color='gray')
    
    # Plot best-fit model
    z_model = np.linspace(0.01, 1.5, 100)
    mu_model = fit_result['inverter'].compute_distance_modulus(z_model)
    ax.plot(z_model, mu_model, 'r-', linewidth=2, label='Weyl model')
    
    # FIX B: Proper ΛCDM reference function (flat ΛCDM with FRW luminosity distance)
    def mu_LCDM_flat(z, H0=70.0, Om=0.3):
        from scipy.integrate import quad
        Ol = 1.0 - Om  # Dark energy density
        c_km_s = 299792.458  # Speed of light in km/s
        
        # Hubble parameter E(z) = sqrt(Om*(1+z)^3 + Ol)
        Ez = lambda zp: np.sqrt(Om*(1+zp)**3 + Ol)
        
        # Comoving distance: D_C = (c/H0) * ∫_0^z dz'/E(z')
        chi = np.array([quad(lambda zz: 1.0/Ez(zz), 0, zi, limit=200)[0] for zi in np.atleast_1d(z)])
        
        # Luminosity distance: D_L = (1+z) * D_C
        dL_Mpc = (c_km_s/H0) * (1+np.atleast_1d(z)) * chi
        
        # Distance modulus: μ = 5*log10(D_L/Mpc) + 25
        return 5*np.log10(dL_Mpc) + 25
    
    mu_lcdm = mu_LCDM_flat(z_model)
    ax.plot(z_model, mu_lcdm, 'k--', linewidth=2, label='ΛCDM baseline')
    
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance Modulus μ')
    ax.set_title('Hubble Diagram (Robust Fit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    residuals = fit_result['residuals']
    
    # Filter out NaN residuals
    valid_mask = ~np.isnan(residuals)
    if np.any(valid_mask):
        ax.errorbar(pantheon_data['z'][valid_mask], residuals[valid_mask], 
                   yerr=pantheon_data['mu_err'][valid_mask], fmt='o', alpha=0.5, markersize=3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Redshift z')
        ax.set_ylabel('Residuals (μ_data - μ_model)')
        ax.set_title('Fit Residuals')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid residuals', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fit Residuals (No Data)')
    
    # Plot 3: Residual histogram
    ax = axes[1, 0]
    if np.any(valid_mask):
        valid_residuals = residuals[valid_mask]
        ax.hist(valid_residuals, bins=20, alpha=0.7, density=True)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid residuals', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Residual Distribution (No Data)')
    
    # Plot 4: Parameter summary
    ax = axes[1, 1]
    param_text = f"α₀ scale: {fit_result['best_params']['alpha0_scale']:.3f}\n"
    param_text += f"ℓ₀: {fit_result['best_params']['ell0_kpc']:.1f} kpc\n"
    param_text += f"p: {fit_result['best_params']['p']:.3f}\n"
    param_text += f"n_coh: {fit_result['best_params']['ncoh']:.3f}\n"
    param_text += f"χ²: {fit_result['chi2_min']:.1f}\n"
    param_text += f"Time: {fit_result['optimization_time']:.1f}s"
    
    ax.text(0.5, 0.5, param_text, ha='center', va='center', transform=ax.transAxes, 
            fontsize=10, fontfamily='monospace')
    ax.set_title('Fit Summary')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plot_file = SCRIPT_DIR / "outputs" / "pantheon_fit_robust.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_file}")
    
    # Save results
    results_file = SCRIPT_DIR / "outputs" / "pantheon_fit_robust.csv"
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
    print("ROBUST FIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
