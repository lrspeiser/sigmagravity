"""
Fast TG-tau fitting with parallel processing (Unicode-safe version)
"""

import math
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import time

# Import base functions from original module
from sigma_redshift_toy_models import (
    C_KM_S, LN10, c_over_H0_mpc, distance_modulus_from_DL_Mpc,
    chi2, aic, bic, time_dilation_penalty, tolman_penalty
)

# ============================================================================
# FAST TG-tau FITTING WITH PARALLEL PROCESSING
# ============================================================================

@dataclass
class TGtauParamsFast:
    HSigma: float = 70.0
    alpha_SB: float = 2.0
    Kbar: float = 1.0
    ell0_LOS_Mpc: float = 0.2

def tg_tau_DL_Mpc_fast(z: np.ndarray, pars: TGtauParamsFast) -> np.ndarray:
    """Fast TG-tau luminosity distance calculation"""
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + np.asarray(z, dtype=float))
    return D * (1.0 + z)**pars.alpha_SB

def tg_tau_time_dilation_fast(z: np.ndarray) -> np.ndarray:
    """Fast time dilation calculation"""
    return 1.0 + z

def tg_tau_micro_loss_constant_fast(pars: TGtauParamsFast) -> float:
    """Fast micro-loss constant calculation"""
    return (pars.HSigma / C_KM_S) * (pars.ell0_LOS_Mpc / max(pars.Kbar, 1e-12))

def evaluate_tg_tau_point_fast(args):
    """Evaluate a single point in the TG-tau parameter space (Unicode-safe)"""
    H, alpha_SB, z, mu, sigma_mu = args
    
    pars = TGtauParamsFast(HSigma=H, alpha_SB=alpha_SB)
    DL = tg_tau_DL_Mpc_fast(z, pars)
    mu_model = distance_modulus_from_DL_Mpc(DL)
    c2 = chi2(mu, mu_model, sigma_mu)
    td_pen = time_dilation_penalty(z, tg_tau_time_dilation_fast(z), weight=5.0)
    tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
    score = c2 + td_pen + tol_pen
    
    return {
        "HSigma": H,
        "alpha_SB": alpha_SB,
        "chi2": c2,
        "td_pen": td_pen,
        "tolman_pen": tol_pen,
        "score": score
    }

def fit_tg_tau_fast(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray, 
                    n_cpus: int = None) -> Dict:
    """Fit TG-tau using fast parallel processing"""
    if n_cpus is None:
        n_cpus = min(cpu_count(), 10)  # Use up to 10 CPUs
    
    print(f"Using {n_cpus} CPUs for parallel TG-tau fitting...")
    
    # Create parameter grid (smaller for speed)
    H_grid = np.linspace(40.0, 100.0, 31)  # Reduced grid
    alpha_grid = np.linspace(0.0, 4.0, 21)   # Reduced grid
    
    # Prepare arguments for parallel processing
    args_list = [(H, alpha, z, mu, sigma_mu) 
                 for H in H_grid for alpha in alpha_grid]
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for better performance
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        results = list(executor.map(evaluate_tg_tau_point_fast, args_list))
    
    # Find best result
    best = min(results, key=lambda r: r["score"])
    
    # Reconstruct parameters object
    pars = TGtauParamsFast(HSigma=best["HSigma"], alpha_SB=best["alpha_SB"])
    
    # Add xi calculation
    xi = tg_tau_micro_loss_constant_fast(pars)
    
    result = {
        "pars": pars,
        "chi2": best["chi2"],
        "td_pen": best["td_pen"],
        "tolman_pen": best["tolman_pen"],
        "score": best["score"],
        "xi_inferred": xi
    }
    
    elapsed_time = time.time() - start_time
    print(f"Parallel TG-tau fitting completed in {elapsed_time:.2f} seconds")
    
    return result

# ============================================================================
# FAST PANTHEON DATA LOADER
# ============================================================================

def load_pantheon_fast(filepath: str, z_col: str = "zCMB", 
                      mu_col: str = "MU_SH0ES", 
                      sigma_col: str = "MU_SH0ES_ERR_DIAG") -> Dict[str, np.ndarray]:
    """Fast Pantheon+ data loader"""
    print(f"Loading Pantheon+ data from {filepath}...")
    
    # Use pandas with optimized settings
    df = pd.read_csv(filepath, sep=r'\s+', low_memory=False)
    
    # Filter out invalid entries
    valid_mask = (
        (df[z_col] > 0) & 
        (df[mu_col] > 0) & 
        (df[sigma_col] > 0) &
        (df[z_col] < 3.0) &  # Reasonable upper limit
        (df[z_col] > 0.001)   # Reasonable lower limit
    )
    
    df_valid = df[valid_mask]
    
    print(f"Loaded {len(df_valid)} valid SNe from {len(df)} total entries")
    print(f"Redshift range: {df_valid[z_col].min():.4f} - {df_valid[z_col].max():.4f}")
    print(f"Distance modulus range: {df_valid[mu_col].min():.2f} - {df_valid[mu_col].max():.2f}")
    
    # Convert to numpy arrays for better performance
    return {
        "z": df_valid[z_col].values.astype(np.float64),
        "mu": df_valid[mu_col].values.astype(np.float64),
        "sigma_mu": df_valid[sigma_col].values.astype(np.float64)
    }

# ============================================================================
# FAST TESTING SCRIPT
# ============================================================================

def run_fast_test():
    """Run a fast test of TG-tau on Pantheon data"""
    print("=" * 60)
    print("FAST TG-tau TEST ON PANTHEON+ DATA")
    print("=" * 60)
    
    # Load data
    data = load_pantheon_fast("../data/pantheon/Pantheon+SH0ES.dat")
    
    # Test fast fitting
    print("\nFast Parallel TG-tau Fitting:")
    try:
        res = fit_tg_tau_fast(data["z"], data["mu"], data["sigma_mu"])
        print(f"   H_Sigma: {res['pars'].HSigma:.2f} km/s/Mpc")
        print(f"   alpha_SB: {res['pars'].alpha_SB:.3f}")
        print(f"   xi: {res['xi_inferred']:.2e}")
        print(f"   chi2: {res['chi2']:.2f}")
        print(f"   Score: {res['score']:.2f}")
        
        # Test different alpha_SB values
        print("\nTesting different alpha_SB values:")
        for alpha_sb in [1.0, 2.0, 4.0]:
            args = (res['pars'].HSigma, alpha_sb, data["z"], data["mu"], data["sigma_mu"])
            test_res = evaluate_tg_tau_point_fast(args)
            print(f"   alpha_SB = {alpha_sb}: chi2 = {test_res['chi2']:.2f}, score = {test_res['score']:.2f}")
            
    except Exception as e:
        print(f"   Fast fitting failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print("+ Parallel CPU processing significantly faster than serial")
    print("+ Reduced grid size for faster iteration")
    print("+ Unicode-safe implementation")

if __name__ == "__main__":
    run_fast_test()
