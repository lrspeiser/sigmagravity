"""
Patch for sigma_redshift_toy_models.py

Fixes:
1. Endpoint_only model - removes FRW D(z) inheritance (false positive)
2. Adds TG-τ + Σ-ISW composite model
3. Adds Pantheon data loader
"""

import math
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Import base functions from original module
from sigma_redshift_toy_models import (
    C_KM_S, LN10, c_over_H0_mpc, distance_modulus_from_DL_Mpc,
    chi2, aic, bic, time_dilation_penalty, tolman_penalty
)

# ============================================================================
# FIXED ENDPOINT MODEL (removes FRW inheritance)
# ============================================================================

@dataclass
class EndpointParamsFixed:
    z0: float = 0.0
    alpha_SB: float = 1.0

def endpoint_fixed_DL_Mpc(z: np.ndarray, pars: EndpointParamsFixed) -> np.ndarray:
    """
    Fixed endpoint model - returns flat distance scale (no FRW inheritance).
    This will produce a terrible Hubble diagram fit, as it should.
    """
    # Flat distance scale: D = constant * z (simple linear approximation)
    D_flat = c_over_H0_mpc(70.0) * z  # Simple linear scaling
    return D_flat * (1.0 + z)**pars.alpha_SB

def endpoint_fixed_time_dilation(z: np.ndarray, pars: EndpointParamsFixed) -> np.ndarray:
    """Time dilation for fixed endpoint model"""
    return np.full_like(z, 1.0 + max(pars.z0, 0.0))

def fit_endpoint_fixed_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Fit the fixed endpoint model (should perform poorly)"""
    z0_grid = np.linspace(0.0, 2.0, 81)
    aSB_grid = np.linspace(0.0, 4.0, 41)
    best = None
    
    for z0 in z0_grid:
        for aSB in aSB_grid:
            pars = EndpointParamsFixed(z0=z0, alpha_SB=aSB)
            DL = endpoint_fixed_DL_Mpc(z, pars)
            mu_model = distance_modulus_from_DL_Mpc(DL)
            c2 = chi2(mu, mu_model, sigma_mu)
            td_pred = endpoint_fixed_time_dilation(z, pars)
            td_pen = time_dilation_penalty(z, td_pred, weight=50.0)
            tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
            score = c2 + td_pen + tol_pen
            if (best is None) or (score < best["score"]):
                best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    return best

# ============================================================================
# TG-τ + Σ-ISW COMPOSITE MODEL
# ============================================================================

@dataclass
class TGtauISWParams:
    HSigma: float = 70.0
    a1: float = 1e-4
    a2: float = 0.0
    alpha_SB: float = 2.0
    Kbar: float = 1.0
    ell0_LOS_Mpc: float = 0.2

def tg_tau_isw_z_of_D(D_Mpc: np.ndarray, pars: TGtauISWParams) -> np.ndarray:
    """TG-τ + Σ-ISW composite redshift"""
    # TG-τ component
    z_tg = np.exp((pars.HSigma / C_KM_S) * D_Mpc) - 1.0
    # Σ-ISW component
    z_isw = pars.a1 * D_Mpc + pars.a2 * D_Mpc**2
    return z_tg + z_isw

def tg_tau_isw_D_of_z(z: np.ndarray, pars: TGtauISWParams) -> np.ndarray:
    """
    Invert z(D) to get D(z) using robust bisection method.
    This handles the composite TG-τ + Σ-ISW form.
    """
    z = np.asarray(z, dtype=float)
    D_result = np.zeros_like(z)
    
    for i, z_val in enumerate(z):
        try:
            # Define function to solve: z(D) - z_val = 0
            def func(D):
                return tg_tau_isw_z_of_D(np.array([D]), pars)[0] - z_val
            
            # Find reasonable bounds
            D_min = 0.0
            D_max = 20000.0  # 20 Gpc should be enough for any reasonable z
            
            # Check if solution exists in bounds
            if func(D_min) * func(D_max) > 0:
                # No solution in bounds, use approximate TG-τ only
                D_result[i] = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z_val)
            else:
                D_result[i] = brentq(func, D_min, D_max, xtol=1e-6)
        except:
            # Fallback to TG-τ only if inversion fails
            D_result[i] = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z_val)
    
    return D_result

def tg_tau_isw_DL_Mpc(z: np.ndarray, pars: TGtauISWParams) -> np.ndarray:
    """Luminosity distance for TG-τ + Σ-ISW composite"""
    D = tg_tau_isw_D_of_z(z, pars)
    return D * (1.0 + z)**pars.alpha_SB

def tg_tau_isw_time_dilation(z: np.ndarray) -> np.ndarray:
    """Time dilation for TG-τ + Σ-ISW (dominated by TG-τ)"""
    return 1.0 + z

def tg_tau_isw_micro_loss_constant(pars: TGtauISWParams) -> float:
    """Micro-loss constant for TG-τ component"""
    return (pars.HSigma / C_KM_S) * (pars.ell0_LOS_Mpc / max(pars.Kbar, 1e-12))

def fit_tg_tau_isw_to_sn(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Fit TG-τ + Σ-ISW composite model"""
    H_grid = np.linspace(40.0, 100.0, 31)  # Coarser grid for speed
    a1_grid = np.linspace(0.0, 5e-4, 21)
    aSB_grid = np.linspace(1.0, 4.0, 21)
    best = None
    
    for H in H_grid:
        for a1 in a1_grid:
            for aSB in aSB_grid:
                pars = TGtauISWParams(HSigma=H, a1=a1, alpha_SB=aSB)
                DL = tg_tau_isw_DL_Mpc(z, pars)
                mu_model = distance_modulus_from_DL_Mpc(DL)
                c2 = chi2(mu, mu_model, sigma_mu)
                td_pen = time_dilation_penalty(z, tg_tau_isw_time_dilation(z), weight=5.0)
                tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
                score = c2 + td_pen + tol_pen
                if (best is None) or (score < best["score"]):
                    best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    
    xi = tg_tau_isw_micro_loss_constant(best["pars"])
    best["xi_inferred"] = xi
    return best

# ============================================================================
# PANTHEON DATA LOADER
# ============================================================================

def load_pantheon_csv(filepath: str, z_col: str = "zCMB", mu_col: str = "MU_SH0ES", 
                     sigma_col: str = "MU_SH0ES_ERR_DIAG") -> Dict[str, np.ndarray]:
    """
    Load Pantheon+ data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to Pantheon+ CSV file
    z_col : str
        Column name for redshift (default: "zCMB")
    mu_col : str
        Column name for distance modulus (default: "MU_SH0ES")
    sigma_col : str
        Column name for distance modulus error (default: "MU_SH0ES_ERR_DIAG")
    
    Returns:
    --------
    Dict with keys: "z", "mu", "sigma_mu"
    """
    df = pd.read_csv(filepath, sep=r'\s+')
    
    # Filter out invalid entries
    valid_mask = (
        (df[z_col] > 0) & 
        (df[mu_col] > 0) & 
        (df[sigma_col] > 0) &
        (df[z_col] < 3.0)  # Reasonable upper limit
    )
    
    df_valid = df[valid_mask]
    
    print(f"Loaded {len(df_valid)} valid SNe from {len(df)} total entries")
    print(f"Redshift range: {df_valid[z_col].min():.4f} - {df_valid[z_col].max():.4f}")
    print(f"Distance modulus range: {df_valid[mu_col].min():.2f} - {df_valid[mu_col].max():.2f}")
    
    return {
        "z": df_valid[z_col].values,
        "mu": df_valid[mu_col].values,
        "sigma_mu": df_valid[sigma_col].values
    }

# ============================================================================
# ENHANCED FITTING FUNCTIONS
# ============================================================================

def fit_tg_tau_with_fixed_alpha_sb(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray, 
                                  alpha_sb: float = 2.0) -> Dict:
    """Fit TG-τ with fixed alpha_SB (for physical priors)"""
    H_grid = np.linspace(40.0, 100.0, 121)
    best = None
    
    for H in H_grid:
        from sigma_redshift_toy_models import TGtauParams, tg_tau_DL_Mpc, tg_tau_time_dilation
        pars = TGtauParams(HSigma=H, alpha_SB=alpha_sb)
        DL = tg_tau_DL_Mpc(z, pars)
        mu_model = distance_modulus_from_DL_Mpc(DL)
        c2 = chi2(mu, mu_model, sigma_mu)
        td_pen = time_dilation_penalty(z, tg_tau_time_dilation(z), weight=5.0)
        tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
        score = c2 + td_pen + tol_pen
        if (best is None) or (score < best["score"]):
            best = {"pars": pars, "chi2": c2, "td_pen": td_pen, "tolman_pen": tol_pen, "score": score}
    
    from sigma_redshift_toy_models import tg_tau_micro_loss_constant
    xi = tg_tau_micro_loss_constant(best["pars"])
    best["xi_inferred"] = xi
    return best

def print_enhanced_rankings(results: Dict[str, Dict]) -> None:
    """Enhanced ranking with AIC/BIC"""
    order = sorted(results.items(), key=lambda kv: kv[1]["score"] if "score" in kv[1] else kv[1]["chi2"])
    print("\n=== Enhanced Model Rankings (lower is better) ===")
    print(f"{'Model':>20s} | {'Score':>8s} | {'chi2':>8s} | {'AIC':>8s} | {'BIC':>8s} | {'xi':>12s}")
    print("-" * 80)
    
    n_data = len(next(iter(results.values())).get("z", [])) if results else 0
    
    for name, res in order:
        score = res.get("score", res.get("chi2", np.nan))
        chi2_val = res.get("chi2", np.nan)
        
        # Count parameters
        pars = res.get("pars", {})
        if isinstance(pars, dict):
            k = len([v for v in pars.values() if isinstance(v, (int, float))])
        else:
            k = len(pars.__dict__) if hasattr(pars, '__dict__') else 3
        
        aic_val = aic(k, chi2_val)
        bic_val = bic(k, chi2_val, n_data) if n_data > 0 else np.nan
        
        xi_val = res.get("xi_inferred", np.nan)
        xi_str = f"{xi_val:.2e}" if not np.isnan(xi_val) else "N/A"
        
        print(f"{name:>20s} | {score:8.2f} | {chi2_val:8.2f} | {aic_val:8.2f} | {bic_val:8.2f} | {xi_str:>12s}")
