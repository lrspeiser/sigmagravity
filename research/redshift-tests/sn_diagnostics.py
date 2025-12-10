"""
SN Diagnostics Module: Advanced validation for TG-tau redshift prescription

Implements:
- Full covariance chi2
- Parameter error estimation  
- Anisotropy/robustness probes
- Redshift slice analysis
- Residual systematics analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata
import time
from typing import Dict, Tuple, Optional, List

# Import our optimized modules
from tg_tau_fast import fit_tg_tau_fast, load_pantheon_fast, TGtauParamsFast
from sigma_redshift_toy_models import (
    C_KM_S, c_over_H0_mpc, distance_modulus_from_DL_Mpc,
    chi2, aic, bic, time_dilation_penalty, tolman_penalty
)

# ============================================================================
# FULL COVARIANCE CHI2
# ============================================================================

def chi2_full_cov(mu_obs: np.ndarray, mu_model: np.ndarray, C: np.ndarray) -> float:
    """
    chi^2 = r^T C^{-1} r using a stable Cholesky solve.
    C must be positive definite (Pantheon+ total covariance is).
    """
    r = np.asarray(mu_obs) - np.asarray(mu_model)
    L = np.linalg.cholesky(C)
    y = np.linalg.solve(L, r)
    return float(y @ y)

def load_pantheon_covariance(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Pantheon+ covariance matrix"""
    print(f"Loading Pantheon+ covariance from {filepath}...")
    
    # Load the covariance file
    cov_data = np.loadtxt(filepath)
    
    # The covariance file format varies - need to handle different cases
    if cov_data.ndim == 1:
        # Diagonal covariance
        C = np.diag(cov_data)
    else:
        # Full covariance matrix
        C = cov_data
    
    print(f"Covariance matrix shape: {C.shape}")
    print(f"Covariance matrix condition number: {np.linalg.cond(C):.2e}")
    
    return C

# ============================================================================
# PARAMETER ERROR ESTIMATION
# ============================================================================

def quad_errors_2d(f: callable, H_best: float, a_best: float, 
                   dH: float = 0.5, da: float = 0.01) -> Tuple[float, float, float]:
    """
    f(H, a) -> chi^2. Returns (sigma_H, sigma_a, rho).
    Uses finite-difference Hessian around best grid point.
    """
    def F(H, a): 
        return f(H, a)

    c = F(H_best, a_best)
    fHH = (F(H_best+dH, a_best) - 2*c + F(H_best-dH, a_best)) / dH**2
    faa = (F(H_best, a_best+da) - 2*c + F(H_best, a_best-da)) / da**2
    fHa = (F(H_best+dH, a_best+da) - F(H_best+dH, a_best-da)
          -F(H_best-dH, a_best+da) + F(H_best-dH, a_best-da)) / (4*dH*da)

    H = np.array([[fHH, fHa],[fHa, faa]])
    Cov = np.linalg.inv(0.5*H)  # because chi^2 curvature ~ 1/2 * Hessian
    sigH, siga = np.sqrt(np.diag(Cov))
    rho = Cov[0,1]/(sigH*siga)
    return float(sigH), float(siga), float(rho)

def estimate_tg_tau_errors(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray,
                          H_best: float, a_best: float, C: Optional[np.ndarray] = None) -> Dict:
    """Estimate parameter errors for TG-tau model"""
    
    def chi2_func(H, a):
        """Chi2 function for parameter error estimation"""
        pars = TGtauParamsFast(HSigma=H, alpha_SB=a)
        D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z)
        DL = D * (1.0 + z)**pars.alpha_SB
        mu_model = distance_modulus_from_DL_Mpc(DL)
        
        if C is not None:
            return chi2_full_cov(mu, mu_model, C)
        else:
            return chi2(mu, mu_model, sigma_mu)
    
    try:
        sigH, siga, rho = quad_errors_2d(chi2_func, H_best, a_best)
        
        return {
            "sigma_H": sigH,
            "sigma_a": siga,
            "correlation": rho,
            "H_best": H_best,
            "a_best": a_best,
            "H_range": (H_best - 2*sigH, H_best + 2*sigH),
            "a_range": (a_best - 2*siga, a_best + 2*siga)
        }
    except Exception as e:
        print(f"Error estimation failed: {e}")
        return {"error": str(e)}

# ============================================================================
# ANISOTROPY ANALYSIS
# ============================================================================

def hemispheres(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray, 
                ra_deg: np.ndarray, dec_deg: np.ndarray, nhat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split data into hemispheres based on nhat direction"""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    zhat = np.sin(dec)
    rhat = np.stack([x,y,zhat], axis=1)
    mask = (rhat @ np.asarray(nhat)) > 0
    return mask, ~mask

def test_hemispherical_anisotropy(data: Dict, nhat: np.ndarray = None) -> Dict:
    """Test for hemispherical anisotropy in TG-tau parameters"""
    if nhat is None:
        nhat = np.array([0, 0, 1])  # Default: north/south split
    
    # Load RA/DEC from data (assuming they exist)
    if 'RA' not in data or 'DEC' not in data:
        print("Warning: RA/DEC not available for anisotropy test")
        return {"error": "RA/DEC not available"}
    
    mask_north, mask_south = hemispheres(
        data["z"], data["mu"], data["sigma_mu"], 
        data["RA"], data["DEC"], nhat
    )
    
    print(f"North hemisphere: {np.sum(mask_north)} SNe")
    print(f"South hemisphere: {np.sum(mask_south)} SNe")
    
    # Fit TG-tau to each hemisphere
    results = {}
    
    try:
        # North hemisphere
        res_north = fit_tg_tau_fast(
            data["z"][mask_north], 
            data["mu"][mask_north], 
            data["sigma_mu"][mask_north]
        )
        results["north"] = res_north
        
        # South hemisphere
        res_south = fit_tg_tau_fast(
            data["z"][mask_south], 
            data["mu"][mask_south], 
            data["sigma_mu"][mask_south]
        )
        results["south"] = res_south
        
        # Compare parameters
        H_diff = res_north["pars"].HSigma - res_south["pars"].HSigma
        a_diff = res_north["pars"].alpha_SB - res_south["pars"].alpha_SB
        
        results["comparison"] = {
            "H_diff": H_diff,
            "a_diff": a_diff,
            "H_north": res_north["pars"].HSigma,
            "H_south": res_south["pars"].HSigma,
            "a_north": res_north["pars"].alpha_SB,
            "a_south": res_south["pars"].alpha_SB
        }
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

# ============================================================================
# REDSHIFT SLICE ANALYSIS
# ============================================================================

def analyze_redshift_slices(data: Dict, z_slices: List[Tuple[float, float]] = None) -> Dict:
    """Analyze TG-tau parameters in different redshift slices"""
    if z_slices is None:
        z_slices = [(0.0, 0.2), (0.2, 0.6), (0.6, 2.0)]
    
    results = {}
    
    for i, (z_min, z_max) in enumerate(z_slices):
        print(f"\nAnalyzing redshift slice {z_min} < z < {z_max}")
        
        # Filter data
        mask = (data["z"] >= z_min) & (data["z"] < z_max)
        n_sne = np.sum(mask)
        
        if n_sne < 50:  # Need sufficient data
            print(f"  Insufficient data: {n_sne} SNe")
            continue
        
        print(f"  {n_sne} SNe in slice")
        
        try:
            # Fit TG-tau to this slice
            res = fit_tg_tau_fast(
                data["z"][mask],
                data["mu"][mask], 
                data["sigma_mu"][mask]
            )
            
            results[f"slice_{i}"] = {
                "z_range": (z_min, z_max),
                "n_sne": n_sne,
                "H_Sigma": res["pars"].HSigma,
                "alpha_SB": res["pars"].alpha_SB,
                "xi": res["xi_inferred"],
                "chi2": res["chi2"],
                "score": res["score"]
            }
            
            print(f"  H_Sigma: {res['pars'].HSigma:.2f} km/s/Mpc")
            print(f"  alpha_SB: {res['pars'].alpha_SB:.3f}")
            print(f"  xi: {res['xi_inferred']:.2e}")
            
        except Exception as e:
            print(f"  Fitting failed: {e}")
            results[f"slice_{i}"] = {"error": str(e)}
    
    return results

# ============================================================================
# RESIDUAL SYSTEMATICS ANALYSIS
# ============================================================================

def analyze_residual_systematics(data: Dict, H_best: float, a_best: float) -> Dict:
    """Analyze residuals for systematic trends"""
    
    # Compute model predictions
    pars = TGtauParamsFast(HSigma=H_best, alpha_SB=a_best)
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + data["z"])
    DL = D * (1.0 + data["z"])**pars.alpha_SB
    mu_model = distance_modulus_from_DL_Mpc(DL)
    
    # Compute residuals
    residuals = data["mu"] - mu_model
    
    results = {
        "residuals": residuals,
        "residual_std": np.std(residuals),
        "residual_mean": np.mean(residuals)
    }
    
    # Analyze trends vs various quantities
    if 'HOST_LOGMASS' in data:
        # Host mass trend
        host_mass = data['HOST_LOGMASS']
        valid_mask = ~np.isnan(host_mass)
        if np.sum(valid_mask) > 100:
            correlation = np.corrcoef(residuals[valid_mask], host_mass[valid_mask])[0,1]
            results["host_mass_correlation"] = correlation
            print(f"Residual vs host mass correlation: {correlation:.4f}")
    
    if 'c' in data:
        # Color trend
        color = data['c']
        valid_mask = ~np.isnan(color)
        if np.sum(valid_mask) > 100:
            correlation = np.corrcoef(residuals[valid_mask], color[valid_mask])[0,1]
            results["color_correlation"] = correlation
            print(f"Residual vs color correlation: {correlation:.4f}")
    
    if 'x1' in data:
        # Stretch trend
        stretch = data['x1']
        valid_mask = ~np.isnan(stretch)
        if np.sum(valid_mask) > 100:
            correlation = np.corrcoef(residuals[valid_mask], stretch[valid_mask])[0,1]
            results["stretch_correlation"] = correlation
            print(f"Residual vs stretch correlation: {correlation:.4f}")
    
    # Redshift trend
    correlation = np.corrcoef(residuals, data["z"])[0,1]
    results["redshift_correlation"] = correlation
    print(f"Residual vs redshift correlation: {correlation:.4f}")
    
    return results

# ============================================================================
# COMPREHENSIVE DIAGNOSTICS
# ============================================================================

def run_comprehensive_diagnostics(data_path: str, cov_path: str = None) -> Dict:
    """Run all diagnostic tests"""
    print("=" * 80)
    print("COMPREHENSIVE TG-tau DIAGNOSTICS")
    print("=" * 80)
    
    # Load data
    data = load_pantheon_fast(data_path)
    
    # Load covariance if available
    C = None
    if cov_path:
        try:
            C = load_pantheon_covariance(cov_path)
        except Exception as e:
            print(f"Warning: Could not load covariance: {e}")
    
    # Initial TG-tau fit
    print("\n1. INITIAL TG-tau FIT")
    print("-" * 50)
    res_initial = fit_tg_tau_fast(data["z"], data["mu"], data["sigma_mu"])
    
    print(f"H_Sigma: {res_initial['pars'].HSigma:.2f} km/s/Mpc")
    print(f"alpha_SB: {res_initial['pars'].alpha_SB:.3f}")
    print(f"xi: {res_initial['xi_inferred']:.2e}")
    print(f"chi2: {res_initial['chi2']:.2f}")
    
    # Parameter error estimation
    print("\n2. PARAMETER ERROR ESTIMATION")
    print("-" * 50)
    errors = estimate_tg_tau_errors(
        data["z"], data["mu"], data["sigma_mu"],
        res_initial["pars"].HSigma, res_initial["pars"].alpha_SB, C
    )
    
    if "error" not in errors:
        print(f"sigma_H: {errors['sigma_H']:.2f} km/s/Mpc")
        print(f"sigma_a: {errors['sigma_a']:.3f}")
        print(f"correlation: {errors['correlation']:.3f}")
    
    # Redshift slice analysis
    print("\n3. REDSHIFT SLICE ANALYSIS")
    print("-" * 50)
    slice_results = analyze_redshift_slices(data)
    
    # Residual systematics
    print("\n4. RESIDUAL SYSTEMATICS")
    print("-" * 50)
    residual_results = analyze_residual_systematics(
        data, res_initial["pars"].HSigma, res_initial["pars"].alpha_SB
    )
    
    # Anisotropy test (if RA/DEC available)
    print("\n5. HEMISPHERICAL ANISOTROPY")
    print("-" * 50)
    anisotropy_results = test_hemispherical_anisotropy(data)
    
    # Compile results
    results = {
        "initial_fit": res_initial,
        "parameter_errors": errors,
        "redshift_slices": slice_results,
        "residual_systematics": residual_results,
        "anisotropy": anisotropy_results,
        "covariance_used": C is not None
    }
    
    return results

if __name__ == "__main__":
    # Run diagnostics on Pantheon+ data
    results = run_comprehensive_diagnostics(
        "../data/pantheon/Pantheon+SH0ES.dat",
        "../data/pantheon/Pantheon+SH0ES_STAT+SYS.cov"
    )
    
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print("Check results dictionary for detailed analysis")
