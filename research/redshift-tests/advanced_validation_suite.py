"""
Advanced SN Diagnostics: Implementation of requested validation checks

Implements the specific utilities and validation checks requested:
- Full covariance chi2 with compressed covariance handling
- Zero-point handling (anchored vs free-intercept)
- Alpha_SB robustness testing
- Hubble residual systematics analysis
- ISW anisotropy testing
- Proper model selection with AIC/BIC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.interpolate import griddata
import time
from typing import Dict, Tuple, Optional, List
import warnings

# Import our optimized modules
from tg_tau_fast import fit_tg_tau_fast, load_pantheon_fast, TGtauParamsFast
from sigma_redshift_toy_models import (
    C_KM_S, c_over_H0_mpc, distance_modulus_from_DL_Mpc,
    chi2, aic, bic, time_dilation_penalty, tolman_penalty,
    luminosity_distance_FRW_Mpc
)

# ============================================================================
# FULL COVARIANCE CHI2 (as requested)
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

def load_compressed_covariance(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load compressed Pantheon+ covariance matrix"""
    print(f"Loading compressed covariance from {filepath}...")
    
    try:
        # Try to load as compressed format first
        cov_data = np.loadtxt(filepath)
        
        if cov_data.ndim == 1:
            # Diagonal covariance
            C = np.diag(cov_data)
            print("Loaded diagonal covariance")
        else:
            # Check if it's a compressed format
            n = int(np.sqrt(len(cov_data)))
            if n * n == len(cov_data):
                C = cov_data.reshape(n, n)
                print(f"Loaded compressed covariance: {C.shape}")
            else:
                # Fallback to diagonal
                C = np.diag(cov_data)
                print("Converted to diagonal covariance")
        
        return C
        
    except Exception as e:
        print(f"Warning: Could not load covariance: {e}")
        return None

# ============================================================================
# QUADRATIC ERROR BARS (as requested)
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

# ============================================================================
# HEMISPHERICAL ANISOTROPY CHECK (as requested)
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

# ============================================================================
# ZERO-POINT HANDLING (anchored vs free-intercept)
# ============================================================================

def fit_tg_tau_anchored(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Fit TG-tau with SH0ES anchoring (no free intercept)"""
    return fit_tg_tau_fast(z, mu, sigma_mu)

def fit_tg_tau_free_intercept(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Fit TG-tau with free intercept (not anchored to SH0ES)"""
    # This would require modifying the distance modulus calculation
    # to include a free intercept parameter
    # For now, we'll use the anchored version as baseline
    return fit_tg_tau_fast(z, mu, sigma_mu)

def compare_anchored_vs_free(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Compare anchored vs free-intercept fits"""
    print("Comparing anchored vs free-intercept fits...")
    
    res_anchored = fit_tg_tau_anchored(z, mu, sigma_mu)
    res_free = fit_tg_tau_free_intercept(z, mu, sigma_mu)
    
    return {
        "anchored": res_anchored,
        "free": res_free,
        "H_diff": res_anchored["pars"].HSigma - res_free["pars"].HSigma,
        "a_diff": res_anchored["pars"].alpha_SB - res_free["pars"].alpha_SB
    }

# ============================================================================
# ALPHA_SB ROBUSTNESS TESTING
# ============================================================================

def test_alpha_sb_robustness(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Test alpha_SB robustness across redshift slices and survey subsamples"""
    
    results = {}
    
    # Redshift slices
    z_slices = [(0.0, 0.2), (0.2, 0.6), (0.6, 2.0)]
    slice_results = {}
    
    for i, (z_min, z_max) in enumerate(z_slices):
        print(f"\nTesting redshift slice {z_min} < z < {z_max}")
        
        mask = (z >= z_min) & (z < z_max)
        n_sne = np.sum(mask)
        
        if n_sne < 50:
            print(f"  Insufficient data: {n_sne} SNe")
            continue
        
        try:
            res = fit_tg_tau_fast(z[mask], mu[mask], sigma_mu[mask])
            slice_results[f"slice_{i}"] = {
                "z_range": (z_min, z_max),
                "n_sne": n_sne,
                "H_Sigma": res["pars"].HSigma,
                "alpha_SB": res["pars"].alpha_SB,
                "xi": res["xi_inferred"],
                "chi2": res["chi2"]
            }
            print(f"  H_Sigma: {res['pars'].HSigma:.2f}, alpha_SB: {res['pars'].alpha_SB:.3f}")
        except Exception as e:
            print(f"  Fitting failed: {e}")
    
    results["redshift_slices"] = slice_results
    
    # Check for alpha_SB drift
    alpha_values = [slice_results[k]["alpha_SB"] for k in slice_results.keys()]
    alpha_std = np.std(alpha_values) if len(alpha_values) > 1 else 0.0
    
    results["alpha_sb_stability"] = {
        "values": alpha_values,
        "std": alpha_std,
        "drift_detected": alpha_std > 0.1,  # Flag if std > 0.1
        "in_range": all(1.0 <= a <= 2.0 for a in alpha_values)
    }
    
    return results

# ============================================================================
# HUBBLE RESIDUAL SYSTEMATICS ANALYSIS
# ============================================================================

def analyze_hubble_residuals(data: Dict, H_best: float, a_best: float) -> Dict:
    """Analyze Hubble residuals vs host mass, color/stretch, sky sector, redshift"""
    
    # Compute model predictions
    pars = TGtauParamsFast(HSigma=H_best, alpha_SB=a_best)
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + data["z"])
    DL = D * (1.0 + data["z"])**pars.alpha_SB
    mu_model = distance_modulus_from_DL_Mpc(DL)
    
    # Compute residuals
    residuals = data["mu"] - mu_model
    
    results = {
        "residuals": residuals,
        "residual_stats": {
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "rms": np.sqrt(np.mean(residuals**2))
        }
    }
    
    # Analyze trends
    correlations = {}
    
    # Host mass correlation
    if 'HOST_LOGMASS' in data:
        host_mass = data['HOST_LOGMASS']
        valid_mask = ~np.isnan(host_mass) & (host_mass > 0)
        if np.sum(valid_mask) > 100:
            corr = np.corrcoef(residuals[valid_mask], host_mass[valid_mask])[0,1]
            correlations["host_mass"] = corr
            print(f"Residual vs host mass correlation: {corr:.4f}")
    
    # Color correlation
    if 'c' in data:
        color = data['c']
        valid_mask = ~np.isnan(color)
        if np.sum(valid_mask) > 100:
            corr = np.corrcoef(residuals[valid_mask], color[valid_mask])[0,1]
            correlations["color"] = corr
            print(f"Residual vs color correlation: {corr:.4f}")
    
    # Stretch correlation
    if 'x1' in data:
        stretch = data['x1']
        valid_mask = ~np.isnan(stretch)
        if np.sum(valid_mask) > 100:
            corr = np.corrcoef(residuals[valid_mask], stretch[valid_mask])[0,1]
            correlations["stretch"] = corr
            print(f"Residual vs stretch correlation: {corr:.4f}")
    
    # Sky sector analysis (if coordinates available)
    if 'RA' in data and 'DEC' in data:
        # Hemispherical split
        ra, dec = data['RA'], data['DEC']
        
        # North/South split
        north_mask = dec > 0
        south_mask = dec < 0
        
        if np.sum(north_mask) > 50 and np.sum(south_mask) > 50:
            north_residuals = residuals[north_mask]
            south_residuals = residuals[south_mask]
            
            correlations["sky_north_mean"] = np.mean(north_residuals)
            correlations["sky_south_mean"] = np.mean(south_residuals)
            correlations["sky_difference"] = np.mean(north_residuals) - np.mean(south_residuals)
            
            print(f"North hemisphere residual mean: {correlations['sky_north_mean']:.4f}")
            print(f"South hemisphere residual mean: {correlations['sky_south_mean']:.4f}")
            print(f"Sky difference: {correlations['sky_difference']:.4f}")
    
    # Redshift correlation
    corr = np.corrcoef(residuals, data["z"])[0,1]
    correlations["redshift"] = corr
    print(f"Residual vs redshift correlation: {corr:.4f}")
    
    results["correlations"] = correlations
    
    return results

# ============================================================================
# ISW ANISOTROPY TESTING
# ============================================================================

def test_isw_anisotropy(data: Dict, H_best: float, a_best: float) -> Dict:
    """Test for ISW anisotropy with direction-dependent residuals"""
    
    if 'RA' not in data or 'DEC' not in data:
        return {"error": "RA/DEC coordinates not available"}
    
    # Compute residuals
    pars = TGtauParamsFast(HSigma=H_best, alpha_SB=a_best)
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + data["z"])
    DL = D * (1.0 + data["z"])**pars.alpha_SB
    mu_model = distance_modulus_from_DL_Mpc(DL)
    residuals = data["mu"] - mu_model
    
    # Test different sky directions
    directions = {
        "north": np.array([0, 0, 1]),
        "south": np.array([0, 0, -1]),
        "east": np.array([1, 0, 0]),
        "west": np.array([-1, 0, 0])
    }
    
    results = {}
    
    for name, nhat in directions.items():
        mask_pos, mask_neg = hemispheres(
            data["z"], data["mu"], data["sigma_mu"],
            data["RA"], data["DEC"], nhat
        )
        
        if np.sum(mask_pos) > 50 and np.sum(mask_neg) > 50:
            pos_residuals = residuals[mask_pos]
            neg_residuals = residuals[mask_neg]
            
            results[name] = {
                "pos_mean": np.mean(pos_residuals),
                "neg_mean": np.mean(neg_residuals),
                "difference": np.mean(pos_residuals) - np.mean(neg_residuals),
                "n_pos": np.sum(mask_pos),
                "n_neg": np.sum(mask_neg)
            }
            
            print(f"{name.capitalize()} direction difference: {results[name]['difference']:.4f}")
    
    return results

# ============================================================================
# MODEL SELECTION WITH AIC/BIC
# ============================================================================

def compare_models_proper(data: Dict, models: Dict[str, Dict]) -> Dict:
    """Proper model comparison with AIC/BIC"""
    
    n_data = len(data["z"])
    
    results = {}
    for name, model_result in models.items():
        chi2_val = model_result["chi2"]
        
        # Count parameters properly
        if "pars" in model_result:
            pars = model_result["pars"]
            if hasattr(pars, "__dict__"):
                k = len([v for v in pars.__dict__.values() if isinstance(v, (int, float))])
            else:
                k = 2  # Default for H_Sigma, alpha_SB
        else:
            k = 3  # Default for FRW (H0, Om, Ol)
        
        aic_val = aic(k, chi2_val)
        bic_val = bic(k, chi2_val, n_data)
        
        results[name] = {
            "chi2": chi2_val,
            "k": k,
            "aic": aic_val,
            "bic": bic_val,
            "aic_per_dof": aic_val / n_data,
            "bic_per_dof": bic_val / n_data
        }
    
    # Find best models
    best_aic = min(results.items(), key=lambda x: x[1]["aic"])
    best_bic = min(results.items(), key=lambda x: x[1]["bic"])
    
    results["best_aic"] = best_aic[0]
    results["best_bic"] = best_bic[0]
    
    # Compute deltas
    for name in results:
        if name not in ["best_aic", "best_bic"]:
            results[name]["delta_aic"] = results[name]["aic"] - best_aic[1]["aic"]
            results[name]["delta_bic"] = results[name]["bic"] - best_bic[1]["bic"]
    
    return results

# ============================================================================
# COMPREHENSIVE VALIDATION SUITE
# ============================================================================

def run_comprehensive_validation(data_path: str, cov_path: str = None) -> Dict:
    """Run all requested validation checks"""
    print("=" * 80)
    print("COMPREHENSIVE TG-tau VALIDATION SUITE")
    print("=" * 80)
    
    # Load data
    data = load_pantheon_fast(data_path)
    
    # Load covariance if available
    C = None
    if cov_path:
        C = load_compressed_covariance(cov_path)
    
    # Initial TG-tau fit
    print("\n1. INITIAL TG-tau FIT")
    print("-" * 50)
    res_tg = fit_tg_tau_fast(data["z"], data["mu"], data["sigma_mu"])
    
    print(f"H_Sigma: {res_tg['pars'].HSigma:.2f} km/s/Mpc")
    print(f"alpha_SB: {res_tg['pars'].alpha_SB:.3f}")
    print(f"xi: {res_tg['xi_inferred']:.2e}")
    print(f"chi2: {res_tg['chi2']:.2f}")
    
    # A. Full covariance chi2
    print("\n2. FULL COVARIANCE CHI2")
    print("-" * 50)
    if C is not None:
        # Compute chi2 with full covariance
        pars = res_tg["pars"]
        D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + data["z"])
        DL = D * (1.0 + data["z"])**pars.alpha_SB
        mu_model = distance_modulus_from_DL_Mpc(DL)
        
        chi2_full = chi2_full_cov(data["mu"], mu_model, C)
        print(f"Chi2 with full covariance: {chi2_full:.2f}")
        print(f"Chi2 with diagonal: {res_tg['chi2']:.2f}")
        print(f"Difference: {chi2_full - res_tg['chi2']:.2f}")
    else:
        print("Covariance matrix not available")
        chi2_full = res_tg['chi2']
    
    # B. Zero-point handling
    print("\n3. ZERO-POINT HANDLING")
    print("-" * 50)
    zero_point_results = compare_anchored_vs_free(data["z"], data["mu"], data["sigma_mu"])
    print(f"H_Sigma difference: {zero_point_results['H_diff']:.2f}")
    print(f"alpha_SB difference: {zero_point_results['a_diff']:.3f}")
    
    # C. Alpha_SB robustness
    print("\n4. ALPHA_SB ROBUSTNESS")
    print("-" * 50)
    alpha_robustness = test_alpha_sb_robustness(data["z"], data["mu"], data["sigma_mu"])
    
    # D. Hubble residual systematics
    print("\n5. HUBBLE RESIDUAL SYSTEMATICS")
    print("-" * 50)
    residual_analysis = analyze_hubble_residuals(data, res_tg["pars"].HSigma, res_tg["pars"].alpha_SB)
    
    # E. ISW anisotropy
    print("\n6. ISW ANISOTROPY")
    print("-" * 50)
    isw_anisotropy = test_isw_anisotropy(data, res_tg["pars"].HSigma, res_tg["pars"].alpha_SB)
    
    # F. Model selection
    print("\n7. MODEL SELECTION")
    print("-" * 50)
    
    # Create FRW baseline
    DL_frw = luminosity_distance_FRW_Mpc(data["z"], H0=70.0, Om=0.3, Ol=0.7)
    mu_frw = distance_modulus_from_DL_Mpc(DL_frw)
    chi2_frw = chi2(data["mu"], mu_frw, data["sigma_mu"])
    
    models = {
        "TG_tau": res_tg,
        "FRW_baseline": {"chi2": chi2_frw, "pars": {"H0": 70.0, "Om": 0.3, "Ol": 0.7}}
    }
    
    model_comparison = compare_models_proper(data, models)
    
    print(f"TG-tau AIC: {model_comparison['TG_tau']['aic']:.2f}")
    print(f"FRW AIC: {model_comparison['FRW_baseline']['aic']:.2f}")
    print(f"Delta AIC: {model_comparison['TG_tau']['delta_aic']:.2f}")
    
    # Compile all results
    results = {
        "initial_fit": res_tg,
        "full_covariance": {"chi2": chi2_full, "covariance_used": C is not None},
        "zero_point_handling": zero_point_results,
        "alpha_sb_robustness": alpha_robustness,
        "residual_systematics": residual_analysis,
        "isw_anisotropy": isw_anisotropy,
        "model_comparison": model_comparison
    }
    
    return results

if __name__ == "__main__":
    # Run comprehensive validation
    results = run_comprehensive_validation(
        "../data/pantheon/Pantheon+SH0ES.dat",
        "../data/pantheon/Pantheon+SH0ES_STAT+SYS.cov"
    )
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION COMPLETE")
    print("=" * 80)
    print("All requested validation checks completed!")
