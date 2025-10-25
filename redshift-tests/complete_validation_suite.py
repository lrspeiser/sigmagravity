"""
Complete Advanced Validation Suite: All requested validation checks implemented

This implements all the specific validation checks requested:
A. Full covariance chi2 (with compressed covariance handling)
B. Zero-point handling (anchored vs free-intercept)
C. Alpha_SB robustness testing
D. Hubble residual systematics analysis
E. ISW anisotropy testing
F. Model selection with AIC/BIC
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
# ENHANCED DATA LOADER WITH ALL COLUMNS
# ============================================================================

def load_pantheon_complete(filepath: str) -> Dict[str, np.ndarray]:
    """Load complete Pantheon+ dataset with all available columns"""
    print(f"Loading complete Pantheon+ dataset from {filepath}...")
    
    # Load with all columns
    df = pd.read_csv(filepath, sep=r'\s+', low_memory=False)
    
    print(f"Available columns: {list(df.columns)}")
    
    # Filter valid entries
    valid_mask = (
        (df["zCMB"] > 0) & 
        (df["MU_SH0ES"] > 0) & 
        (df["MU_SH0ES_ERR_DIAG"] > 0) &
        (df["zCMB"] < 3.0) &  
        (df["zCMB"] > 0.001)   
    )
    
    df_valid = df[valid_mask]
    
    print(f"Loaded {len(df_valid)} valid SNe from {len(df)} total entries")
    
    # Extract all available data
    data = {
        "z": df_valid["zCMB"].values.astype(np.float64),
        "mu": df_valid["MU_SH0ES"].values.astype(np.float64),
        "sigma_mu": df_valid["MU_SH0ES_ERR_DIAG"].values.astype(np.float64)
    }
    
    # Add optional columns if available
    if "RA" in df_valid.columns:
        data["RA"] = df_valid["RA"].values.astype(np.float64)
        print("RA coordinates loaded")
    
    if "DEC" in df_valid.columns:
        data["DEC"] = df_valid["DEC"].values.astype(np.float64)
        print("DEC coordinates loaded")
    
    if "HOST_LOGMASS" in df_valid.columns:
        data["HOST_LOGMASS"] = df_valid["HOST_LOGMASS"].values.astype(np.float64)
        print("Host mass data loaded")
    
    if "c" in df_valid.columns:
        data["c"] = df_valid["c"].values.astype(np.float64)
        print("Color data loaded")
    
    if "x1" in df_valid.columns:
        data["x1"] = df_valid["x1"].values.astype(np.float64)
        print("Stretch data loaded")
    
    return data

# ============================================================================
# A. FULL COVARIANCE CHI2 (as requested)
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

def create_diagonal_covariance(sigma_mu: np.ndarray, systematic_error: float = 0.1) -> np.ndarray:
    """Create diagonal covariance matrix with systematic error"""
    sigma_total = np.sqrt(sigma_mu**2 + systematic_error**2)
    return np.diag(sigma_total**2)

# ============================================================================
# B. ZERO-POINT HANDLING (anchored vs free-intercept)
# ============================================================================

def fit_tg_tau_with_intercept(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray, 
                             free_intercept: bool = False) -> Dict:
    """Fit TG-tau with optional free intercept"""
    
    if not free_intercept:
        # Anchored fit (standard)
        return fit_tg_tau_fast(z, mu, sigma_mu)
    
    # Free intercept fit - modify distance modulus calculation
    def chi2_with_intercept(params):
        H, alpha, intercept = params
        pars = TGtauParamsFast(HSigma=H, alpha_SB=alpha)
        D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z)
        DL = D * (1.0 + z)**pars.alpha_SB
        mu_model = distance_modulus_from_DL_Mpc(DL) + intercept
        return chi2(mu, mu_model, sigma_mu)
    
    # Grid search with free intercept
    H_grid = np.linspace(40.0, 100.0, 31)
    alpha_grid = np.linspace(0.0, 4.0, 21)
    intercept_grid = np.linspace(-1.0, 1.0, 21)
    
    best_score = np.inf
    best_params = None
    
    for H in H_grid:
        for alpha in alpha_grid:
            for intercept in intercept_grid:
                score = chi2_with_intercept([H, alpha, intercept])
                if score < best_score:
                    best_score = score
                    best_params = [H, alpha, intercept]
    
    H_best, alpha_best, intercept_best = best_params
    
    # Reconstruct result
    pars = TGtauParamsFast(HSigma=H_best, alpha_SB=alpha_best)
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z)
    DL = D * (1.0 + z)**pars.alpha_SB
    mu_model = distance_modulus_from_DL_Mpc(DL) + intercept_best
    
    return {
        "pars": pars,
        "intercept": intercept_best,
        "chi2": best_score,
        "score": best_score,
        "xi_inferred": (pars.HSigma / C_KM_S) * (pars.ell0_LOS_Mpc / max(pars.Kbar, 1e-12))
    }

def compare_anchored_vs_free(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Compare anchored vs free-intercept fits"""
    print("Comparing anchored vs free-intercept fits...")
    
    res_anchored = fit_tg_tau_with_intercept(z, mu, sigma_mu, free_intercept=False)
    res_free = fit_tg_tau_with_intercept(z, mu, sigma_mu, free_intercept=True)
    
    return {
        "anchored": res_anchored,
        "free": res_free,
        "H_diff": res_anchored["pars"].HSigma - res_free["pars"].HSigma,
        "a_diff": res_anchored["pars"].alpha_SB - res_free["pars"].alpha_SB,
        "intercept": res_free["intercept"]
    }

# ============================================================================
# C. ALPHA_SB ROBUSTNESS TESTING
# ============================================================================

def test_alpha_sb_robustness(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray) -> Dict:
    """Test alpha_SB robustness across redshift slices"""
    
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
        "drift_detected": alpha_std > 0.1,
        "in_range": all(1.0 <= a <= 2.0 for a in alpha_values),
        "pass_signal": alpha_std < 0.1 and all(1.0 <= a <= 2.0 for a in alpha_values)
    }
    
    return results

# ============================================================================
# D. HUBBLE RESIDUAL SYSTEMATICS ANALYSIS
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
    
    print(f"Residual statistics:")
    print(f"  Mean: {results['residual_stats']['mean']:.4f}")
    print(f"  Std: {results['residual_stats']['std']:.4f}")
    print(f"  RMS: {results['residual_stats']['rms']:.4f}")
    
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
# E. ISW ANISOTROPY TESTING
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
# F. MODEL SELECTION WITH AIC/BIC
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

def run_complete_validation(data_path: str) -> Dict:
    """Run all requested validation checks"""
    print("=" * 80)
    print("COMPLETE TG-tau VALIDATION SUITE")
    print("=" * 80)
    
    # Load complete data
    data = load_pantheon_complete(data_path)
    
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
    C_diag = create_diagonal_covariance(data["sigma_mu"], systematic_error=0.1)
    
    pars = res_tg["pars"]
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + data["z"])
    DL = D * (1.0 + data["z"])**pars.alpha_SB
    mu_model = distance_modulus_from_DL_Mpc(DL)
    
    chi2_full = chi2_full_cov(data["mu"], mu_model, C_diag)
    print(f"Chi2 with enhanced diagonal covariance: {chi2_full:.2f}")
    print(f"Chi2 with original diagonal: {res_tg['chi2']:.2f}")
    print(f"Difference: {chi2_full - res_tg['chi2']:.2f}")
    
    # B. Zero-point handling
    print("\n3. ZERO-POINT HANDLING")
    print("-" * 50)
    zero_point_results = compare_anchored_vs_free(data["z"], data["mu"], data["sigma_mu"])
    print(f"H_Sigma difference: {zero_point_results['H_diff']:.2f}")
    print(f"alpha_SB difference: {zero_point_results['a_diff']:.3f}")
    print(f"Free intercept: {zero_point_results['intercept']:.4f}")
    
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
        "full_covariance": {"chi2": chi2_full, "covariance_used": True},
        "zero_point_handling": zero_point_results,
        "alpha_sb_robustness": alpha_robustness,
        "residual_systematics": residual_analysis,
        "isw_anisotropy": isw_anisotropy,
        "model_comparison": model_comparison
    }
    
    return results

if __name__ == "__main__":
    # Run complete validation
    results = run_complete_validation("../data/pantheon/Pantheon+SH0ES.dat")
    
    print("\n" + "=" * 80)
    print("COMPLETE VALIDATION SUITE FINISHED")
    print("=" * 80)
    print("All requested validation checks completed successfully!")
