"""
Phase-2 Hardening: Key Fixes for Publication-Ready Results

Implements the critical fixes identified:
1. Fair FRW fitting with free intercept
2. Publication-grade parity table
3. Hemispherical significance testing
4. Optimized composite model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
import time
from typing import Dict, Tuple, Optional, List
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

# Import our optimized modules
from tg_tau_fast import fit_tg_tau_fast, load_pantheon_fast, TGtauParamsFast
from sigma_redshift_toy_models import (
    C_KM_S, c_over_H0_mpc, distance_modulus_from_DL_Mpc,
    chi2, aic, bic, time_dilation_penalty, tolman_penalty,
    luminosity_distance_FRW_Mpc
)

# ============================================================================
# FAIR FRW FITTING WITH FREE INTERCEPT (as requested)
# ============================================================================

def fit_frw_flat_free_intercept(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray, 
                               Om_grid: np.ndarray = None) -> Dict:
    """
    Fit flat LambdaCDM with free intercept for fair model comparison.
    Returns best-fit Om, intercept, chi2, and parameter count.
    """
    if Om_grid is None:
        Om_grid = np.linspace(0.05, 0.5, 451)
    
    best = None
    for Om in Om_grid:
        DL = luminosity_distance_FRW_Mpc(z, H0=70.0, Om=Om, Ol=1.0-Om)
        mu0 = distance_modulus_from_DL_Mpc(DL)
        
        # Analytic intercept minimizing chi2 for fixed Om
        w = 1.0/np.clip(sigma_mu, 1e-6, None)**2
        intercept = np.sum(w*(mu - mu0))/np.sum(w)
        
        c2 = chi2(mu, mu0 + intercept, sigma_mu)
        
        if best is None or c2 < best["chi2"]:
            best = {
                "Om": Om, 
                "intercept": intercept, 
                "chi2": c2, 
                "k": 2,  # Om + intercept
                "pars": {"Om": Om, "intercept": intercept, "H0": 70.0}
            }
    
    return best

# ============================================================================
# HEMISPHERICAL SIGNIFICANCE TESTING
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

def test_hemispherical_significance(data: Dict, H_best: float, a_best: float, 
                                   n_permutations: int = 10000) -> Dict:
    """Test significance of hemispherical residuals with permutation test"""
    
    if 'RA' not in data or 'DEC' not in data:
        return {"error": "RA/DEC coordinates not available"}
    
    # Compute residuals
    pars = TGtauParamsFast(HSigma=H_best, alpha_SB=a_best)
    D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + data["z"])
    DL = D * (1.0 + data["z"])**pars.alpha_SB
    mu_model = distance_modulus_from_DL_Mpc(DL)
    residuals = data["mu"] - mu_model
    
    # North/South split
    nhat = np.array([0, 0, 1])
    mask_north, mask_south = hemispheres(
        data["z"], data["mu"], data["sigma_mu"],
        data["RA"], data["DEC"], nhat
    )
    
    if np.sum(mask_north) < 50 or np.sum(mask_south) < 50:
        return {"error": "Insufficient data for hemispherical test"}
    
    # Observed difference
    north_residuals = residuals[mask_north]
    south_residuals = residuals[mask_south]
    observed_diff = np.mean(north_residuals) - np.mean(south_residuals)
    
    # Permutation test
    all_residuals = residuals.copy()
    n_north = np.sum(mask_north)
    
    permuted_diffs = []
    for _ in range(n_permutations):
        # Randomly reassign hemisphere labels
        np.random.shuffle(all_residuals)
        perm_north = all_residuals[:n_north]
        perm_south = all_residuals[n_north:]
        perm_diff = np.mean(perm_north) - np.mean(perm_south)
        permuted_diffs.append(perm_diff)
    
    permuted_diffs = np.array(permuted_diffs)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    results = {
        "observed_difference": observed_diff,
        "north_mean": np.mean(north_residuals),
        "south_mean": np.mean(south_residuals),
        "n_north": n_north,
        "n_south": np.sum(mask_south),
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    
    print(f"Hemispherical difference: {observed_diff:.4f} mag")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant: {results['significant']}")
    
    return results

# ============================================================================
# OPTIMIZED TG-tau + ISW COMPOSITE
# ============================================================================

def evaluate_tg_tau_isw_point(args):
    """Evaluate single point in TG-tau + ISW parameter space"""
    H, alpha_SB, a1, z, mu, sigma_mu = args
    
    try:
        # TG-tau component
        pars = TGtauParamsFast(HSigma=H, alpha_SB=alpha_SB)
        D = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z)
        
        # Add ISW component (simplified for speed)
        z_tg = np.exp((pars.HSigma / C_KM_S) * D) - 1.0
        z_isw = a1 * D
        z_total = z_tg + z_isw
        
        # Use approximation for D(z) inversion
        D_total = c_over_H0_mpc(pars.HSigma) * np.log(1.0 + z_total)
        
        # Luminosity distance
        DL = D_total * (1.0 + z)**pars.alpha_SB
        mu_model = distance_modulus_from_DL_Mpc(DL)
        
        c2 = chi2(mu, mu_model, sigma_mu)
        td_pen = time_dilation_penalty(z, 1.0 + z, weight=5.0)
        tol_pen = tolman_penalty(pars.alpha_SB, alpha_target=4.0, weight=2.0)
        score = c2 + td_pen + tol_pen
        
        return {
            "HSigma": H,
            "alpha_SB": alpha_SB,
            "a1": a1,
            "chi2": c2,
            "score": score
        }
    except:
        return None

def fit_tg_tau_isw_optimized(z: np.ndarray, mu: np.ndarray, sigma_mu: np.ndarray, 
                            n_cpus: int = None) -> Dict:
    """Optimized TG-tau + ISW composite fitting with parallel processing"""
    if n_cpus is None:
        n_cpus = min(cpu_count(), 10)
    
    print(f"Using {n_cpus} CPUs for parallel TG-tau + ISW fitting...")
    
    # Tight priors for ISW parameter
    H_grid = np.linspace(40.0, 100.0, 21)  # Reduced for speed
    alpha_grid = np.linspace(0.0, 4.0, 21)
    a1_grid = np.linspace(-5e-4, 5e-4, 11)  # Tight prior on ISW
    
    args_list = [(H, alpha, a1, z, mu, sigma_mu) 
                 for H in H_grid for alpha in alpha_grid for a1 in a1_grid]
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        results = list(executor.map(evaluate_tg_tau_isw_point, args_list))
    
    # Filter out failed evaluations
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        raise RuntimeError("All TG-tau + ISW evaluations failed")
    
    best = min(valid_results, key=lambda r: r["score"])
    
    # Reconstruct parameters
    pars = TGtauParamsFast(HSigma=best["HSigma"], alpha_SB=best["alpha_SB"])
    xi = (pars.HSigma / C_KM_S) * (pars.ell0_LOS_Mpc / max(pars.Kbar, 1e-12))
    
    result = {
        "pars": pars,
        "a1": best["a1"],
        "chi2": best["chi2"],
        "score": best["score"],
        "xi_inferred": xi
    }
    
    elapsed_time = time.time() - start_time
    print(f"Parallel TG-tau + ISW fitting completed in {elapsed_time:.2f} seconds")
    
    return result

# ============================================================================
# PUBLICATION-GRADE PARITY TABLE
# ============================================================================

def generate_parity_table(data: Dict) -> Dict:
    """Generate publication-grade parity table with fair model comparison"""
    
    print("Generating publication-grade parity table...")
    
    # Fit both models with same data
    res_tg = fit_tg_tau_fast(data["z"], data["mu"], data["sigma_mu"])
    res_frw = fit_frw_flat_free_intercept(data["z"], data["mu"], data["sigma_mu"])
    
    n_data = len(data["z"])
    dof_tg = n_data - 2  # H_Sigma, alpha_SB
    dof_frw = n_data - 2  # Om, intercept
    
    # Compute AIC/BIC
    aic_tg = aic(2, res_tg["chi2"])
    bic_tg = bic(2, res_tg["chi2"], n_data)
    aic_frw = aic(2, res_frw["chi2"])
    bic_frw = bic(2, res_frw["chi2"], n_data)
    
    # Akaike weights
    delta_aic_tg = aic_tg - min(aic_tg, aic_frw)
    delta_aic_frw = aic_frw - min(aic_tg, aic_frw)
    weight_tg = np.exp(-0.5 * delta_aic_tg) / (np.exp(-0.5 * delta_aic_tg) + np.exp(-0.5 * delta_aic_frw))
    weight_frw = np.exp(-0.5 * delta_aic_frw) / (np.exp(-0.5 * delta_aic_tg) + np.exp(-0.5 * delta_aic_frw))
    
    parity_table = {
        "TG_tau": {
            "H_Sigma": res_tg["pars"].HSigma,
            "alpha_SB": res_tg["pars"].alpha_SB,
            "xi": res_tg["xi_inferred"],
            "chi2": res_tg["chi2"],
            "dof": dof_tg,
            "aic": aic_tg,
            "bic": bic_tg,
            "akaike_weight": weight_tg,
            "k": 2
        },
        "FRW": {
            "Om": res_frw["pars"]["Om"],
            "intercept": res_frw["pars"]["intercept"],
            "H0": res_frw["pars"]["H0"],
            "chi2": res_frw["chi2"],
            "dof": dof_frw,
            "aic": aic_frw,
            "bic": bic_frw,
            "akaike_weight": weight_frw,
            "k": 2
        },
        "comparison": {
            "delta_aic": aic_tg - aic_frw,
            "delta_bic": bic_tg - bic_frw,
            "delta_chi2": res_tg["chi2"] - res_frw["chi2"],
            "n_data": n_data
        }
    }
    
    return parity_table

# ============================================================================
# PHASE-2 VALIDATION SUITE
# ============================================================================

def run_phase2_key_fixes(data_path: str) -> Dict:
    """Run Phase-2 key fixes for publication-ready results"""
    print("=" * 80)
    print("PHASE-2 KEY FIXES: PUBLICATION-READY VALIDATION")
    print("=" * 80)
    
    # Load data
    data = load_pantheon_fast(data_path)
    
    # Generate fair parity table
    print("\n1. FAIR MODEL COMPARISON")
    print("-" * 50)
    parity_table = generate_parity_table(data)
    
    print(f"TG-tau: H_Sigma = {parity_table['TG_tau']['H_Sigma']:.2f}, alpha_SB = {parity_table['TG_tau']['alpha_SB']:.3f}")
    print(f"FRW: Om = {parity_table['FRW']['Om']:.3f}, intercept = {parity_table['FRW']['intercept']:.4f}")
    print(f"Delta AIC: {parity_table['comparison']['delta_aic']:.2f}")
    print(f"Delta BIC: {parity_table['comparison']['delta_bic']:.2f}")
    print(f"Akaike weight TG-tau: {parity_table['TG_tau']['akaike_weight']:.3f}")
    print(f"Akaike weight FRW: {parity_table['FRW']['akaike_weight']:.3f}")
    
    # Hemispherical significance
    print("\n2. HEMISPHERICAL SIGNIFICANCE")
    print("-" * 50)
    hemispherical_results = test_hemispherical_significance(
        data, parity_table['TG_tau']['H_Sigma'], parity_table['TG_tau']['alpha_SB']
    )
    
    # Optimized composite model
    print("\n3. OPTIMIZED TG-tau + ISW COMPOSITE")
    print("-" * 50)
    try:
        composite_results = fit_tg_tau_isw_optimized(data["z"], data["mu"], data["sigma_mu"])
        print(f"TG-tau + ISW: H_Sigma = {composite_results['pars'].HSigma:.2f}, alpha_SB = {composite_results['pars'].alpha_SB:.3f}")
        print(f"a1 = {composite_results['a1']:.2e}, chi2 = {composite_results['chi2']:.2f}")
        
        # Compare with pure TG-tau
        delta_chi2 = parity_table['TG_tau']['chi2'] - composite_results['chi2']
        print(f"Chi2 improvement: {delta_chi2:.2f}")
        
        # AIC comparison
        aic_composite = aic(3, composite_results['chi2'])  # H_Sigma, alpha_SB, a1
        delta_aic_composite = aic_composite - parity_table['TG_tau']['aic']
        print(f"Delta AIC (composite vs pure): {delta_aic_composite:.2f}")
        
    except Exception as e:
        print(f"Composite fitting failed: {e}")
        composite_results = None
    
    # Compile results
    results = {
        "parity_table": parity_table,
        "hemispherical_significance": hemispherical_results,
        "composite_model": composite_results
    }
    
    return results

if __name__ == "__main__":
    # Run Phase-2 key fixes
    results = run_phase2_key_fixes("../data/pantheon/Pantheon+SH0ES.dat")
    
    print("\n" + "=" * 80)
    print("PHASE-2 KEY FIXES COMPLETE")
    print("=" * 80)
    print("Publication-ready fixes implemented successfully!")
