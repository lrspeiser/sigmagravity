"""
Test time-coherence kernel on Milky Way.
"""

from __future__ import annotations

import numpy as np
import json
import pandas as pd
import argparse
from pathlib import Path
from coherence_time_kernel import compute_coherence_kernel

# Load fiducial parameters
_fiducial_path = Path(__file__).parent / "time_coherence_fiducial.json"
if _fiducial_path.exists():
    with open(_fiducial_path, "r") as f:
        _fiducial = json.load(f)
else:
    _fiducial = {"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0, "A_global": 1.0, "p": 0.757, "n_coh": 0.5}

# Try to load MW empirical kernel for comparison
MW_FIT_JSON = "gravitywavebaseline/theory_metric_resonance_mw_fit_improved.json"


def load_mw_profile(r_min: float = 12.0, r_max: float = 16.0):
    """
    Load MW baryonic profile from empirical fit data.
    
    Uses the empirical MW kernel fit to get R and g_bar.
    """
    # Try to load from empirical fit JSON
    if Path(MW_FIT_JSON).exists():
        mw_data = json.loads(Path(MW_FIT_JSON).read_text())
        # Extract R range from fit
        # For now, create a reasonable R grid
        R_kpc = np.linspace(r_min, r_max, 1000)
    else:
        # Fallback: create reasonable MW profile
        R_kpc = np.linspace(r_min, r_max, 1000)
    
    # Compute g_bar from circular velocity
    # MW: v_circ ~ 200 km/s at R ~ 8 kpc, decreasing to ~180 km/s at R ~ 20 kpc
    v_circ_kms = 200.0 * np.exp(-(R_kpc - 8.0) / 20.0)  # Rough approximation
    g_bar_kms2 = (v_circ_kms**2) / (R_kpc * 1e3)  # km/s²
    
    # Estimate density for tidal method
    # From g_bar, estimate rho: g ~ GM/R², so rho ~ g / (G R)
    G_msun_kpc_km2_s2 = 4.302e-6
    rho_bar_msun_pc3 = g_bar_kms2 / (G_msun_kpc_km2_s2 * R_kpc * 1e3) * 1e-9  # Convert to Msun/pc³
    
    return R_kpc, g_bar_kms2, rho_bar_msun_pc3


def main():
    parser = argparse.ArgumentParser(description='Test time-coherence kernel on Milky Way')
    parser.add_argument('--params-json', type=str, 
                       default='time-coherence/time_coherence_fiducial.json',
                       help='Path to parameters JSON file')
    parser.add_argument('--out-json', type=str, 
                       default='time-coherence/mw_coherence_test.json',
                       help='Path to output JSON file')
    args = parser.parse_args()
    
    # Load parameters
    params_path = Path(args.params_json)
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        print(f"Warning: {params_path} not found, using defaults")
        params = _fiducial
    
    print("Testing time-coherence kernel on Milky Way...")
    print("=" * 80)
    print(f"Using parameters from: {params_path}")
    print()
    
    R_kpc, g_bar_kms2, rho_bar_msun_pc3 = load_mw_profile(r_min=12.0, r_max=16.0)
    sigma_v_mw = 30.0  # km/s
    
    print(f"R range: {R_kpc.min():.2f} - {R_kpc.max():.2f} kpc")
    print(f"sigma_v: {sigma_v_mw} km/s")
    print()
    
    # Use fiducial parameters
    A_global = params.get("A_global", 1.0)
    p = params.get("p", 0.757)
    n_coh = params.get("n_coh", 0.5)
    alpha_length = params.get("alpha_length", 0.037)
    beta_sigma = params.get("beta_sigma", 1.5)
    backreaction_cap = params.get("backreaction_cap")
    tau_geom_method = params.get("tau_geom_method", "tidal")
    
    print(f"Parameters: A={A_global}, p={p}, n_coh={n_coh}, alpha={alpha_length}, beta={beta_sigma}")
    if backreaction_cap:
        print(f"Backreaction cap: K_max = {backreaction_cap}")
    print()
    
    # Compute kernel with fiducial parameters
    K = compute_coherence_kernel(
        R_kpc=R_kpc,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_mw,
        A_global=A_global,
        p=p,
        n_coh=n_coh,
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3 if tau_geom_method == "tidal" else None,
        tau_geom_method=tau_geom_method,
        alpha_length=alpha_length,
        beta_sigma=beta_sigma,
        backreaction_cap=backreaction_cap,
    )
    
    # Load observed velocities for RMS calculation
    try:
        mw_df = pd.read_parquet("gravitywavebaseline/gaia_with_gr_baseline.parquet")
        mask = (mw_df["R_kpc"] >= 12.0) & (mw_df["R_kpc"] <= 16.0)
        mw_slice = mw_df[mask].copy()
        
        if len(mw_slice) > 0:
            V_gr = mw_slice["V_gr"].values
            V_obs = mw_slice["V_obs"].values
            R_mw = mw_slice["R_kpc"].values
            
            # Interpolate K to R_mw
            K_interp = np.interp(R_mw, R_kpc, K)
            f_enh = 1.0 + K_interp
            V_model = V_gr * np.sqrt(np.clip(f_enh, 0.0, None))
            
            rms = np.sqrt(np.mean((V_model - V_obs)**2))
            rms_gr = np.sqrt(np.mean((V_gr - V_obs)**2))
            
            print(f"MW Performance:")
            print(f"  GR-only RMS: {rms_gr:.2f} km/s")
            print(f"  Time-coherence RMS: {rms:.2f} km/s")
            print(f"  Improvement: {rms_gr - rms:.2f} km/s")
    except Exception as e:
        print(f"Warning: Could not compute RMS: {e}")
        rms = None
        rms_gr = None
    
    # Compute coherence scales
    from coherence_time_kernel import (
        compute_tau_geom,
        compute_tau_noise,
        compute_tau_coh,
        compute_coherence_length,
    )
    
    tau_geom = compute_tau_geom(
        R_kpc, g_bar_kms2, rho_bar_msun_pc3 if tau_geom_method == "tidal" else None, 
        method=tau_geom_method
    )
    tau_noise = compute_tau_noise(R_kpc, sigma_v_mw, method="galaxy", beta_sigma=beta_sigma)
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    ell_coh = compute_coherence_length(tau_coh, alpha=alpha_length)
    
    # Store results
    results = {
        "parameters": params,
        "rms": rms,
        "rms_gr": rms_gr,
        "ell_coh_mean_kpc": float(np.mean(ell_coh)),
        "K_max": float(np.max(K)),
        "K_mean": float(np.mean(K)),
    }
    
    # Save to JSON
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

