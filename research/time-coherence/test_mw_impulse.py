"""
Milky Way: Orbit-by-orbit impulse test.

Tests if the K(Xi) relation works at the impulse level (g * T_orb * enhancement)
for individual stars, not just globally.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coherence_time_kernel import (
    compute_coherence_kernel,
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_exposure_factor,
)

G_MSUN_KPC_KM2_S2 = 4.302e-6
C_LIGHT_KMS = 299792.458


def impulse_factor_from_Xi(Xi, a, b=None):
    """
    Compute enhancement factor from exposure factor.
    
    Uses extra-time model: K = (a * Xi) / (1 - b * Xi)
    Or linear if b is None: K = a * Xi
    """
    Xi_clipped = np.clip(Xi, 0, None)
    
    if b is None:
        # Linear model
        return a * Xi_clipped
    else:
        # Extra-time model
        denominator = np.clip(1.0 - b * Xi_clipped, 1e-6, None)
        return (a * Xi_clipped) / denominator


def main():
    # Load MW data
    mw_data_path = Path("gravitywavebaseline/gaia_with_gr_baseline.parquet")
    if not mw_data_path.exists():
        print(f"Error: {mw_data_path} not found")
        print("  Run gravitywavebaseline scripts first to generate MW data")
        return

    print(f"Loading MW data from {mw_data_path}...")
    df = pd.read_parquet(mw_data_path)

    # Determine columns
    if {"g_obs", "g_bar"}.issubset(df.columns):
        g_obs = df["g_obs"].to_numpy()
        g_bar = df["g_bar"].to_numpy()
        R = df["R"].to_numpy() if "R" in df.columns else df["R_kpc"].to_numpy()
        v_gr = np.sqrt(g_bar * R * 1e3) / 1e3  # km/s
    elif {"v_phi", "v_phi_GR"}.issubset(df.columns):
        R = df["R"].to_numpy() if "R" in df.columns else df["R_kpc"].to_numpy()
        v_obs = df["v_phi"].to_numpy()
        v_gr = df["v_phi_GR"].to_numpy()
        g_obs = v_obs**2 / (R * 1e3)  # km/s^2
        g_bar = v_gr**2 / (R * 1e3)  # km/s^2
    else:
        print("Error: Need either (g_obs, g_bar) or (v_phi, v_phi_GR) columns")
        return

    # Filter valid data (focus on outer disk)
    mask = (
        (g_bar > 1e-8)
        & np.isfinite(g_obs)
        & np.isfinite(g_bar)
        & (R > 0)
        & (R >= 12.0)
        & (R <= 16.0)
    )
    
    if np.sum(mask) == 0:
        print("Error: No valid data points in 12-16 kpc range")
        return

    g_obs = g_obs[mask]
    g_bar = g_bar[mask]
    R = R[mask]
    v_gr = v_gr[mask] if "v_gr" in locals() else np.sqrt(g_bar * R * 1e3) / 1e3

    print(f"  {len(R)} stars with valid data in 12-16 kpc band")

    # Load parameters
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
            params = json.load(f)
    else:
        params = {
            "A_global": 1.0,
            "p": 0.757,
            "n_coh": 0.5,
            "alpha_length": 0.037,
            "beta_sigma": 1.5,
            "alpha_geom": 1.0,
            "backreaction_cap": 10.0,
            "tau_geom_method": "tidal",
        }

    # Load K(Xi) fit if available
    fit_path = Path("time-coherence/K_vs_Xi_fit.json")
    if fit_path.exists():
        with open(fit_path, "r") as f:
            fit_results = json.load(f)
        # Use best model (lowest RMS)
        if fit_results:
            best_model = min(fit_results.items(), key=lambda x: x[1]["rms"])
            model_name = best_model[0]
            fit_params = best_model[1]["params"]
            print(f"\nUsing K(Xi) fit: {model_name} with params {fit_params}")
            
            if model_name == "linear":
                a = fit_params[0]
                b = None
            elif model_name == "extra_time":
                a, b = fit_params[0], fit_params[1]
            else:
                # Power law - approximate as extra-time for now
                a, b = fit_params[0], 0.0
        else:
            # Default values
            a, b = 5.0, 1.0
            print("\nUsing default K(Xi) params: a=5.0, b=1.0")
    else:
        # Default values
        a, b = 5.0, 1.0
        print("\nUsing default K(Xi) params: a=5.0, b=1.0")

    # Compute orbital period
    R_km = R * 3.086e16  # km
    v_gr_ms = v_gr * 1e3  # m/s
    T_orb_sec = 2.0 * np.pi * R_km / v_gr_ms  # seconds

    # Estimate density
    rho_bar_msun_pc3 = g_bar / (G_MSUN_KPC_KM2_S2 * R * 1e3) * 1e-9
    sigma_v_kms = 30.0  # MW dispersion

    print("Computing K and Xi for each star...")

    # Process in batches
    batch_size = 1000
    K_obs = np.zeros_like(R)
    Xi = np.zeros_like(R)

    for i in range(0, len(R), batch_size):
        end_idx = min(i + batch_size, len(R))
        batch_R = R[i:end_idx]
        batch_g_bar = g_bar[i:end_idx]
        batch_rho = rho_bar_msun_pc3[i:end_idx]

        # Compute timescales
        tau_geom = compute_tau_geom(
            batch_R,
            batch_g_bar,
            batch_rho,
            method=params.get("tau_geom_method", "tidal"),
            alpha_geom=params.get("alpha_geom", 1.0),
        )

        tau_noise = compute_tau_noise(
            batch_R,
            sigma_v_kms,
            method="galaxy",
            beta_sigma=params.get("beta_sigma", 1.5),
        )

        tau_coh = compute_tau_coh(tau_geom, tau_noise)

        # Compute kernel
        K_obs[i:end_idx] = compute_coherence_kernel(
            R_kpc=batch_R,
            g_bar_kms2=batch_g_bar,
            sigma_v_kms=sigma_v_kms,
            A_global=params["A_global"],
            p=params["p"],
            n_coh=params["n_coh"],
            method="galaxy",
            rho_bar_msun_pc3=batch_rho,
            tau_geom_method=params.get("tau_geom_method", "tidal"),
            alpha_length=params["alpha_length"],
            beta_sigma=params["beta_sigma"],
            alpha_geom=params.get("alpha_geom", 1.0),
            backreaction_cap=params.get("backreaction_cap", 10.0),
        )

        # Compute exposure factor
        Xi[i:end_idx] = compute_exposure_factor(
            batch_R,
            batch_g_bar,
            tau_coh
        )

        if (i + batch_size) % 5000 == 0:
            print(f"  Processed {end_idx}/{len(R)} stars...")

    # Observed enhancement from kernel
    boost_obs = 1.0 + K_obs

    # Predicted enhancement from Xi using fitted relation
    K_pred_from_Xi = impulse_factor_from_Xi(Xi, a, b)
    boost_pred = 1.0 + K_pred_from_Xi

    # Also compute required boost from data
    boost_req = g_obs / g_bar

    # Compare
    good = np.isfinite(boost_obs) & np.isfinite(boost_pred) & np.isfinite(boost_req)

    if np.sum(good) < 10:
        print("Error: Insufficient valid data points")
        return

    boost_obs_good = boost_obs[good]
    boost_pred_good = boost_pred[good]
    boost_req_good = boost_req[good]
    K_obs_good = K_obs[good]
    K_pred_good = K_pred_from_Xi[good]
    Xi_good = Xi[good]

    # Statistics
    rms_boost = float(np.sqrt(np.mean((boost_obs_good - boost_pred_good) ** 2)))
    corr_boost = float(np.corrcoef(boost_obs_good, boost_pred_good)[0, 1]) if np.std(boost_obs_good) > 1e-6 else 0.0
    
    rms_K = float(np.sqrt(np.mean((K_obs_good - K_pred_good) ** 2)))
    corr_K = float(np.corrcoef(K_obs_good, K_pred_good)[0, 1]) if np.std(K_obs_good) > 1e-6 else 0.0

    # Save results
    result = {
        "n_points": int(np.sum(good)),
        "R_range_kpc": [float(R[good].min()), float(R[good].max())],
        "K_Xi_params": {"a": float(a), "b": float(b) if b is not None else None},
        "boost_level": {
            "rms_boost_obs_vs_pred": float(rms_boost),
            "corr_boost_obs_vs_pred": float(corr_boost),
            "mean_boost_obs": float(np.mean(boost_obs_good)),
            "mean_boost_pred": float(np.mean(boost_pred_good)),
            "mean_boost_req": float(np.mean(boost_req_good)),
        },
        "K_level": {
            "rms_K_obs_vs_pred": float(rms_K),
            "corr_K_obs_vs_pred": float(corr_K),
            "mean_K_obs": float(np.mean(K_obs_good)),
            "mean_K_pred": float(np.mean(K_pred_good)),
        },
        "Xi_stats": {
            "mean_Xi": float(np.mean(Xi_good)),
            "median_Xi": float(np.median(Xi_good)),
            "max_Xi": float(np.max(Xi_good)),
        },
    }

    outpath = Path("time-coherence/mw_impulse_test.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print("MILKY WAY IMPULSE-LEVEL TEST")
    print("=" * 80)
    print(f"\nResults saved to {outpath}")
    print(f"\nStatistics ({result['n_points']} stars in 12-16 kpc band):")
    print(f"\nBoost level (1 + K):")
    print(f"  RMS(boost_obs - boost_pred): {result['boost_level']['rms_boost_obs_vs_pred']:.4f}")
    print(f"  corr(boost_obs, boost_pred): {result['boost_level']['corr_boost_obs_vs_pred']:.3f}")
    print(f"  Mean boost_obs: {result['boost_level']['mean_boost_obs']:.3f}")
    print(f"  Mean boost_pred: {result['boost_level']['mean_boost_pred']:.3f}")
    print(f"  Mean boost_req: {result['boost_level']['mean_boost_req']:.3f}")
    
    print(f"\nK level:")
    print(f"  RMS(K_obs - K_pred): {result['K_level']['rms_K_obs_vs_pred']:.4f}")
    print(f"  corr(K_obs, K_pred): {result['K_level']['corr_K_obs_vs_pred']:.3f}")
    print(f"  Mean K_obs: {result['K_level']['mean_K_obs']:.3f}")
    print(f"  Mean K_pred: {result['K_level']['mean_K_pred']:.3f}")
    
    print(f"\nExposure factor:")
    print(f"  Mean Xi: {result['Xi_stats']['mean_Xi']:.3e}")
    print(f"  Median Xi: {result['Xi_stats']['median_Xi']:.3e}")
    print(f"  Max Xi: {result['Xi_stats']['max_Xi']:.3e}")

    if corr_boost > 0.9:
        print("\n[EXCELLENT] Very high correlation - K(Xi) relation works at impulse level!")
    elif corr_boost > 0.7:
        print("\n[GOOD] High correlation - K(Xi) relation works well")
    elif corr_boost > 0.5:
        print("\n[MODERATE] Moderate correlation - K(Xi) relation needs refinement")
    else:
        print("\n[WEAK] Low correlation - K(Xi) relation may not be universal")


if __name__ == "__main__":
    main()

