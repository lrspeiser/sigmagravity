"""
Milky Way: Star-by-star exposure vs boost test.

Compares K_required (from g_obs/g_bar - 1) vs K_rough (from time-coherence kernel)
for individual Gaia stars.
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
        R = df["R_kpc"].to_numpy()
    elif {"v_phi", "v_phi_GR"}.issubset(df.columns):
        R = df["R"].to_numpy() if "R" in df.columns else df["R_kpc"].to_numpy()
        v_obs = df["v_phi"].to_numpy()
        v_gr = df["v_phi_GR"].to_numpy()
        g_obs = v_obs**2 / (R * 1e3)  # km/s^2
        g_bar = v_gr**2 / (R * 1e3)  # km/s^2
    else:
        print("Error: Need either (g_obs, g_bar) or (v_phi, v_phi_GR) columns")
        return

    # Filter valid data
    mask = (g_bar > 1e-8) & np.isfinite(g_obs) & np.isfinite(g_bar) & (R > 0)
    g_obs = g_obs[mask]
    g_bar = g_bar[mask]
    R = R[mask]

    print(f"  {len(R)} stars with valid data")

    # Required boost
    K_req = g_obs / g_bar - 1.0

    # σ_v: use MW dispersion or per-star if present
    if "sigma_v" in df.columns:
        sigma_v = df["sigma_v"].to_numpy()[mask]
    else:
        sigma_v = np.full_like(R, 30.0)  # km/s

    # Circular speed for coherence
    v_circ_kms = np.sqrt(g_bar * R * 1e3)  # km/s
    
    # Estimate density
    rho_bar_msun_pc3 = g_bar / (G_MSUN_KPC_KM2_S2 * R * 1e3) * 1e-9

    # Load parameters
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
            params = json.load(f)
    else:
        print(f"Warning: {fiducial_path} not found, using defaults")
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

    print("Computing K_rough and Xi for each star...")
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    K_rough = np.zeros_like(R)
    Xi = np.zeros_like(R)

    for i in range(0, len(R), batch_size):
        end_idx = min(i + batch_size, len(R))
        batch_R = R[i:end_idx]
        batch_g_bar = g_bar[i:end_idx]
        batch_rho = rho_bar_msun_pc3[i:end_idx]
        batch_sigma_v = sigma_v[i:end_idx]

        for j, (R_j, g_j, rho_j, sig_j) in enumerate(
            zip(batch_R, batch_g_bar, batch_rho, batch_sigma_v)
        ):
            idx = i + j
            
            # Compute timescales
            tau_geom = compute_tau_geom(
                np.array([R_j]),
                np.array([g_j]),
                np.array([rho_j]),
                method=params.get("tau_geom_method", "tidal"),
                alpha_geom=params.get("alpha_geom", 1.0),
            )[0]

            tau_noise = compute_tau_noise(
                np.array([R_j]),
                sig_j,
                method="galaxy",
                beta_sigma=params.get("beta_sigma", 1.5),
            )[0]

            tau_coh = compute_tau_coh(
                np.array([tau_geom]),
                np.array([tau_noise])
            )[0]

            # Compute kernel
            K_rough[idx] = compute_coherence_kernel(
                R_kpc=np.array([R_j]),
                g_bar_kms2=np.array([g_j]),
                sigma_v_kms=sig_j,
                A_global=params["A_global"],
                p=params["p"],
                n_coh=params["n_coh"],
                method="galaxy",
                rho_bar_msun_pc3=np.array([rho_j]),
                tau_geom_method=params.get("tau_geom_method", "tidal"),
                alpha_length=params["alpha_length"],
                beta_sigma=params["beta_sigma"],
                alpha_geom=params.get("alpha_geom", 1.0),
                backreaction_cap=params.get("backreaction_cap", 10.0),
            )[0]

            # Compute exposure factor
            Xi[idx] = compute_exposure_factor(
                np.array([R_j]),
                np.array([g_j]),
                np.array([tau_coh])
            )[0]

        if (i + batch_size) % 5000 == 0:
            print(f"  Processed {end_idx}/{len(R)} stars...")

    # Compare where both are defined
    good = (
        np.isfinite(K_rough)
        & np.isfinite(K_req)
        & (K_req >= -0.5)
        & (R >= 12.0)
        & (R <= 16.0)  # Focus on outer disk
    )

    if not np.any(good):
        print("Error: No valid data points")
        return

    K_req_good = K_req[good]
    K_rough_good = K_rough[good]
    Xi_good = Xi[good]
    R_good = R[good]

    # Compute statistics
    if np.std(K_req_good) < 1e-6 or np.std(K_rough_good) < 1e-6:
        corr = 0.0
    else:
        corr = float(np.corrcoef(K_req_good, K_rough_good)[0, 1])

    rms_diff = float(np.sqrt(np.mean((K_req_good - K_rough_good) ** 2)))
    mean_abs_diff = float(np.mean(np.abs(K_req_good - K_rough_good)))
    mean_K_req = float(np.mean(np.abs(K_req_good)))
    rel_error = mean_abs_diff / max(mean_K_req, 1e-6)

    out = {
        "n_points": int(good.sum()),
        "R_range_kpc": [float(R_good.min()), float(R_good.max())],
        "corr_Kreq_Krough": corr,
        "rms_diff_K": rms_diff,
        "mean_abs_diff_K": mean_abs_diff,
        "rel_error": rel_error,
        "Kreq_mean": float(np.mean(K_req_good)),
        "Kreq_median": float(np.median(K_req_good)),
        "Kreq_std": float(np.std(K_req_good)),
        "Krough_mean": float(np.mean(K_rough_good)),
        "Krough_median": float(np.median(K_rough_good)),
        "Krough_std": float(np.std(K_rough_good)),
        "Xi_mean": float(np.mean(Xi_good)),
        "Xi_median": float(np.median(Xi_good)),
        "Xi_max": float(np.max(Xi_good)),
    }

    outpath = Path("time-coherence/mw_roughness_vs_required.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 80}")
    print("MILKY WAY ROUGHNESS VS REQUIRED BOOST")
    print(f"{'=' * 80}")
    print(f"\nResults saved to {outpath}")
    print(f"\nStatistics ({out['n_points']} stars in 12-16 kpc band):")
    print(f"  Correlation: {out['corr_Kreq_Krough']:.3f}")
    print(f"  RMS difference: {out['rms_diff_K']:.3f}")
    print(f"  Mean absolute difference: {out['mean_abs_diff_K']:.3f}")
    print(f"  Relative error: {out['rel_error']:.2%}")
    print(f"\n  Mean K_req: {out['Kreq_mean']:.3f} ± {out['Kreq_std']:.3f}")
    print(f"  Mean K_rough: {out['Krough_mean']:.3f} ± {out['Krough_std']:.3f}")
    print(f"  Mean Xi: {out['Xi_mean']:.3e} (max: {out['Xi_max']:.3e})")

    if corr > 0.7:
        print("\n[STRONG] High correlation suggests roughness picture is correct")
    elif corr > 0.5:
        print("\n[GOOD] Moderate correlation supports roughness picture")
    else:
        print("\n[WEAK] Low correlation - may need refinement")


if __name__ == "__main__":
    main()

