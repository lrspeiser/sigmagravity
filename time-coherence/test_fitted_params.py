"""Test fitted hyperparameters on full SPARC dataset."""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from coherence_time_kernel import compute_coherence_kernel


def load_rotmod(path: str) -> pd.DataFrame:
    """Load SPARC rotmod file."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
    v_gr = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    df["V_gr"] = v_gr
    return df[["R_kpc", "V_obs", "V_gr"]]


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def process_galaxy(path: str, sigma_v_kms: float, params: dict) -> dict | None:
    """Process a single galaxy with fitted parameters."""
    galaxy = Path(path).name.replace("_rotmod.dat", "")
    
    try:
        df = load_rotmod(path)
    except Exception:
        return None
    
    R = df["R_kpc"].to_numpy(float)
    V_obs = df["V_obs"].to_numpy(float)
    V_gr = df["V_gr"].to_numpy(float)
    
    if len(R) < 4:
        return None
    
    # Compute g_bar from V_gr
    g_bar_kms2 = (V_gr**2) / (R * 1e3)  # km/s²
    
    # Estimate density
    G_msun_kpc_km2_s2 = 4.302e-6
    rho_bar_msun_pc3 = g_bar_kms2 / (G_msun_kpc_km2_s2 * R * 1e3) * 1e-9
    
    # Compute coherence kernel with fitted params
    K = compute_coherence_kernel(
        R_kpc=R,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_kms,
        A_global=params["A_global"],
        p=params["p"],
        n_coh=params["n_coh"],
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3,
        delta_R_kpc=params["delta_R_kpc"],
        tau_geom_method="tidal",
        alpha_length=0.037,
        beta_sigma=1.5,
    )
    
    # Compute coherence scales
    from coherence_time_kernel import (
        compute_tau_geom,
        compute_tau_noise,
        compute_tau_coh,
        compute_coherence_length,
    )
    
    tau_geom = compute_tau_geom(
        R, g_bar_kms2, rho_bar_msun_pc3, delta_R_kpc=params["delta_R_kpc"], method="tidal"
    )
    tau_noise = compute_tau_noise(R, sigma_v_kms, method="galaxy", beta_sigma=1.5)
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    ell_coh = compute_coherence_length(tau_coh, alpha=0.037)
    
    # Apply enhancement
    f_enh = 1.0 + K
    V_model = V_gr * np.sqrt(np.clip(f_enh, 0.0, None))
    
    # Compute RMS
    rms_gr = rms(V_obs - V_gr)
    rms_coherence = rms(V_obs - V_model)
    
    return {
        "galaxy": galaxy,
        "sigma_v_kms": sigma_v_kms,
        "n_points": len(R),
        "rms_gr": rms_gr,
        "rms_coherence": rms_coherence,
        "delta_rms": rms_coherence - rms_gr,
        "K_mean": float(np.mean(K)),
        "K_max": float(np.max(K)),
        "ell_coh_mean_kpc": float(np.mean(ell_coh)),
        "tau_coh_mean_yr": float(np.mean(tau_coh) / (365.25 * 86400)),
    }


def main():
    # Load fitted parameters
    fit_json = Path("time-coherence/time_coherence_fit_hyperparams.json")
    if not fit_json.exists():
        print(f"Error: {fit_json} not found. Run fit_time_coherence_hyperparams.py first.")
        return
    
    params = json.loads(fit_json.read_text())
    print("=" * 80)
    print("TESTING FITTED PARAMETERS ON FULL SPARC DATASET")
    print("=" * 80)
    print(f"\nFitted parameters:")
    print(f"  A_global: {params['A_global']:.6g}")
    print(f"  p: {params['p']:.6g}")
    print(f"  n_coh: {params['n_coh']:.6g}")
    print(f"  delta_R_kpc: {params['delta_R_kpc']:.6g} kpc")
    
    # Load SPARC data
    rotmod_dir = "data/Rotmod_LTG"
    summary_csv = "data/sparc/sparc_combined.csv"
    
    summary = pd.read_csv(summary_csv)
    sigma_map = dict(
        zip(
            summary["galaxy_name"].astype(str),
            summary["sigma_velocity"].astype(float),
        )
    )
    
    rotmod_paths = sorted(glob.glob(os.path.join(rotmod_dir, "*_rotmod.dat")))
    print(f"\nProcessing {len(rotmod_paths)} SPARC galaxies...")
    
    results = []
    for i, path in enumerate(rotmod_paths):
        galaxy = Path(path).name.replace("_rotmod.dat", "")
        sigma_v = sigma_map.get(galaxy, 25.0)
        
        result = process_galaxy(path, sigma_v, params)
        if result:
            results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rotmod_paths)} galaxies...")
    
    df = pd.DataFrame(results)
    output_csv = "time-coherence/sparc_coherence_fitted_params.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults:")
    print(f"  Total galaxies: {len(df)}")
    print(f"  Mean delta_rms: {df['delta_rms'].mean():.3f} km/s")
    print(f"  Median delta_rms: {df['delta_rms'].median():.3f} km/s")
    print(f"  Improved: {(df['delta_rms'] < 0).sum()}/{len(df)} ({(df['delta_rms'] < 0).sum()/len(df)*100:.1f}%)")
    print(f"  Mean ell_coh: {df['ell_coh_mean_kpc'].mean():.2f} kpc")
    print(f"  Median ell_coh: {df['ell_coh_mean_kpc'].median():.2f} kpc")
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()

