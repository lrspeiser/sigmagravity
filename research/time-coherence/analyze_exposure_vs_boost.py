"""
Analyze exposure factor Xi(R) = tau_coh / T_orb vs kernel K(R) and mass discrepancy.

Tests the "rough spacetime => extra time in the field" picture directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coherence_time_kernel import (
    compute_coherence_kernel,
    compute_tau_geom,
    compute_tau_noise,
    compute_tau_coh,
    compute_coherence_length,
    compute_exposure_factor,
)

FIDUCIAL_PATH = Path("time-coherence/time_coherence_fiducial.json")


def load_fiducial():
    if FIDUCIAL_PATH.exists():
        with open(FIDUCIAL_PATH, "r") as f:
            return json.load(f)
    # Fallback
    return {
        "A_global": 1.0,
        "p": 0.757,
        "n_coh": 0.5,
        "alpha_length": 0.037,
        "beta_sigma": 1.5,
        "alpha_geom": 1.0,
        "backreaction_cap": 10.0,
        "tau_geom_method": "tidal",
    }


def analyze_mw_exposure():
    """
    MW: Xi(R) vs K(R) in the 12–16 kpc band, plus global exposure vs RMS gain.
    """
    from test_mw_coherence import load_mw_profile

    params = load_fiducial()

    R_kpc, g_bar_kms2, rho_bar_msun_pc3 = load_mw_profile(12.0, 16.0)
    sigma_v_mw = 30.0

    K = compute_coherence_kernel(
        R_kpc=R_kpc,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v_mw,
        A_global=params["A_global"],
        p=params["p"],
        n_coh=params["n_coh"],
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3
        if params.get("tau_geom_method", "tidal") == "tidal"
        else None,
        tau_geom_method=params.get("tau_geom_method", "tidal"),
        alpha_length=params["alpha_length"],
        beta_sigma=params["beta_sigma"],
        alpha_geom=params.get("alpha_geom", 1.0),
        backreaction_cap=params.get("backreaction_cap", 10.0),
    )

    tau_geom = compute_tau_geom(
        R_kpc,
        g_bar_kms2,
        rho_bar_msun_pc3,
        method=params.get("tau_geom_method", "tidal"),
        alpha_geom=params.get("alpha_geom", 1.0),
    )
    tau_noise = compute_tau_noise(
        R_kpc,
        sigma_v_mw,
        method="galaxy",
        beta_sigma=params["beta_sigma"],
    )
    tau_coh = compute_tau_coh(tau_geom, tau_noise)
    ell_coh = compute_coherence_length(tau_coh, alpha=params["alpha_length"])

    Xi = compute_exposure_factor(R_kpc, g_bar_kms2, tau_coh)

    # Simple global diagnostics
    corr_K_Xi = float(np.corrcoef(K, Xi)[0, 1])
    exposure_mean = float(np.mean(Xi))
    exposure_peak = float(np.max(Xi))

    out = {
        "R_kpc": R_kpc.tolist(),
        "K": K.tolist(),
        "Xi": Xi.tolist(),
        "ell_coh_kpc": ell_coh.tolist(),
        "corr_K_Xi": corr_K_Xi,
        "Xi_mean": exposure_mean,
        "Xi_max": exposure_peak,
    }

    outpath = Path("time-coherence/mw_exposure_profile.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    
    print("=" * 80)
    print("MILKY WAY EXPOSURE ANALYSIS")
    print("=" * 80)
    print(f"MW exposure profile saved to {outpath}")
    print(f"corr(K, Xi) = {corr_K_Xi:.3f}")
    print(f"Xi_mean = {exposure_mean:.3e}")
    print(f"Xi_max = {exposure_peak:.3e}")
    print(f"Mean K = {np.mean(K):.3f}")
    print(f"Mean ell_coh = {np.mean(ell_coh):.3f} kpc")


def analyze_sparc_exposure():
    """
    Across SPARC: does avg exposure predict mass discrepancy?
    """
    from test_sparc_coherence import load_rotmod

    params = load_fiducial()

    summary_csv = Path("data/sparc/sparc_combined.csv")
    rotmod_dir = Path("data/Rotmod_LTG")

    if not summary_csv.exists():
        print(f"Warning: {summary_csv} not found, skipping SPARC exposure analysis")
        return

    if not rotmod_dir.exists():
        print(f"Warning: {rotmod_dir} not found, skipping SPARC exposure analysis")
        return

    summary = pd.read_csv(summary_csv)
    sigma_map = dict(
        zip(summary["galaxy_name"].astype(str), summary["sigma_velocity"].astype(float))
    )

    rows = []

    rotmod_paths = sorted(rotmod_dir.glob("*_rotmod.dat"))
    print(f"\nProcessing {len(rotmod_paths)} SPARC galaxies for exposure analysis...")

    for i, rotmod_path in enumerate(rotmod_paths):
        galaxy = rotmod_path.name.replace("_rotmod.dat", "")

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(rotmod_paths)} galaxies...")

        try:
            df = load_rotmod(str(rotmod_path))
        except Exception:
            continue

        R = df["R_kpc"].to_numpy(float)
        V_obs = df["V_obs"].to_numpy(float)
        V_gr = df["V_gr"].to_numpy(float)

        if len(R) < 4:
            continue

        g_bar_kms2 = (V_gr**2) / (R * 1e3)
        G = 4.302e-6
        rho_bar_msun_pc3 = g_bar_kms2 / (G * R * 1e3) * 1e-9

        sigma_v = sigma_map.get(galaxy, 25.0)

        K = compute_coherence_kernel(
            R_kpc=R,
            g_bar_kms2=g_bar_kms2,
            sigma_v_kms=sigma_v,
            A_global=params["A_global"],
            p=params["p"],
            n_coh=params["n_coh"],
            method="galaxy",
            rho_bar_msun_pc3=rho_bar_msun_pc3
            if params.get("tau_geom_method", "tidal") == "tidal"
            else None,
            tau_geom_method=params.get("tau_geom_method", "tidal"),
            alpha_length=params["alpha_length"],
            beta_sigma=params["beta_sigma"],
            alpha_geom=params.get("alpha_geom", 1.0),
            backreaction_cap=params.get("backreaction_cap", 10.0),
        )

        tau_geom = compute_tau_geom(
            R,
            g_bar_kms2,
            rho_bar_msun_pc3,
            method=params.get("tau_geom_method", "tidal"),
            alpha_geom=params.get("alpha_geom", 1.0),
        )
        tau_noise = compute_tau_noise(
            R,
            sigma_v,
            method="galaxy",
            beta_sigma=params["beta_sigma"],
        )
        tau_coh = compute_tau_coh(tau_geom, tau_noise)

        Xi = compute_exposure_factor(R, g_bar_kms2, tau_coh)

        # Observed mass discrepancy profile
        mask = (V_gr > 1e-3) & (V_obs > 0)
        if np.count_nonzero(mask) < 4:
            continue
        disc = (V_obs[mask] ** 2 / V_gr[mask] ** 2) - 1.0
        K_eff = K[mask]
        Xi_eff = Xi[mask]

        rows.append(
            {
                "galaxy": galaxy,
                "sigma_v": sigma_v,
                "Xi_mean": float(np.mean(Xi_eff)),
                "Xi_max": float(np.max(Xi_eff)),
                "K_mean": float(np.mean(K_eff)),
                "K_max": float(np.max(K_eff)),
                "disc_mean": float(np.mean(disc)),
                "disc_med": float(np.median(disc)),
            }
        )

    df_out = pd.DataFrame(rows)
    outpath = Path("time-coherence/sparc_exposure_summary.csv")
    df_out.to_csv(outpath, index=False)
    
    print(f"\nSPARC exposure summary saved to {outpath}")

    if len(df_out) > 3:
        corr_disc_Xi = df_out["disc_mean"].corr(df_out["Xi_mean"])
        corr_K_Xi = df_out["K_mean"].corr(df_out["Xi_mean"])
        corr_disc_K = df_out["disc_mean"].corr(df_out["K_mean"])
        
        print("\n" + "=" * 80)
        print("SPARC EXPOSURE CORRELATIONS")
        print("=" * 80)
        print(f"corr(disc_mean, Xi_mean) = {corr_disc_Xi:.3f}")
        print(f"corr(K_mean, Xi_mean)   = {corr_K_Xi:.3f}")
        print(f"corr(disc_mean, K_mean) = {corr_disc_K:.3f}")
        print(f"\nMean Xi across SPARC: {df_out['Xi_mean'].mean():.3e}")
        print(f"Mean K across SPARC: {df_out['K_mean'].mean():.3f}")
        print(f"Mean disc across SPARC: {df_out['disc_mean'].mean():.3f}")


def main():
    analyze_mw_exposure()
    analyze_sparc_exposure()


if __name__ == "__main__":
    main()

