"""
Joint MW + SPARC fit for time-coherence kernel hyperparameters.

Goal:
  - Keep the successful time-coherence functional form,
  - Tune a small set of hyperparameters so that:
      * MW outer disk 12–16 kpc: RMS matches empirical kernel (~40 km/s)
      * SPARC: mean ΔRMS ≈ 0, 70–80% galaxies improved
      * Derived coherence length for MW moves toward ~5–20 kpc

Usage:
  python time-coherence/fit_time_coherence_hyperparams.py \
      --mw-parquet gravitywavebaseline/gaia_with_gr_baseline.parquet \
      --sparc-rotmod-dir data/Rotmod_LTG \
      --sparc-summary data/sparc/sparc_combined.csv \
      --out-json time-coherence/time_coherence_fit_hyperparams.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from coherence_time_kernel import compute_coherence_kernel


def rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def load_mw_baseline(parquet_path: str, r_min: float = 12.0, r_max: float = 16.0) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required = {"R", "v_phi", "v_phi_GR"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"MW parquet missing columns: {missing}")
    mask = (
        df["R"].between(r_min, r_max)
        & np.isfinite(df["v_phi"])
        & np.isfinite(df["v_phi_GR"])
    )
    sub = df.loc[mask, ["R", "v_phi", "v_phi_GR"]].copy()
    if sub.empty:
        raise RuntimeError(f"No MW points in [{r_min}, {r_max}] kpc from {parquet_path}")
    return sub


def load_sparc_rotmod(rotmod_path: str) -> pd.DataFrame:
    """Load SPARC rotmod file."""
    df = pd.read_csv(
        rotmod_path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
    if df.empty:
        raise RuntimeError(f"No data parsed from {rotmod_path}")
    df["V_gr"] = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    return df[["R_kpc", "V_obs", "V_gr"]]


def load_sparc_sigma(summary_csv: str, galaxy_col: str, sigma_col: str) -> Dict[str, float]:
    df = pd.read_csv(summary_csv)
    if galaxy_col not in df.columns or sigma_col not in df.columns:
        raise KeyError(f"Summary CSV missing {galaxy_col} or {sigma_col}")
    return dict(zip(df[galaxy_col].astype(str), df[sigma_col].astype(float)))


def list_sparc_galaxies(rotmod_dir: str) -> List[Tuple[str, Path]]:
    paths = sorted(Path(rotmod_dir).glob("*_rotmod.dat"))
    result: List[Tuple[str, Path]] = []
    for p in paths:
        name = p.name.replace("_rotmod.dat", "")
        result.append((name, p))
    return result


def compute_multiplier(
    R_kpc: np.ndarray,
    V_gr: np.ndarray,
    sigma_v: float,
    *,
    A_global: float,
    p: float,
    n_coh: float,
    delta_R_kpc: float,
    tau_geom_method: str,
    alpha_length: float = 0.037,
    beta_sigma: float = 1.5,
) -> np.ndarray:
    """
    Compute time-coherence multiplier f_time such that:
        V_model = V_gr * sqrt(1 + K)
    where K is the enhancement kernel.
    """
    # Compute g_bar from V_gr
    g_bar_kms2 = (V_gr**2) / (R_kpc * 1e3)  # km/s²
    
    # Estimate density for tidal method
    G_msun_kpc_km2_s2 = 4.302e-6
    rho_bar_msun_pc3 = g_bar_kms2 / (G_msun_kpc_km2_s2 * R_kpc * 1e3) * 1e-9
    
    # Compute coherence kernel
    K = compute_coherence_kernel(
        R_kpc=R_kpc,
        g_bar_kms2=g_bar_kms2,
        sigma_v_kms=sigma_v,
        A_global=A_global,
        p=p,
        n_coh=n_coh,
        method="galaxy",
        rho_bar_msun_pc3=rho_bar_msun_pc3 if tau_geom_method == "tidal" else None,
        delta_R_kpc=delta_R_kpc,
        tau_geom_method=tau_geom_method,
        alpha_length=alpha_length,
        beta_sigma=beta_sigma,
    )
    
    # Return multiplier: f_time = 1 + K
    return 1.0 + K


def make_objective(
    mw_df: pd.DataFrame,
    sparc_subset: List[Tuple[str, Path]],
    sigma_lookup: Dict[str, float],
    sigma_ref: float,
    mw_weight: float = 1.0,
    sparc_weight: float = 1.0,
):
    R_mw = mw_df["R"].to_numpy(dtype=float)
    V_obs_mw = mw_df["v_phi"].to_numpy(dtype=float)
    V_gr_mw = mw_df["v_phi_GR"].to_numpy(dtype=float)
    rms_gr_mw = rms(V_obs_mw - V_gr_mw)

    def objective(theta: np.ndarray) -> float:
        """
        theta = [A_global, p, n_coh, log10_delta_R_kpc]
        """
        A_global, p, n_coh, log10_delta_R_kpc = theta
        delta_R_kpc = 10.0 ** log10_delta_R_kpc
        
        # Use tidal method (more physical)
        tau_geom_method = "tidal"

        # --- MW contribution ---
        f_mw = compute_multiplier(
            R_kpc=R_mw,
            V_gr=V_gr_mw,
            sigma_v=sigma_ref,
            A_global=A_global,
            p=p,
            n_coh=n_coh,
            delta_R_kpc=delta_R_kpc,
            tau_geom_method=tau_geom_method,
            alpha_length=0.037,
            beta_sigma=1.5,
        )
        V_model_mw = V_gr_mw * np.sqrt(np.clip(f_mw, 0.0, None))
        rms_mw = rms(V_obs_mw - V_model_mw)

        # Target MW RMS ~40 km/s (from empirical fit)
        target_mw = 40.0  # km/s
        loss_mw = ((rms_mw - target_mw) / target_mw) ** 2

        # --- SPARC subset contribution ---
        sparc_losses: List[float] = []
        for gal_name, path in sparc_subset:
            try:
                sigma_v = float(sigma_lookup.get(gal_name, sigma_ref))
                df_g = load_sparc_rotmod(str(path))
            except Exception:
                continue

            R_g = df_g["R_kpc"].to_numpy(dtype=float)
            V_obs_g = df_g["V_obs"].to_numpy(dtype=float)
            V_gr_g = df_g["V_gr"].to_numpy(dtype=float)
            rms_gr_g = rms(V_obs_g - V_gr_g)

            f_g = compute_multiplier(
                R_kpc=R_g,
                V_gr=V_gr_g,
                sigma_v=sigma_v,
                A_global=A_global,
                p=p,
                n_coh=n_coh,
                delta_R_kpc=delta_R_kpc,
                tau_geom_method=tau_geom_method,
                alpha_length=0.037,
                beta_sigma=1.5,
            )
            V_model_g = V_gr_g * np.sqrt(np.clip(f_g, 0.0, None))
            rms_g = rms(V_obs_g - V_model_g)

            # Penalize large positive Δ more than small negatives
            delta = rms_g - rms_gr_g
            if delta >= 0:
                sparc_losses.append((delta / max(rms_gr_g, 1.0)) ** 2)
            else:
                # Small reward for improvement, but don't overfit
                sparc_losses.append(0.1 * (delta / max(rms_gr_g, 1.0)) ** 2)

        if sparc_losses:
            loss_sparc = float(np.mean(sparc_losses))
        else:
            loss_sparc = 1.0

        total_loss = mw_weight * loss_mw + sparc_weight * loss_sparc
        return float(total_loss)

    return objective, rms_gr_mw


def main():
    parser = argparse.ArgumentParser(
        description="Joint MW+SPARC fit for time-coherence kernel hyperparameters."
    )
    parser.add_argument(
        "--mw-parquet",
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
    )
    parser.add_argument(
        "--sparc-rotmod-dir",
        default="data/Rotmod_LTG",
    )
    parser.add_argument(
        "--sparc-summary",
        default="data/sparc/sparc_combined.csv",
    )
    parser.add_argument(
        "--summary-galaxy-col",
        default="galaxy_name",
    )
    parser.add_argument(
        "--summary-sigma-col",
        default="sigma_velocity",
    )
    parser.add_argument(
        "--n-sparc",
        type=int,
        default=40,
        help="Number of SPARC galaxies to use in the fit (subset for speed).",
    )
    parser.add_argument(
        "--sigma-ref",
        type=float,
        default=30.0,
        help="Reference σ_v in km/s (used for MW and as fallback).",
    )
    parser.add_argument(
        "--out-json",
        default="time-coherence/time_coherence_fit_hyperparams.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=100)
    args = parser.parse_args()

    print("=" * 80)
    print("TIME-COHERENCE HYPERPARAMETER FITTING")
    print("=" * 80)

    # --- Load MW slice ---
    mw_df = load_mw_baseline(args.mw_parquet, r_min=12.0, r_max=16.0)
    print(f"\nLoaded MW slice with {len(mw_df)} points.")
    rms_gr_mw = rms(mw_df["v_phi"] - mw_df["v_phi_GR"])
    print(f"GR-only RMS (MW 12–16 kpc): {rms_gr_mw:.2f} km/s")

    # --- Load SPARC list + sigmas and pick a subset ---
    sigma_lookup = load_sparc_sigma(
        args.sparc_summary,
        args.summary_galaxy_col,
        args.summary_sigma_col,
    )
    all_gals = list_sparc_galaxies(args.sparc_rotmod_dir)
    if args.n_sparc < len(all_gals):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(all_gals), size=args.n_sparc, replace=False)
        subset = [all_gals[int(i)] for i in idx]
    else:
        subset = all_gals
    print(f"Using {len(subset)} SPARC galaxies in the fit.")

    objective, _ = make_objective(
        mw_df=mw_df,
        sparc_subset=subset,
        sigma_lookup=sigma_lookup,
        sigma_ref=args.sigma_ref,
    )

    # --- Optimize ---
    # Bounds: A_global, p, n_coh, log10_delta_R_kpc
    bounds = [
        (0.1, 5.0),      # A_global
        (0.3, 1.5),      # p (Burr-XII shape)
        (0.1, 1.0),      # n_coh (Burr-XII shape)
        (-2.0, 1.0),     # log10_delta_R_kpc (0.01 to 10 kpc)
    ]

    print(f"\nOptimizing with bounds:")
    print(f"  A_global: {bounds[0]}")
    print(f"  p: {bounds[1]}")
    print(f"  n_coh: {bounds[2]}")
    print(f"  log10_delta_R_kpc: {bounds[3]}")
    print(f"\nRunning differential evolution (maxiter={args.maxiter})...")

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=args.seed,
        maxiter=args.maxiter,
        polish=True,
        workers=1,  # Single worker for reproducibility
    )

    best_theta = result.x
    A_global, p, n_coh, log10_delta_R_kpc = best_theta
    delta_R_kpc = 10.0 ** log10_delta_R_kpc

    # Evaluate final performance
    print("\n" + "=" * 80)
    print("FITTING COMPLETE")
    print("=" * 80)
    
    # Test on MW
    f_mw = compute_multiplier(
        R_kpc=mw_df["R"].to_numpy(dtype=float),
        V_gr=mw_df["v_phi_GR"].to_numpy(dtype=float),
        sigma_v=args.sigma_ref,
        A_global=A_global,
        p=p,
        n_coh=n_coh,
        delta_R_kpc=delta_R_kpc,
        tau_geom_method="tidal",
    )
    V_model_mw = mw_df["v_phi_GR"].to_numpy(dtype=float) * np.sqrt(np.clip(f_mw, 0.0, None))
    rms_mw_final = rms(mw_df["v_phi"] - V_model_mw)
    
    print(f"\nBest-fit hyperparameters:")
    print(f"  A_global: {A_global:.6g}")
    print(f"  p: {p:.6g}")
    print(f"  n_coh: {n_coh:.6g}")
    print(f"  delta_R_kpc: {delta_R_kpc:.6g} kpc")
    print(f"\nMW performance:")
    print(f"  GR-only RMS: {rms_gr_mw:.2f} km/s")
    print(f"  Time-coherence RMS: {rms_mw_final:.2f} km/s")
    print(f"  Improvement: {rms_gr_mw - rms_mw_final:.2f} km/s")
    print(f"  Target: ~40 km/s")

    out = {
        "A_global": float(A_global),
        "p": float(p),
        "n_coh": float(n_coh),
        "delta_R_kpc": float(delta_R_kpc),
        "log10_delta_R_kpc": float(log10_delta_R_kpc),
        "tau_geom_method": "tidal",
        "seed": args.seed,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "mw_parquet": args.mw_parquet,
        "sparc_rotmod_dir": args.sparc_rotmod_dir,
        "sparc_summary": args.sparc_summary,
        "n_sparc": len(subset),
        "sigma_ref": args.sigma_ref,
        "rms_gr_mw": float(rms_gr_mw),
        "rms_time_coherence_mw": float(rms_mw_final),
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()

