"""
Fit theory-based metric resonance kernel parameters to the empirical MW kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from metric_resonance_multiplier import metric_resonance_multiplier
from theory_metric_resonance import compute_theory_kernel


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def load_mw_slice(parquet_path: str, r_min: float, r_max: float) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required = {"R", "v_phi", "v_phi_GR"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {parquet_path}: {missing}")
    mask = (
        df["R"].between(r_min, r_max)
        & np.isfinite(df["v_phi"])
        & np.isfinite(df["v_phi_GR"])
    )
    sl = df.loc[mask, ["R", "v_phi", "v_phi_GR"]].copy()
    if sl.empty:
        raise RuntimeError(f"No MW data in [{r_min}, {r_max}] kpc for {parquet_path}")
    return sl


def empirical_kernel(R_kpc: np.ndarray, mw_fit: dict) -> np.ndarray:
    lam_orb = 2.0 * np.pi * R_kpc
    f_emp = metric_resonance_multiplier(
        R_kpc=R_kpc,
        lambda_orb_kpc=lam_orb,
        A=mw_fit["A"],
        ell0_kpc=mw_fit["ell0_kpc"],
        p=mw_fit["p"],
        n_coh=mw_fit["n_coh"],
        lambda_peak_kpc=mw_fit["lambda_peak_kpc"],
        sigma_ln_lambda=mw_fit["sigma_ln_lambda"],
    )
    return f_emp - 1.0


def fit_theory_params(R_kpc: np.ndarray, K_emp: np.ndarray, sigma_v_kms: float) -> dict:
    bounds = [
        (0.1, 20.0),    # A_global
        (1.0, 6.0),     # alpha
        (1.0, 40.0),    # lam_coh_kpc
        (50.0, 1000.0), # lam_cut_kpc
    ]

    def objective(theta: np.ndarray) -> float:
        A_global, alpha, lam_coh_kpc, lam_cut_kpc = theta
        K_th = compute_theory_kernel(
            R_kpc=R_kpc,
            sigma_v_kms=sigma_v_kms,
            alpha=alpha,
            lam_coh_kpc=lam_coh_kpc,
            lam_cut_kpc=lam_cut_kpc,
            A_global=A_global,
        )
        return rms(K_th - K_emp)

    result = differential_evolution(objective, bounds=bounds, maxiter=80, polish=True)
    A_global, alpha, lam_coh_kpc, lam_cut_kpc = result.x
    K_best = compute_theory_kernel(
        R_kpc=R_kpc,
        sigma_v_kms=sigma_v_kms,
        alpha=alpha,
        lam_coh_kpc=lam_coh_kpc,
        lam_cut_kpc=lam_cut_kpc,
        A_global=A_global,
    )
    corr = float(np.corrcoef(K_emp, K_best)[0, 1])
    return {
        "A_global": float(A_global),
        "alpha": float(alpha),
        "lam_coh_kpc": float(lam_coh_kpc),
        "lam_cut_kpc": float(lam_cut_kpc),
        "rms_diff": float(rms(K_best - K_emp)),
        "corr_K_emp_theory": corr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit compute_theory_kernel parameters to MW empirical kernel."
    )
    parser.add_argument(
        "--baseline-parquet",
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
    )
    parser.add_argument(
        "--mw-fit-json",
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
    )
    parser.add_argument("--r-min", type=float, default=12.0)
    parser.add_argument("--r-max", type=float, default=16.0)
    parser.add_argument("--sigma-v", type=float, default=30.0, help="MW dispersion (km/s)")
    parser.add_argument(
        "--out-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    args = parser.parse_args()

    df = load_mw_slice(args.baseline_parquet, args.r_min, args.r_max)
    R = df["R"].to_numpy(float)
    v_obs = df["v_phi"].to_numpy(float)
    v_gr = df["v_phi_GR"].to_numpy(float)

    rms_gr = rms(v_obs - v_gr)
    print(f"[info] GR-only RMS in MW slice: {rms_gr:.2f} km/s (N={len(df)})")

    mw_fit = json.loads(Path(args.mw_fit_json).read_text())
    K_emp = empirical_kernel(R, mw_fit)
    print(
        f"[info] Empirical kernel range: K_min={K_emp.min():.4f}, "
        f"K_max={K_emp.max():.4f}"
    )

    theory_fit = fit_theory_params(R, K_emp, args.sigma_v)
    print("[info] Best-fit theory parameters:")
    for k, v in theory_fit.items():
        print(f"  {k}: {v}")

    out = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "sigma_v": args.sigma_v,
        "n_points": len(df),
        "rms_gr": rms_gr,
        "mw_fit_params": mw_fit,
        "theory_fit_params": theory_fit,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[info] Saved theory MW fit to {args.out_json}")


if __name__ == "__main__":
    main()


