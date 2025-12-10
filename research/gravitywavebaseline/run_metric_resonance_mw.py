"""
Phase-1 harness: fit the metric-resonance kernel to the Milky Way outer disk.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from metric_resonance_multiplier import metric_resonance_multiplier


def load_mw_baseline(parquet_path, r_min=12.0, r_max=16.0):
    df = pd.read_parquet(parquet_path)
    required_cols = {"R", "v_phi", "v_phi_GR"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    mask = (
        df["R"].between(r_min, r_max)
        & np.isfinite(df["v_phi"])
        & np.isfinite(df["v_phi_GR"])
    )
    sliced = df.loc[mask, ["R", "v_phi", "v_phi_GR"]].copy()
    if sliced.empty:
        raise RuntimeError(
            f"No MW points between {r_min}-{r_max} kpc in {parquet_path}"
        )
    return sliced


def rms(residuals):
    return float(np.sqrt(np.mean(residuals**2)))


def fit_metric_resonance(df, bounds=None, seed=42):
    rng = np.random.default_rng(seed)
    R = df["R"].to_numpy(dtype=float)
    v_obs = df["v_phi"].to_numpy(dtype=float)
    v_gr = df["v_phi_GR"].to_numpy(dtype=float)
    lambda_orb = 2.0 * np.pi * R

    ell0_kpc = 5.0
    p = 0.757
    n_coh = 0.5

    if bounds is None:
        bounds = [
            (0.0, 8.0),      # A
            (5.0, 120.0),    # lambda_peak_kpc
            (0.3, 4.0),      # sigma_ln_lambda
        ]

    def objective(theta):
        A, lambda_peak_kpc, sigma_ln_lambda = theta
        f_res = metric_resonance_multiplier(
            R,
            lambda_orb,
            A=A,
            ell0_kpc=ell0_kpc,
            p=p,
            n_coh=n_coh,
            lambda_peak_kpc=lambda_peak_kpc,
            sigma_ln_lambda=sigma_ln_lambda,
        )
        v_model = v_gr * np.sqrt(np.clip(f_res, 0.0, None))
        return rms(v_model - v_obs)

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=rng,
        maxiter=150,
        polish=True,
    )

    best_theta = result.x
    best_rms = objective(best_theta)
    return {
        "A": float(best_theta[0]),
        "lambda_peak_kpc": float(best_theta[1]),
        "sigma_ln_lambda": float(best_theta[2]),
        "ell0_kpc": ell0_kpc,
        "p": p,
        "n_coh": n_coh,
        "rms_best": best_rms,
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit metric resonance kernel to MW GR baseline."
    )
    parser.add_argument(
        "--baseline-parquet",
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
        help="Path to gaia_with_gr_baseline.parquet.",
    )
    parser.add_argument("--r-min", type=float, default=12.0)
    parser.add_argument("--r-max", type=float, default=16.0)
    parser.add_argument(
        "--out-json",
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_mw_baseline(args.baseline_parquet, args.r_min, args.r_max)
    print(f"Loaded {len(df)} MW points between {args.r_min}-{args.r_max} kpc")

    rms_gr = rms(df["v_phi"] - df["v_phi_GR"])
    print(f"GR-only RMS: {rms_gr:.2f} km/s")

    best = fit_metric_resonance(df, seed=args.seed)
    print("\nBest-fit metric resonance parameters:")
    for key in ["A", "lambda_peak_kpc", "sigma_ln_lambda", "ell0_kpc", "p", "n_coh"]:
        print(f"  {key}: {best[key]:.6g}")
    print(f"  rms_best: {best['rms_best']:.2f} km/s")

    Path(args.out_json).write_text(json.dumps(best, indent=2))
    print(f"\nSaved fit to {args.out_json}")


if __name__ == "__main__":
    main()
