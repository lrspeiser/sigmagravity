"""
Compare the theory-based metric resonance kernel against:
  1) empirical metric_resonance_multiplier (Phase-1 fit)
  2) actual Gaia MW outer-disk data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_resonance_multiplier import metric_resonance_multiplier
from theory_metric_resonance import theory_metric_resonance_multiplier


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
        raise RuntimeError(f"No MW points in R=[{r_min},{r_max}] within {parquet_path}")
    return sl


def main():
    parser = argparse.ArgumentParser(
        description="Theory vs empirical metric resonance comparison on MW data."
    )
    parser.add_argument(
        "--baseline-parquet",
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
        help="Path to Gaia GR baseline parquet with columns R, v_phi, v_phi_GR.",
    )
    parser.add_argument(
        "--mw-fit-json",
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
        help="Empirical Phase-1 fit JSON.",
    )
    parser.add_argument("--sigma-v", type=float, default=30.0, help="MW dispersion (km/s).")
    parser.add_argument("--r-min", type=float, default=12.0)
    parser.add_argument("--r-max", type=float, default=16.0)
    parser.add_argument(
        "--out-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_results.json",
        help="Where to store summary JSON.",
    )
    args = parser.parse_args()

    df = load_mw_slice(args.baseline_parquet, args.r_min, args.r_max)
    R = df["R"].to_numpy(float)
    v_obs = df["v_phi"].to_numpy(float)
    v_gr = df["v_phi_GR"].to_numpy(float)

    rms_gr = rms(v_obs - v_gr)
    print(f"[MW] GR-only RMS: {rms_gr:.2f} km/s (N={len(df)})")

    fit = json.loads(Path(args.mw_fit_json).read_text())
    f_emp = metric_resonance_multiplier(
        R_kpc=R,
        lambda_orb_kpc=2.0 * np.pi * R,
        A=fit["A"],
        ell0_kpc=fit["ell0_kpc"],
        p=fit["p"],
        n_coh=fit["n_coh"],
        lambda_peak_kpc=fit["lambda_peak_kpc"],
        sigma_ln_lambda=fit["sigma_ln_lambda"],
    )
    v_emp = v_gr * np.sqrt(np.clip(f_emp, 0.0, None))
    rms_emp = rms(v_obs - v_emp)
    print(f"[MW] Empirical metric resonance RMS: {rms_emp:.2f} km/s")

    f_theory = theory_metric_resonance_multiplier(
        R_kpc=R,
        v_circ_kms=v_gr,
        sigma_v_kms=args.sigma_v,
        A_global=1.0,
        lam_matter_mode="orbital_circumference",
        lam_coh_kpc=20.0,
        p_coh=2.0,
        spectrum_params=dict(alpha=3.5, lam_cut_kpc=500.0),
        resonance_extra=dict(),
    )

    d_theory = f_theory - 1.0
    d_emp = f_emp - 1.0
    num = np.sum(d_theory * d_emp)
    den = np.sum(d_theory * d_theory) + 1e-12
    A_fit = float(num / den)
    f_theory_adj = 1.0 + A_fit * d_theory
    v_theory = v_gr * np.sqrt(np.clip(f_theory_adj, 0.0, None))
    rms_theory = rms(v_obs - v_theory)
    corr = np.corrcoef(d_emp, A_fit * d_theory)[0, 1]

    print(f"[MW] Theory amplitude fit A_fit={A_fit:.3f}")
    print(f"[MW] Theory RMS after scaling: {rms_theory:.2f} km/s")
    print(f"[MW] Corr(K_emp, K_theory)={corr:.3f}")

    out = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "sigma_v": args.sigma_v,
        "n_points": int(len(df)),
        "rms_gr": rms_gr,
        "rms_empirical": rms_emp,
        "rms_theory": rms_theory,
        "corr_K_emp_theory": float(corr),
        "A_fit_for_theory": A_fit,
        "mw_fit_params": fit,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[MW] Wrote results to {args.out_json}")


if __name__ == "__main__":
    main()


