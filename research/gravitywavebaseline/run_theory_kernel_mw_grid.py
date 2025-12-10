"""
Grid / random scan of the first-principles theory kernel against the Milky Way.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_resonance_multiplier import metric_resonance_multiplier
from theory_metric_resonance import compute_theory_kernel


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


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
        raise RuntimeError(f"No MW data in [{r_min}, {r_max}] kpc within {parquet_path}")
    return sl


def compute_empirical_kernel(R_kpc: np.ndarray, mw_fit: dict) -> np.ndarray:
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


def iter_parameter_grid(
    *,
    alphas: list[float],
    lam_cohs: list[float],
    lam_cuts: list[float],
    amplitudes: list[float],
    burr_ell0s: list[float],
) -> dict:
    for alpha, lam_coh, lam_cut, amp, ell0 in itertools.product(
        alphas, lam_cohs, lam_cuts, amplitudes, burr_ell0s
    ):
        yield dict(
            alpha=alpha,
            lam_coh_kpc=lam_coh,
            lam_cut_kpc=lam_cut,
            A_global=amp,
            burr_ell0_kpc=ell0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan theory kernel parameters for the Milky Way outer disk."
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
    parser.add_argument("--sigma-v", type=float, default=30.0)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0],
        help="Power-law slopes for fluctuation spectrum.",
    )
    parser.add_argument(
        "--lam-cohs",
        type=float,
        nargs="+",
        default=[10.0, 40.0, 80.0],
        help="Coherence lengths (kpc) to scan.",
    )
    parser.add_argument(
        "--lam-cuts",
        type=float,
        nargs="+",
        default=[500.0, 900.0, 1300.0],
        help="λ-cutoff scales (kpc) to scan.",
    )
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0],
        help="Global amplitudes to test.",
    )
    parser.add_argument(
        "--burr-ell0s",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 40.0],
        help="Burr ℓ0 coherence scales (kpc).",
    )
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_kernel_mw_grid.csv",
    )
    args = parser.parse_args()

    df = load_mw_slice(args.baseline_parquet, args.r_min, args.r_max)
    R = df["R"].to_numpy(float)
    v_obs = df["v_phi"].to_numpy(float)
    v_gr = df["v_phi_GR"].to_numpy(float)
    lam_orb = 2.0 * np.pi * R

    mw_fit = json.loads(Path(args.mw_fit_json).read_text())
    burr_p = float(mw_fit.get("p", 1.0))
    burr_n = float(mw_fit.get("n_coh", 0.5))

    K_emp = compute_empirical_kernel(R, mw_fit)
    f_emp = 1.0 + K_emp
    v_emp = v_gr * np.sqrt(np.clip(f_emp, 0.0, None))
    rms_emp = rms(v_obs - v_emp)
    rms_gr = rms(v_obs - v_gr)

    print(f"[MW grid] GR-only RMS: {rms_gr:.2f} km/s")
    print(f"[MW grid] Empirical kernel RMS: {rms_emp:.2f} km/s")

    rows: list[dict] = []

    grid_iter = iter_parameter_grid(
        alphas=args.alphas,
        lam_cohs=args.lam_cohs,
        lam_cuts=args.lam_cuts,
        amplitudes=args.amplitudes,
        burr_ell0s=args.burr_ell0s,
    )

    for params in grid_iter:
        K_th = compute_theory_kernel(
            R_kpc=R,
            sigma_v_kms=args.sigma_v,
            alpha=params["alpha"],
            lam_coh_kpc=params["lam_coh_kpc"],
            lam_cut_kpc=params["lam_cut_kpc"],
            A_global=params["A_global"],
            burr_ell0_kpc=params["burr_ell0_kpc"],
            burr_p=burr_p,
            burr_n=burr_n,
        )

        f_pos = 1.0 + K_th
        v_pos = v_gr * np.sqrt(np.clip(f_pos, 0.0, None))
        rms_pos = rms(v_obs - v_pos)
        corr_pos = np.corrcoef(K_emp, K_th)[0, 1]

        f_neg = 1.0 - K_th
        v_neg = v_gr * np.sqrt(np.clip(f_neg, 0.0, None))
        rms_neg = rms(v_obs - v_neg)
        corr_neg = np.corrcoef(K_emp, -K_th)[0, 1]

        if not np.isfinite(corr_pos):
            corr_pos = 0.0
        if not np.isfinite(corr_neg):
            corr_neg = 0.0

        if rms_pos <= rms_neg:
            phase_sign = 1.0
            corr = corr_pos
            rms_th = rms_pos
        else:
            phase_sign = -1.0
            corr = corr_neg
            rms_th = rms_neg

        corr = float(np.clip(corr, -1.0, 1.0))

        rows.append(
            {
                **params,
                "phase_sign": phase_sign,
                "corr_K_emp_theory": float(corr),
                "rms_theory": rms_th,
                "rms_empirical": rms_emp,
                "rms_gr": rms_gr,
                "delta_rms_vs_emp": rms_th - rms_emp,
                "delta_rms_vs_gr": rms_th - rms_gr,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[MW grid] Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


