"""
Apply theory kernel parameters (fit on MW) to a batch of SPARC galaxies.
"""

from __future__ import annotations

import argparse
import glob
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from theory_metric_resonance import compute_theory_kernel


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def load_rotmod(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
    if df.empty:
        raise RuntimeError(f"No data parsed from {path}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Test theory kernel across SPARC galaxies."
    )
    parser.add_argument("--rotmod-dir", default="data/Rotmod_LTG")
    parser.add_argument("--sparc-summary", default="data/sparc/sparc_combined.csv")
    parser.add_argument("--summary-galaxy-col", default="galaxy_name")
    parser.add_argument("--summary-sigma-col", default="sigma_velocity")
    parser.add_argument("--sigma-ref", type=float, default=25.0)
    parser.add_argument("--beta-sigma", type=float, default=1.0)
    parser.add_argument(
        "--theory-fit-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_kernel_sparc_batch.csv",
    )
    args = parser.parse_args()

    theory_fit = json.loads(Path(args.theory_fit_json).read_text())
    th = theory_fit["theory_fit_params"]

    summary = pd.read_csv(args.sparc_summary)
    sigma_map = dict(
        zip(
            summary[args.summary_galaxy_col].astype(str),
            summary[args.summary_sigma_col].astype(float),
        )
    )

    rows: list[dict] = []
    rotmod_paths = sorted(glob.glob(os.path.join(args.rotmod_dir, "*_rotmod.dat")))
    print(f"[info] processing {len(rotmod_paths)} SPARC rotmod files")

    for path in rotmod_paths:
        galaxy = Path(path).name.replace("_rotmod.dat", "")
        try:
            df = load_rotmod(path)
        except Exception as exc:
            print(f"[warn] skip {galaxy}: {exc}")
            continue

        R = df["R_kpc"].to_numpy(float)
        V_obs = df["V_obs"].to_numpy(float)
        V_gr = df["V_gr"].to_numpy(float)
        if len(R) < 4:
            continue

        sigma_true = sigma_map.get(galaxy, args.sigma_ref)
        v_flat = np.nanmedian(V_gr[-min(len(V_gr), 5):]) or 200.0
        Q = v_flat / max(sigma_true, 1e-3)
        G_sigma = (Q**args.beta_sigma) / (1.0 + Q**args.beta_sigma)

        K_th = compute_theory_kernel(
            R_kpc=R,
            sigma_v_kms=sigma_true,
            alpha=th["alpha"],
            lam_coh_kpc=th["lam_coh_kpc"],
            lam_cut_kpc=th["lam_cut_kpc"],
            A_global=th["A_global"] * G_sigma,
            burr_ell0_kpc=th.get("burr_ell0_kpc"),
            burr_p=th.get("burr_p", 1.0),
            burr_n=th.get("burr_n", 0.5),
        )
        f_th = 1.0 + K_th
        V_model = V_gr * np.sqrt(np.clip(f_th, 0.0, None))

        rms_gr = rms(V_obs - V_gr)
        rms_th = rms(V_obs - V_model)

        rows.append(
            dict(
                galaxy=galaxy,
                n_points=len(R),
                sigma_v_true=sigma_true,
                Q_gal=Q,
                G_sigma=G_sigma,
                rms_gr=rms_gr,
                rms_theory=rms_th,
                delta_rms=rms_th - rms_gr,
            )
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[info] wrote theory kernel batch results to {args.out_csv}")


if __name__ == "__main__":
    main()


