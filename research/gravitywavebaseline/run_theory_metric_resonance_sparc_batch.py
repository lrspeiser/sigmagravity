"""
Apply the theory-based metric resonance kernel to SPARC galaxies and
compare to GR-only RMS.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from theory_metric_resonance import theory_metric_resonance_multiplier


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def load_sparc_rotmod(rotmod_path: str) -> pd.DataFrame:
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


def lookup_sigma_v(
    galaxy_name: str,
    summary: pd.DataFrame,
    galaxy_col: str,
    sigma_col: str,
    sigma_ref: float,
) -> float:
    mask = summary[galaxy_col].astype(str).str.strip() == galaxy_name
    sub = summary.loc[mask]
    if sub.empty:
        print(f"[WARN] No sigma_v for {galaxy_name}, using sigma_ref={sigma_ref}")
        return sigma_ref
    val = float(sub.iloc[0][sigma_col])
    if not np.isfinite(val) or val <= 0:
        print(f"[WARN] Invalid sigma_v={val} for {galaxy_name}, using sigma_ref={sigma_ref}")
        return sigma_ref
    return val


def main():
    parser = argparse.ArgumentParser(
        description="Theory metric resonance batch run on SPARC rotmod files."
    )
    parser.add_argument("--rotmod-dir", default="data/Rotmod_LTG")
    parser.add_argument("--sparc-summary", default="data/sparc/sparc_combined.csv")
    parser.add_argument("--summary-galaxy-col", default="galaxy_name")
    parser.add_argument("--summary-sigma-col", default="sigma_velocity")
    parser.add_argument("--sigma-ref", type=float, default=25.0)
    parser.add_argument("--beta-sigma", type=float, default=1.0)
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_metric_resonance_sparc_batch.csv",
    )
    args = parser.parse_args()

    sparc_summary = pd.read_csv(args.sparc_summary)
    rotmod_paths = sorted(glob.glob(os.path.join(args.rotmod_dir, "*_rotmod.dat")))
    print(f"[INFO] Found {len(rotmod_paths)} rotmod files in {args.rotmod_dir}")

    rows: list[dict] = []
    for path in rotmod_paths:
        galaxy = os.path.basename(path).replace("_rotmod.dat", "")
        try:
            df = load_sparc_rotmod(path)
        except Exception as exc:
            print(f"[WARN] Skipping {galaxy}: {exc}")
            continue

        R = df["R_kpc"].to_numpy(float)
        V_obs = df["V_obs"].to_numpy(float)
        V_gr = df["V_gr"].to_numpy(float)
        if len(R) < 4:
            print(f"[WARN] {galaxy} has too few points; skipping")
            continue

        sigma_v_true = lookup_sigma_v(
            galaxy,
            sparc_summary,
            args.summary_galaxy_col,
            args.summary_sigma_col,
            args.sigma_ref,
        )

        v_flat = np.median(V_gr[-min(5, len(V_gr)):])
        Q_gal = v_flat / max(sigma_v_true, 1e-3)
        G_sigma = (Q_gal**args.beta_sigma) / (1.0 + Q_gal**args.beta_sigma)
        sigma_eff = args.sigma_ref / max(G_sigma, 1e-3)

        f_theory = theory_metric_resonance_multiplier(
            R_kpc=R,
            v_circ_kms=V_gr,
            sigma_v_kms=sigma_eff,
            A_global=1.0,
            lam_matter_mode="orbital_circumference",
            lam_coh_kpc=20.0,
            p_coh=2.0,
            spectrum_params=dict(alpha=3.5, lam_cut_kpc=500.0),
            resonance_extra=dict(),
        )

        V_model = V_gr * np.sqrt(np.clip(f_theory, 0.0, None))
        rms_gr = rms(V_obs - V_gr)
        rms_theory = rms(V_obs - V_model)

        rows.append(
            dict(
                galaxy=galaxy,
                n_points=len(R),
                sigma_v_true=sigma_v_true,
                Q_gal=Q_gal,
                sigma_eff=sigma_eff,
                G_sigma=G_sigma,
                rms_gr=rms_gr,
                rms_theory=rms_theory,
                delta_rms=rms_theory - rms_gr,
            )
        )

        print(
            f"[GAL] {galaxy:10s} sigma_v={sigma_v_true:5.1f} "
            f"Q={Q_gal:5.2f} G_sigma={G_sigma:5.2f} "
            f"RMS_GR={rms_gr:6.2f} RMS_theory={rms_theory:6.2f}"
        )

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).write_text(out_df.to_csv(index=False))
    print(f"[DONE] Wrote batch summary to {args.out_csv}")


if __name__ == "__main__":
    main()


