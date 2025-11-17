"""
Apply the theory kernel to cluster baryon profiles and compare mass boosts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from theory_metric_resonance import compute_theory_kernel


G_KPC_KM2_S2_MSUN = 4.30091e-6  # gravitational constant in (kpc km^2 / s^2 / Msun)


def load_cluster_profile(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"R_kpc", "M_baryon"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    return df


def compute_enhanced_mass(
    df: pd.DataFrame,
    theory_params: dict,
    sigma_v_kms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = df["R_kpc"].to_numpy(dtype=float)
    M_baryon = df["M_baryon"].to_numpy(dtype=float)
    lam_orb = 2.0 * np.pi * R

    g_gr = G_KPC_KM2_S2_MSUN * M_baryon / np.maximum(R * R, 1e-6)
    th = theory_params["theory_fit_params"]

    K_th = compute_theory_kernel(
        R_kpc=R,
        sigma_v_kms=sigma_v_kms,
        alpha=th["alpha"],
        lam_coh_kpc=th["lam_coh_kpc"],
        lam_cut_kpc=th["lam_cut_kpc"],
        A_global=th["A_global"],
        burr_ell0_kpc=th.get("burr_ell0_kpc"),
        burr_p=th.get("burr_p", 1.0),
        burr_n=th.get("burr_n", 0.5),
    )
    phase_sign = float(th.get("phase_sign", 1.0))
    f_th = 1.0 + phase_sign * K_th

    g_eff = g_gr * np.clip(f_th, 0.0, None)
    M_eff = g_eff * R * R / G_KPC_KM2_S2_MSUN
    return g_gr, g_eff, M_eff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate theory kernel predictions on cluster baryon profiles."
    )
    parser.add_argument(
        "--cluster-list",
        required=True,
        help="Text file listing cluster profile CSV paths (one per line).",
    )
    parser.add_argument(
        "--theory-fit-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument(
        "--default-sigma-v",
        type=float,
        default=1000.0,
        help="Fallback cluster velocity dispersion in km/s.",
    )
    parser.add_argument(
        "--sigma-col",
        default=None,
        help="Optional column in cluster CSV to use for sigma_v per cluster.",
    )
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_kernel_cluster_summary.csv",
    )
    args = parser.parse_args()

    theory_params = json.loads(Path(args.theory_fit_json).read_text())
    rows: list[dict] = []

    for line in Path(args.cluster_list).read_text().splitlines():
        path = line.strip()
        if not path:
            continue
        df = load_cluster_profile(path)
        sigma_v = float(df[args.sigma_col].iloc[0]) if args.sigma_col else args.default_sigma_v
        g_gr, g_eff, M_eff = compute_enhanced_mass(df, theory_params, sigma_v)

        cluster_row = {
            "cluster_profile": path,
            "sigma_v_kms": sigma_v,
            "r_max_kpc": float(df["R_kpc"].max()),
            "M_baryon_max": float(df["M_baryon"].max()),
            "M_eff_max": float(M_eff.max()),
            "mass_boost_max": float(np.max(np.divide(M_eff, df["M_baryon"], where=df["M_baryon"] > 0))),
        }

        if "M_lens_obs" in df.columns:
            M_obs = df["M_lens_obs"].to_numpy(dtype=float)
            valid = np.isfinite(M_obs) & (M_obs > 0)
            if np.any(valid):
                resid = M_eff[valid] - M_obs[valid]
                cluster_row["rms_mass_resid"] = float(np.sqrt(np.mean(resid * resid)))
            else:
                cluster_row["rms_mass_resid"] = np.nan
        else:
            cluster_row["rms_mass_resid"] = np.nan

        rows.append(cluster_row)

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).write_text(out_df.to_csv(index=False))
    print(f"[cluster] wrote summary for {len(rows)} profiles to {args.out_csv}")


if __name__ == "__main__":
    main()


