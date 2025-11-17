"""
Analyze how the theory kernel amplitude scales with sigma_v across SPARC galaxies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from theory_metric_resonance import compute_theory_kernel


def main():
    parser = argparse.ArgumentParser(
        description="Compare theory kernel strength vs empirical Q-gate across sigmas."
    )
    parser.add_argument(
        "--theory-fit-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument("--sparc-summary", default="data/sparc/sparc_combined.csv")
    parser.add_argument("--summary-galaxy-col", default="galaxy_name")
    parser.add_argument("--summary-sigma-col", default="sigma_velocity")
    parser.add_argument("--sigma-ref", type=float, default=25.0)
    parser.add_argument("--beta-sigma", type=float, default=1.0)
    parser.add_argument("--out-csv", default="gravitywavebaseline/theory_sigma_scaling_summary.csv")
    args = parser.parse_args()

    theory_fit = json.loads(Path(args.theory_fit_json).read_text())
    th = theory_fit["theory_fit_params"]

    summary = pd.read_csv(args.sparc_summary)
    sigmas = summary[args.summary_sigma_col].astype(float).to_numpy()
    galaxies = summary[args.summary_galaxy_col].astype(str).to_numpy()

    R_grid = np.linspace(5.0, 15.0, 40)
    rows: list[dict] = []

    for gal, sigma_v in zip(galaxies, sigmas):
        K_th = compute_theory_kernel(
            R_kpc=R_grid,
            sigma_v_kms=float(sigma_v),
            alpha=th["alpha"],
            lam_coh_kpc=th["lam_coh_kpc"],
            lam_cut_kpc=th["lam_cut_kpc"],
            A_global=th["A_global"],
        )
        K_mean = float(np.mean(K_th))
        v_circ_est = 200.0
        Q = v_circ_est / max(float(sigma_v), 1e-3)
        G_emp = Q**args.beta_sigma / (1.0 + Q**args.beta_sigma)

        rows.append(
            dict(
                galaxy=gal,
                sigma_v=float(sigma_v),
                Q=Q,
                K_mean=K_mean,
                G_emp=G_emp,
            )
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[info] wrote sigma scaling summary to {args.out_csv}")


if __name__ == "__main__":
    main()


