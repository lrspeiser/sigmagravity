"""
Compare theory kernel SPARC results against the empirical Phase-3 Σ-Gravity run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join theory-vs-empirical SPARC batch metrics for comparison."
    )
    parser.add_argument(
        "--theory-batch-csv",
        default="gravitywavebaseline/theory_kernel_sparc_batch.csv",
    )
    parser.add_argument(
        "--empirical-batch-csv",
        default="gravitywavebaseline/metric_resonance_sigma_phase3_beta1p5.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_vs_empirical_sparc_summary.csv",
    )
    args = parser.parse_args()

    theory = pd.read_csv(args.theory_batch_csv)
    empirical = pd.read_csv(args.empirical_batch_csv)

    merged = theory.merge(
        empirical,
        on="galaxy",
        suffixes=("_theory", "_emp"),
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No overlapping galaxies between theory and empirical CSVs.")

    merged["delta_emp"] = merged["delta_rms_emp"]
    merged["delta_theory"] = merged["delta_rms_theory"]
    merged["delta_emp_vs_gr"] = merged["rms_metric_resonance_sigma"] - merged["rms_gr_emp"]
    merged["delta_theory_vs_gr"] = merged["rms_theory"] - merged["rms_gr_theory"]

    Path(args.out_csv).write_text(merged.to_csv(index=False))
    print(f"[compare] wrote merged summary for {len(merged)} galaxies to {args.out_csv}")

    corr_sigma_k = merged["sigma_v_true"].corr(merged["K_mean"])
    corr_sigma_gate = merged["sigma_v"].corr(merged["sigma_gate"])
    print(f"[compare] corr(sigma_v, K_mean_theory) = {corr_sigma_k:.3f}")
    print(f"[compare] corr(sigma_v, sigma_gate_emp) = {corr_sigma_gate:.3f}")

    bins = [0, 20, 25, 30, 35, 40, 80]
    labels = ["<20", "20-25", "25-30", "30-35", "35-40", ">=40"]
    merged["sigma_bin"] = pd.cut(merged["sigma_v_true"], bins=bins, labels=labels)

    stats = merged.groupby("sigma_bin")[["delta_theory_vs_gr", "delta_emp_vs_gr"]].agg(
        ["mean", "median", "count"]
    )
    print("\n[compare] Delta RMS vs GR by sigma bin:")
    print(stats)


if __name__ == "__main__":
    main()


