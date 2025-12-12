#!/usr/bin/env python3
"""Plot SPARC GR mismatch vs 2MASS XSC J−K.

Reads:
- resources/photometry/sparc_2mass_xsc_color_residuals.csv

Writes:
- figures/photometry/sparc_gr_discrepancy_vs_2mass_jk.png

The plot is meant to visually answer: is the GR+baryons mismatch random with respect to a
stellar-population color (J−K), or does it show structure?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(x > 0, x, np.nan)
    return np.log10(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="resources/photometry/sparc_2mass_xsc_color_residuals.csv",
        help="Input merged table",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="figures/photometry/sparc_gr_discrepancy_vs_2mass_jk.png",
        help="Output PNG path",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    in_path = (repo_root / args.in_path).resolve()
    out_path = (repo_root / args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # All usable matches
    all_mask = (
        (df.get("match_found") == True)
        & df["J_minus_K_ext"].notna()
        & df["md_med"].notna()
        & df["median_log10_Vobs_over_Vbar"].notna()
    )
    df_all = df.loc[all_mask].copy()

    # Preferred quality subset (as defined in analyze_sparc_2mass_xsc_m2l_test.py)
    if "jk_good" in df.columns:
        df_q = df_all[df_all["jk_good"] == True].copy()
    else:
        # fallback
        df_q = df_all.copy()

    x_all = df_all["J_minus_K_ext"].astype(float).values
    y_md_all = df_all["md_med"].astype(float).values
    y_bias_all = df_all["median_log10_Vobs_over_Vbar"].astype(float).values

    x = df_q["J_minus_K_ext"].astype(float).values
    y_md = df_q["md_med"].astype(float).values
    y_bias = df_q["median_log10_Vobs_over_Vbar"].astype(float).values

    # Correlations (quality subset)
    md_s = stats.spearmanr(x, y_md)
    md_p = stats.pearsonr(x, y_md)
    bias_s = stats.spearmanr(x, y_bias)
    bias_p = stats.pearsonr(x, y_bias)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)

    # Panel A: mass discrepancy proxy
    ax = axes[0]
    ax.scatter(x_all, _safe_log10(y_md_all), s=14, c="#999999", alpha=0.25, edgecolors="none", label="All XSC matches")
    ax.scatter(x, _safe_log10(y_md), s=18, c="#1f77b4", alpha=0.85, edgecolors="none", label="Quality subset")

    # Fit line on log10(md_med)
    y_fit = _safe_log10(y_md)
    ok = np.isfinite(x) & np.isfinite(y_fit)
    if ok.sum() >= 3:
        b, a = np.polyfit(x[ok], y_fit[ok], 1)
        xx = np.linspace(np.nanmin(x[ok]) - 0.02, np.nanmax(x[ok]) + 0.02, 200)
        ax.plot(xx, a + b * xx, color="#1f77b4", lw=2.0, alpha=0.9)

    ax.set_xlabel("2MASS XSC J−K (ext)")
    ax.set_ylabel("log10 median((Vobs/Vbar)^2)")
    ax.set_title("GR+baryons mass discrepancy vs J−K")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.02,
        f"quality n={len(df_q)}\nSpearman r={md_s.correlation:+.3f} (p={md_s.pvalue:.1e})\nPearson r={md_p.statistic:+.3f} (p={md_p.pvalue:.1e})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Panel B: bias in log-space
    ax = axes[1]
    ax.scatter(x_all, y_bias_all, s=14, c="#999999", alpha=0.25, edgecolors="none")
    ax.scatter(x, y_bias, s=18, c="#d62728", alpha=0.85, edgecolors="none")

    ok = np.isfinite(x) & np.isfinite(y_bias)
    if ok.sum() >= 3:
        b, a = np.polyfit(x[ok], y_bias[ok], 1)
        xx = np.linspace(np.nanmin(x[ok]) - 0.02, np.nanmax(x[ok]) + 0.02, 200)
        ax.plot(xx, a + b * xx, color="#d62728", lw=2.0, alpha=0.9)

    ax.set_xlabel("2MASS XSC J−K (ext)")
    ax.set_ylabel("median log10(Vobs/Vbar)")
    ax.set_title("GR+baryons bias vs J−K")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.02,
        f"quality n={len(df_q)}\nSpearman r={bias_s.correlation:+.3f} (p={bias_s.pvalue:.1e})\nPearson r={bias_p.statistic:+.3f} (p={bias_p.pvalue:.1e})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
