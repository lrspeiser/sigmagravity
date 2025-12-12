#!/usr/bin/env python3
"""Analyze whether galaxy color correlates with Σ-Gravity rotation-curve residuals.

This answers: does WISE W1–W2 (a proxy for dust/AGN / stellar population) correlate with
how well Σ-Gravity predicts *star speeds* (SPARC rotation curves)?

Inputs:
- SPARC rotmod files in data/Rotmod_LTG/ (loaded via scripts/run_regression_extended.py)
- WISE-derived color table: resources/photometry/sparc_allwise_matches.csv

Outputs:
- resources/photometry/sparc_color_sigma_residuals.csv

Notes:
- We DO NOT tune per galaxy.
- We evaluate correlation between an observable (color) and model residuals.
- If correlation exists mainly for very red W1–W2, that usually indicates dust/AGN contamination
  (i.e., photometry-based baryonic mass systematics), not a new force-law parameter.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


def load_run_regression_extended(repo_root: Path):
    # Prevent writing/rewriting tracked __pycache__ artifacts in this repo.
    sys.dont_write_bytecode = True
    path = repo_root / "scripts" / "run_regression_extended.py"
    spec = importlib.util.spec_from_file_location("run_regression_extended", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def choose_color(df: pd.DataFrame) -> pd.Series:
    """Prefer extended-source elliptical aperture color when available."""
    color = df["w1_w2_gmag"].copy()
    mask = color.isna()
    color.loc[mask] = df.loc[mask, "w1_w2_mpro"]
    return pd.to_numeric(color, errors="coerce")


def add_quality_flags(wise: pd.DataFrame) -> pd.DataFrame:
    out = wise.copy()
    cc = out.get("cc_flags", pd.Series(["" for _ in range(len(out))])).fillna("")
    ph = out.get("ph_qual", pd.Series(["" for _ in range(len(out))])).fillna("")
    out["cc_w1"] = cc.str.slice(0, 1)
    out["cc_w2"] = cc.str.slice(1, 2)
    out["ph_w1"] = ph.str.slice(0, 1)
    out["ph_w2"] = ph.str.slice(1, 2)

    # Conservative “clean match” heuristic.
    out["good_phot"] = (
        out["ph_w1"].isin(["A", "B"]) & out["ph_w2"].isin(["A", "B"]) & (out["cc_w1"] == "0") & (out["cc_w2"] == "0")
    )
    out["good_match"] = pd.to_numeric(out.get("sep_arcsec", np.nan), errors="coerce") <= 2.0
    out["good"] = out["good_phot"] & out["good_match"]
    return out


def compute_per_galaxy_metrics(rre: Any, data_dir: Path) -> pd.DataFrame:
    gals = rre.load_sparc(data_dir)
    rows: List[Dict[str, Any]] = []

    for g in gals:
        name = g["name"]
        R = np.asarray(g["R"], dtype=float)
        V_obs = np.asarray(g["V_obs"], dtype=float)
        V_bar = np.asarray(g["V_bar"], dtype=float)
        R_d = float(g["R_d"])
        h_disk = float(g.get("h_disk", 0.15 * R_d))
        f_bulge = float(g.get("f_bulge", 0.0))

        V_pred = np.asarray(rre.predict_velocity(R, V_bar, R_d, h_disk, f_bulge), dtype=float)
        V_mond = np.asarray(rre.predict_mond(R, V_bar), dtype=float)

        rms_sigma = float(np.sqrt(np.mean((V_pred - V_obs) ** 2)))
        rms_mond = float(np.sqrt(np.mean((V_mond - V_obs) ** 2)))

        valid = (V_obs > 0) & (V_pred > 0)
        if np.any(valid):
            log_ratio = np.log10(V_obs[valid] / V_pred[valid])
            med_log = float(np.median(log_ratio))
            rar_scatter = float(np.std(log_ratio))
        else:
            med_log = float("nan")
            rar_scatter = float("nan")

        rows.append(
            {
                "name": name,
                "n_points": int(len(R)),
                "rms_sigma_kms": rms_sigma,
                "rms_mond_kms": rms_mond,
                "delta_rms_sigma_minus_mond_kms": rms_sigma - rms_mond,
                "median_log10_Vobs_over_Vsigma": med_log,
                "rar_scatter_dex": rar_scatter,
            }
        )

    return pd.DataFrame(rows)


def corr(x: pd.Series, y: pd.Series) -> CorrResult:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df) < 5:
        return CorrResult(n=len(df), pearson_r=float("nan"), pearson_p=float("nan"), spearman_r=float("nan"), spearman_p=float("nan"))

    pr = stats.pearsonr(df["x"].values, df["y"].values)
    sr = stats.spearmanr(df["x"].values, df["y"].values)
    return CorrResult(n=len(df), pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue), spearman_r=float(sr.correlation), spearman_p=float(sr.pvalue))


def fmt(res: CorrResult) -> str:
    return (
        f"n={res.n:3d} | pearson r={res.pearson_r:+.3f} p={res.pearson_p:.3g} "
        f"| spearman r={res.spearman_r:+.3f} p={res.spearman_p:.3g}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-arcsec", type=float, default=60.0, help="(Informational) match radius used when building WISE table")
    ap.add_argument("--out", default="resources/photometry/sparc_color_sigma_residuals.csv", help="Output CSV path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    rre = load_run_regression_extended(repo_root)

    data_dir = repo_root / "data"
    metrics = compute_per_galaxy_metrics(rre, data_dir)

    wise_path = repo_root / "resources" / "photometry" / "sparc_allwise_matches.csv"
    wise = pd.read_csv(wise_path)
    wise = add_quality_flags(wise)
    wise["color_w1_w2"] = choose_color(wise)

    # Merge
    df = metrics.merge(
        wise[
            [
                "name",
                "color_w1_w2",
                "w1_w2_gmag",
                "w1_w2_mpro",
                "ext_flg",
                "sep_arcsec",
                "cc_flags",
                "ph_qual",
                "good",
                "good_phot",
                "good_match",
            ]
        ],
        on="name",
        how="left",
    )

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Report correlations
    color = df["color_w1_w2"]

    print("SPARC×AllWISE color correlation test")
    print(f"- galaxies: {len(df)}")
    print(f"- WISE table: {wise_path}")
    print(f"- output: {out_path}")
    print(f"- note: W1–W2 uses gmag when available, else mpro (match radius used previously: {args.radius_arcsec:.0f}\")")

    def subset(label: str, sdf: pd.DataFrame) -> None:
        print(f"\n[{label}]")
        print("rms_sigma vs color:", fmt(corr(sdf["color_w1_w2"], sdf["rms_sigma_kms"])))
        print("delta_rms (sigma - mond) vs color:", fmt(corr(sdf["color_w1_w2"], sdf["delta_rms_sigma_minus_mond_kms"])))
        print("bias (median log10 Vobs/Vsigma) vs color:", fmt(corr(sdf["color_w1_w2"], sdf["median_log10_Vobs_over_Vsigma"])))

    subset("all", df)
    subset("good (sep<=2\" & W1/W2 cc=0 & ph_qual in {A,B})", df[df["good"] == True])
    subset("gmag-only (extended-source ellipse mags)", df[df["w1_w2_gmag"].notna()])
    subset("gmag-only + good", df[(df["w1_w2_gmag"].notna()) & (df["good"] == True)])

    # Clip out dusty/AGN colors (heuristic)
    subset("good + color<=0.30 (exclude very red W1–W2)", df[(df["good"] == True) & (df["color_w1_w2"] <= 0.30)])


if __name__ == "__main__":
    main()
