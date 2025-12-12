#!/usr/bin/env python3
"""Test whether a stellar-population color (2MASS XSC J−K) correlates with dynamics residuals.

This is the "color that traces stellar M/L" follow-up to the WISE W1–W2 analysis.
We use 2MASS XSC integrated magnitudes (J.ext and K.ext) via the prebuilt table:
  resources/photometry/sparc_2mass_xsc_matches.csv

We then test correlations against per-galaxy residual metrics computed from SPARC:
- GR+baryons baseline (V_GR = V_bar)
- Σ-Gravity prediction
- MOND prediction (for reference)

Outputs:
- resources/photometry/sparc_2mass_xsc_color_residuals.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

# Avoid writing tracked __pycache__ artifacts in this repo.
sys.dont_write_bytecode = True


@dataclass(frozen=True)
class CorrResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


def load_run_regression_extended(repo_root: Path):
    path = repo_root / "scripts" / "run_regression_extended.py"
    spec = importlib.util.spec_from_file_location("run_regression_extended", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corr(x: pd.Series, y: pd.Series) -> CorrResult:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df) < 10:
        return CorrResult(n=len(df), pearson_r=float("nan"), pearson_p=float("nan"), spearman_r=float("nan"), spearman_p=float("nan"))

    pr = stats.pearsonr(df["x"].values, df["y"].values)
    sr = stats.spearmanr(df["x"].values, df["y"].values)
    return CorrResult(n=len(df), pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue), spearman_r=float(sr.correlation), spearman_p=float(sr.pvalue))


def fmt(res: CorrResult) -> str:
    return (
        f"n={res.n:3d} | pearson r={res.pearson_r:+.3f} p={res.pearson_p:.3g} "
        f"| spearman r={res.spearman_r:+.3f} p={res.spearman_p:.3g}"
    )


def compute_sparc_metrics(rre: Any, data_dir: Path) -> pd.DataFrame:
    gals = rre.load_sparc(data_dir)
    rows: List[Dict[str, float]] = []

    for g in gals:
        name = str(g["name"])
        R = np.asarray(g["R"], dtype=float)
        V_obs = np.asarray(g["V_obs"], dtype=float)
        V_bar = np.asarray(g["V_bar"], dtype=float)
        R_d = float(g["R_d"])
        h_disk = float(g.get("h_disk", 0.15 * R_d))
        f_bulge = float(g.get("f_bulge", 0.0))

        # Σ-Gravity
        V_sig = np.asarray(rre.predict_velocity(R, V_bar, R_d, h_disk, f_bulge), dtype=float)
        rms_sig = float(np.sqrt(np.mean((V_sig - V_obs) ** 2)))
        valid_sig = (V_obs > 0) & (V_sig > 0)
        med_log_sig = float(np.median(np.log10(V_obs[valid_sig] / V_sig[valid_sig]))) if np.any(valid_sig) else float("nan")

        # MOND
        V_mond = np.asarray(rre.predict_mond(R, V_bar), dtype=float)
        rms_mond = float(np.sqrt(np.mean((V_mond - V_obs) ** 2)))

        # GR+baryons
        valid_gr = (V_obs > 0) & (V_bar > 0)
        rms_gr = float(np.sqrt(np.mean((V_bar[valid_gr] - V_obs[valid_gr]) ** 2))) if np.any(valid_gr) else float("nan")
        med_log_gr = float(np.median(np.log10(V_obs[valid_gr] / V_bar[valid_gr]))) if np.any(valid_gr) else float("nan")
        md_med = float(np.median((V_obs[valid_gr] / V_bar[valid_gr]) ** 2)) if np.any(valid_gr) else float("nan")

        rows.append(
            {
                "name": name,
                "n_points": int(len(R)),
                "rms_gr_kms": rms_gr,
                "median_log10_Vobs_over_Vbar": med_log_gr,
                "md_med": md_med,
                "rms_sigma_kms": rms_sig,
                "rms_mond_kms": rms_mond,
                "delta_rms_sigma_minus_mond_kms": rms_sig - rms_mond,
                "median_log10_Vobs_over_Vsigma": med_log_sig,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xsc", default="resources/photometry/sparc_2mass_xsc_matches.csv", help="2MASS XSC match table")
    ap.add_argument("--out", default="resources/photometry/sparc_2mass_xsc_color_residuals.csv", help="Output merged table")
    ap.add_argument("--e-max", type=float, default=0.10, help="Max photometric error for J and K")
    ap.add_argument("--sep-max", type=float, default=10.0, help="Max separation (arcsec) for match quality")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    rre = load_run_regression_extended(repo_root)

    metrics = compute_sparc_metrics(rre, repo_root / "data")
    xsc = pd.read_csv(repo_root / args.xsc)

    xsc["J_minus_K_ext"] = pd.to_numeric(xsc.get("J_minus_K_ext"), errors="coerce")
    xsc["e_J_ext_mag"] = pd.to_numeric(xsc.get("e_J_ext_mag"), errors="coerce")
    xsc["e_K_ext_mag"] = pd.to_numeric(xsc.get("e_K_ext_mag"), errors="coerce")
    xsc["sep_arcsec"] = pd.to_numeric(xsc.get("sep_arcsec"), errors="coerce")

    merged = metrics.merge(
        xsc[["name", "match_found", "2masx", "sep_arcsec", "J_ext_mag", "e_J_ext_mag", "K_ext_mag", "e_K_ext_mag", "J_minus_K_ext"]],
        on="name",
        how="left",
    )

    # Quality mask
    merged["jk_good"] = (
        (merged["match_found"] == True)
        & merged["J_minus_K_ext"].notna()
        & (merged["e_J_ext_mag"] <= float(args.e_max))
        & (merged["e_K_ext_mag"] <= float(args.e_max))
        & (merged["sep_arcsec"] <= float(args.sep_max))
    )

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("2MASS XSC J−K vs dynamics residuals (SPARC)")
    print(f"- SPARC galaxies (quality-cut RCs): {len(metrics)}")
    print(f"- XSC matches: {int((merged['match_found']==True).sum())} / {len(merged)}")
    print(f"- J−K available: {int(merged['J_minus_K_ext'].notna().sum())}")
    print(f"- Quality cut: eJ,eK <= {args.e_max} mag and sep <= {args.sep_max}\"")
    print(f"- Passing quality cut: {int(merged['jk_good'].sum())}")
    print(f"- Output: {out_path}")

    def show(label: str, sdf: pd.DataFrame) -> None:
        print(f"\n[{label}]")
        print("GR mismatch (md_med) vs J-K:", fmt(corr(sdf["J_minus_K_ext"], sdf["md_med"])))
        print("GR bias (median log10 Vobs/Vbar) vs J-K:", fmt(corr(sdf["J_minus_K_ext"], sdf["median_log10_Vobs_over_Vbar"])))
        print("GR RMS (km/s) vs J-K:", fmt(corr(sdf["J_minus_K_ext"], sdf["rms_gr_kms"])))
        print("Σ RMS (km/s) vs J-K:", fmt(corr(sdf["J_minus_K_ext"], sdf["rms_sigma_kms"])))
        print("ΔRMS (Σ−MOND) vs J-K:", fmt(corr(sdf["J_minus_K_ext"], sdf["delta_rms_sigma_minus_mond_kms"])))

    show("all XSC matches", merged[merged["match_found"] == True])
    show("quality-filtered", merged[merged["jk_good"] == True])


if __name__ == "__main__":
    main()
