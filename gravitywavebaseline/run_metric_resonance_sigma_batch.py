"""
Phase-2: batch sigma-gated metric-resonance test on SPARC galaxies.

This script:
  1) Loads the Milky Way metric-resonance fit (A, lambda_peak, sigma_ln_lambda, ...).
  2) Loops over a list of SPARC rotmod files.
  3) Looks up each galaxy's sigma_v from a SPARC summary CSV.
  4) Applies the sigma-gated metric-resonance multiplier.
  5) Writes a CSV with GR vs resonance RMS for each galaxy.

Usage example
-------------
python gravitywavebaseline/run_metric_resonance_sigma_batch.py \
    --rotmod-dir data/Rotmod_LTG \
    --sparc-summary data/sparc/sparc_combined.csv \
    --mw-fit-json gravitywavebaseline/metric_resonance_mw_fit.json \
    --out-csv gravitywavebaseline/metric_resonance_sigma_batch_beta0p6.csv \
    --sigma-ref 30.0 \
    --beta-sigma 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from metric_resonance_multiplier import (
    metric_resonance_multiplier_sigma,
    sigma_gate_amplitude,
)


# --- Helpers ---------------------------------------------------------------


def load_sparc_rotmod(rotmod_path: str) -> pd.DataFrame:
    """
    Minimal loader for SPARC Rotmod LTG files.

    Expected columns (after skipping comments):
        Rad  Vobs  errV  Vgas  Vdisk  Vbul ...

    Returns DataFrame with:
        R_kpc, V_obs, V_gr
    """

    df = pd.read_csv(
        rotmod_path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
    )
    if df.empty:
        raise RuntimeError(f"No data parsed from {rotmod_path}")

    V_gr = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    df["V_gr"] = V_gr
    return df[["R_kpc", "V_obs", "V_gr"]]


def lookup_sigma_v(
    galaxy_name: str,
    sparc_summary: pd.DataFrame,
    *,
    col_name_galaxy: str,
    col_name_sigma: str,
) -> float:
    """
    Look up sigma_v [km/s] for a galaxy in a SPARC summary DataFrame.
    """

    key = galaxy_name.strip().lower()
    series = sparc_summary[col_name_galaxy].astype(str).str.strip().str.lower()
    mask = series == key
    sub = sparc_summary.loc[mask]
    if sub.empty:
        raise KeyError(
            f"Galaxy {galaxy_name!r} not found in summary "
            f"(column {col_name_galaxy!r})."
        )
    return float(sub[col_name_sigma].to_numpy(dtype=float)[0])


def rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def infer_galaxy_name_from_path(rotmod_path: Path) -> str:
    """
    Infer galaxy name from a file like NGC2403_rotmod.dat -> 'NGC2403'.
    """

    stem = rotmod_path.stem  # e.g. 'NGC2403_rotmod'
    if stem.endswith("_rotmod"):
        return stem[:-7]
    return stem


def load_galaxy_list(list_path: Path) -> List[str]:
    names: List[str] = []
    for line in list_path.read_text().splitlines():
        name = line.strip()
        if name:
            names.append(name)
    return names


# --- Main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch sigma-gated metric resonance on SPARC rotmod files."
    )
    parser.add_argument(
        "--rotmod-dir",
        type=str,
        required=True,
        help="Directory containing SPARC Rotmod LTG files (e.g., data/Rotmod_LTG).",
    )
    parser.add_argument(
        "--sparc-summary",
        type=str,
        required=True,
        help="SPARC summary CSV with galaxy- and sigma_v-level columns.",
    )
    parser.add_argument(
        "--summary-galaxy-col",
        type=str,
        default="galaxy_name",
        help="Column name in the SPARC summary for galaxy identifiers.",
    )
    parser.add_argument(
        "--summary-sigma-col",
        type=str,
        default="sigma_velocity",
        help="Column name in the SPARC summary for sigma_v [km/s].",
    )
    parser.add_argument(
        "--mw-fit-json",
        type=str,
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
        help="Milky Way metric resonance fit JSON from Phase-1.",
    )
    parser.add_argument(
        "--galaxy-list",
        type=str,
        default=None,
        help=(
            "Optional text file listing galaxy names to include (one per line). "
            "If omitted, all *_rotmod.dat files in --rotmod-dir are used."
        ),
    )
    parser.add_argument(
        "--sigma-ref",
        type=float,
        default=30.0,
        help="Reference dispersion sigma_ref [km/s] for sigma gating.",
    )
    parser.add_argument(
        "--beta-sigma",
        type=float,
        default=0.4,
        help="Exponent beta for sigma gating: A_eff ∝ (sigma_ref/sigma_v)^beta.",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="If set, allow A_eff > A_base (not recommended for first passes).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path for per-galaxy RMS results.",
    )
    args = parser.parse_args()

    rotmod_dir = Path(args.rotmod_dir)
    if not rotmod_dir.is_dir():
        raise NotADirectoryError(f"Rotmod directory not found: {rotmod_dir}")

    # Load MW fit
    fit_path = Path(args.mw_fit_json)
    fit = json.loads(fit_path.read_text())
    A_base = float(fit["A"])
    ell0_kpc = float(fit["ell0_kpc"])
    p = float(fit["p"])
    n_coh = float(fit["n_coh"])
    lambda_peak_kpc = float(fit["lambda_peak_kpc"])
    sigma_ln_lambda = float(fit["sigma_ln_lambda"])

    print("[info] MW metric-resonance fit loaded:")
    print(
        "       A={:.3f}, lambda_peak={:.2f} kpc, sigma_ln_lambda={:.3f}, "
        "ell0={:.2f} kpc, p={:.3f}, n_coh={:.3f}".format(
            A_base, lambda_peak_kpc, sigma_ln_lambda, ell0_kpc, p, n_coh
        )
    )

    # Determine galaxy list
    if args.galaxy_list:
        list_path = Path(args.galaxy_list)
        if not list_path.is_file():
            raise FileNotFoundError(f"Galaxy list file not found: {list_path}")
        galaxy_names = load_galaxy_list(list_path)
        rotmod_paths = [rotmod_dir / f"{name}_rotmod.dat" for name in galaxy_names]
        print(f"[info] Loaded {len(galaxy_names)} galaxies from {list_path}")
    else:
        rotmod_paths = sorted(rotmod_dir.glob("*_rotmod.dat"))
        if not rotmod_paths:
            raise RuntimeError(f"No *_rotmod.dat files found in {rotmod_dir}")
        galaxy_names = [infer_galaxy_name_from_path(p) for p in rotmod_paths]
        print(f"[info] Found {len(rotmod_paths)} rotmod files in {rotmod_dir}")

    # Load SPARC summary once
    sparc_summary = pd.read_csv(args.sparc_summary)
    if args.summary_galaxy_col not in sparc_summary.columns:
        raise KeyError(
            f"Column {args.summary_galaxy_col!r} missing from {args.sparc_summary}"
        )
    if args.summary_sigma_col not in sparc_summary.columns:
        raise KeyError(
            f"Column {args.summary_sigma_col!r} missing from {args.sparc_summary}"
        )

    rows = []

    for gal_name, rpath in zip(galaxy_names, rotmod_paths):
        print(f"\n[galaxy] {gal_name}  ({rpath.name})")

        if not rpath.exists():
            print("  [WARN] Rotmod file missing, skipping.")
            continue

        try:
            df = load_sparc_rotmod(str(rpath))
        except Exception as exc:
            print(f"  [WARN] Failed to load rotmod: {exc}")
            continue

        R = df["R_kpc"].to_numpy(dtype=float)
        V_obs = df["V_obs"].to_numpy(dtype=float)
        V_gr = df["V_gr"].to_numpy(dtype=float)

        if len(df) == 0 or not np.any(V_gr > 0):
            print("  [WARN] No valid V_gr values, skipping.")
            continue

        rms_gr = rms(V_obs - V_gr)
        print(f"  GR-only RMS: {rms_gr:.2f} km/s")

        try:
            sigma_v = lookup_sigma_v(
                gal_name,
                sparc_summary,
                col_name_galaxy=args.summary_galaxy_col,
                col_name_sigma=args.summary_sigma_col,
            )
        except Exception as exc:
            print(
                f"  [WARN] sigma_v lookup failed ({exc}); "
                f"falling back to sigma_ref={args.sigma_ref:.1f}"
            )
            sigma_v = args.sigma_ref

        lambda_orb = 2.0 * np.pi * R
        f_res = metric_resonance_multiplier_sigma(
            R_kpc=R,
            lambda_orb_kpc=lambda_orb,
            sigma_v_kms=sigma_v,
            A_base=A_base,
            sigma_ref_kms=args.sigma_ref,
            beta_sigma=args.beta_sigma,
            ell0_kpc=ell0_kpc,
            p=p,
            n_coh=n_coh,
            lambda_peak_kpc=lambda_peak_kpc,
            sigma_ln_lambda=sigma_ln_lambda,
            clamp=not args.no_clamp,
        )
        V_model = V_gr * np.sqrt(np.clip(f_res, 0.0, None))
        rms_res = rms(V_obs - V_model)

        A_eff = sigma_gate_amplitude(
            sigma_v_kms=sigma_v,
            A_base=A_base,
            sigma_ref_kms=args.sigma_ref,
            beta_sigma=args.beta_sigma,
            clamp=not args.no_clamp,
        )

        print(
            f"  sigma_v={sigma_v:.2f} km/s -> A_eff={A_eff:.3f}, "
            f"RMS(metric_res)={rms_res:.2f} km/s"
        )

        rows.append(
            {
                "galaxy": gal_name,
                "rotmod_path": str(rpath),
                "sigma_v": sigma_v,
                "sigma_ref": args.sigma_ref,
                "beta_sigma": args.beta_sigma,
                "A_base": A_base,
                "A_eff": A_eff,
                "n_points": int(len(df)),
                "rms_gr": rms_gr,
                "rms_metric_resonance_sigma": rms_res,
                "delta_rms": rms_res - rms_gr,
            }
        )

    if not rows:
        raise RuntimeError("No galaxies were successfully processed; nothing to write.")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_df.to_csv(out_path, index=False)
    print(f"\n[OK] Wrote batch results to {out_path}")

    mean_delta = out_df["delta_rms"].mean()
    improved = int((out_df["delta_rms"] < 0).sum())
    print(
        f"[summary] Galaxies: {len(out_df)}, improved: {improved}, "
        f"mean delta_RMS = {mean_delta:.2f} km/s"
    )


if __name__ == "__main__":
    main()


