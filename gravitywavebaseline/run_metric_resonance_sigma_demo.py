"""
Phase-2 demo: apply the MW metric-resonance kernel with sigma gating
to one or more SPARC galaxies.

Usage examples
--------------
Single galaxy with explicit sigma_v:
    python gravitywavebaseline/run_metric_resonance_sigma_demo.py \
        --rotmod-path data/Rotmod_LTG/NGC2403_rotmod.dat \
        --sigma-v 20.0 \
        --mw-fit-json gravitywavebaseline/metric_resonance_mw_fit.json

Use sigma_v from a SPARC summary table:
    python gravitywavebaseline/run_metric_resonance_sigma_demo.py \
        --rotmod-path data/Rotmod_LTG/NGC2403_rotmod.dat \
        --galaxy-name NGC2403 \
        --sparc-summary data/sparc/sparc_combined.csv \
        --mw-fit-json gravitywavebaseline/metric_resonance_mw_fit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_resonance_multiplier import (
    metric_resonance_multiplier_sigma,
    sigma_gate_amplitude,
)


def load_sparc_rotmod(rotmod_path: str) -> pd.DataFrame:
    """
    Minimal loader for SPARC Rotmod LTG files.

    Expected columns (after skipping comments):
        Rad  Vobs  errV  Vgas  Vdisk  Vbul ...

    Returns a DataFrame with columns:
        R_kpc, V_obs, V_gr
    """

    df = pd.read_csv(
        rotmod_path,
        delim_whitespace=True,
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


def rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def lookup_sigma_v(
    galaxy_name: str,
    sparc_summary_path: str,
    col_name_galaxy: str = "galaxy",
    col_name_sigma: str = "sigma_v",
) -> float:
    """
    Look up sigma_v [km/s] for a galaxy in a SPARC summary CSV.

    The default column names assume a table like:
        galaxy, sigma_v, ...
    """

    df = pd.read_csv(sparc_summary_path)
    mask = df[col_name_galaxy].astype(str).str.strip() == galaxy_name.strip()
    sub = df.loc[mask]
    if sub.empty:
        raise KeyError(
            f"Galaxy {galaxy_name!r} not found in {sparc_summary_path} "
            f"(using column {col_name_galaxy!r})."
        )
    sigma_vals = sub[col_name_sigma].to_numpy(dtype=float)
    return float(sigma_vals[0])


def main():
    parser = argparse.ArgumentParser(
        description="Apply MW metric resonance with sigma gating to a SPARC galaxy."
    )
    parser.add_argument(
        "--rotmod-path",
        required=True,
        help=(
            "Path to SPARC Rotmod LTG file "
            "(e.g., data/Rotmod_LTG/NGC2403_rotmod.dat)."
        ),
    )
    parser.add_argument(
        "--mw-fit-json",
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
        help="Path to MW metric resonance fit JSON.",
    )

    # Sigma handling: either explicit value or via SPARC summary
    parser.add_argument(
        "--sigma-v",
        type=float,
        default=None,
        help=(
            "Galaxy-level velocity dispersion [km/s]. "
            "If omitted, --galaxy-name + --sparc-summary must be provided."
        ),
    )
    parser.add_argument(
        "--galaxy-name",
        type=str,
        default=None,
        help="Galaxy name key to look up sigma_v in the SPARC summary.",
    )
    parser.add_argument(
        "--sparc-summary",
        type=str,
        default=None,
        help="Path to SPARC summary CSV with sigma_v column.",
    )

    parser.add_argument(
        "--sigma-ref",
        type=float,
        default=30.0,
        help="Reference dispersion [km/s] for sigma gating.",
    )
    parser.add_argument(
        "--beta-sigma",
        type=float,
        default=0.4,
        help=(
            "Exponent beta for sigma gating "
            "(A_eff proportional to (sigma_ref/sigma)**beta)."
        ),
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="If set, do NOT clamp A_eff <= A_base.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional output JSON path for summary metrics.",
    )
    args = parser.parse_args()

    # Resolve sigma_v
    if args.sigma_v is not None:
        sigma_v = args.sigma_v
        sigma_source = "explicit"
    else:
        if not (args.galaxy_name and args.sparc_summary):
            raise SystemExit(
                "Either provide --sigma-v explicitly OR "
                "--galaxy-name and --sparc-summary."
            )
        sigma_v = lookup_sigma_v(args.galaxy_name, args.sparc_summary)
        sigma_source = f"lookup:{args.galaxy_name}"
    print(f"[info] Using sigma_v = {sigma_v:.2f} km/s ({sigma_source})")

    # Load SPARC rotmod file
    df = load_sparc_rotmod(args.rotmod_path)
    R = df["R_kpc"].to_numpy(dtype=float)
    V_obs = df["V_obs"].to_numpy(dtype=float)
    V_gr = df["V_gr"].to_numpy(dtype=float)

    rms_gr = rms(V_obs - V_gr)
    print(f"[info] GR-only RMS: {rms_gr:.2f} km/s")

    # Load MW Phase-1 fit
    fit = json.loads(Path(args.mw_fit_json).read_text())
    A_base = float(fit["A"])
    ell0_kpc = float(fit["ell0_kpc"])
    p = float(fit["p"])
    n_coh = float(fit["n_coh"])
    lambda_peak_kpc = float(fit["lambda_peak_kpc"])
    sigma_ln_lambda = float(fit["sigma_ln_lambda"])

    # Compute sigma-gated amplitude
    A_eff = sigma_gate_amplitude(
        sigma_v_kms=sigma_v,
        A_base=A_base,
        sigma_ref_kms=args.sigma_ref,
        beta_sigma=args.beta_sigma,
        clamp=not args.no_clamp,
    )
    print(f"[info] A_base = {A_base:.3f}, A_eff(sigma_v) = {A_eff:.3f}")

    # Apply sigma-gated metric resonance
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
    print(f"[info] Sigma-gated metric resonance RMS: {rms_res:.2f} km/s")

    if args.out_json:
        out = {
            "rotmod_path": args.rotmod_path,
            "sigma_v": sigma_v,
            "sigma_ref": args.sigma_ref,
            "beta_sigma": args.beta_sigma,
            "A_base": A_base,
            "A_eff": A_eff,
            "rms_gr": rms_gr,
            "rms_metric_resonance_sigma": rms_res,
            "n_points": int(len(df)),
            "mw_fit_params": fit,
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[info] Wrote summary to {args.out_json}")


if __name__ == "__main__":
    main()


