"""
Apply the Milky Way metric-resonance fit to a SPARC galaxy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric_resonance_multiplier import metric_resonance_multiplier


def load_sparc_rotmod(rotmod_path):
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


def rms(residuals):
    return float(np.sqrt(np.mean(residuals**2)))


def main():
    parser = argparse.ArgumentParser(
        description="Apply MW metric resonance kernel to a SPARC galaxy."
    )
    parser.add_argument(
        "--rotmod-path",
        required=True,
        help="Path to SPARC Rotmod LTG file.",
    )
    parser.add_argument(
        "--mw-fit-json",
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
        help="Path to MW metric resonance fit JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output summary JSON path.",
    )
    args = parser.parse_args()

    df = load_sparc_rotmod(args.rotmod_path)
    R = df["R_kpc"].to_numpy(dtype=float)
    V_obs = df["V_obs"].to_numpy(dtype=float)
    V_gr = df["V_gr"].to_numpy(dtype=float)

    rms_gr = rms(V_obs - V_gr)
    print(f"GR-only RMS: {rms_gr:.2f} km/s")

    fit = json.loads(Path(args.mw_fit_json).read_text())
    f_res = metric_resonance_multiplier(
        R_kpc=R,
        lambda_orb_kpc=2.0 * np.pi * R,
        A=fit["A"],
        ell0_kpc=fit["ell0_kpc"],
        p=fit["p"],
        n_coh=fit["n_coh"],
        lambda_peak_kpc=fit["lambda_peak_kpc"],
        sigma_ln_lambda=fit["sigma_ln_lambda"],
    )
    V_model = V_gr * np.sqrt(np.clip(f_res, 0.0, None))
    rms_res = rms(V_obs - V_model)
    print(f"Metric resonance RMS: {rms_res:.2f} km/s")

    if args.out_json:
        out = {
            "rotmod_path": args.rotmod_path,
            "rms_gr": rms_gr,
            "rms_metric_resonance": rms_res,
            "n_points": len(df),
            "params": fit,
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote summary to {args.out_json}")


if __name__ == "__main__":
    main()
