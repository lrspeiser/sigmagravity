#!/usr/bin/env python3
"""Fit coherence/decoherence test models to SPARC rotation curves.

Data provenance
---------------
- Real observational data: SPARC rotation-curve reconstructions stored
  under gravitywavebaseline/sparc_results/*.json (read-only).
- Real galaxy metadata: data/sparc/sparc_combined.csv for velocity
  dispersions (read-only).
- Derived quantities: orbital wavelength lambda ~= 2*pi*R (from the SPARC
  radii) and baryonic acceleration g_bar = V_GR^2 / R.

This script lives in the speculative coherence tests workspace and
writes summaries to coherence tests/results/ without touching the root
science data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from coherence_models import apply_coherence_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GALAXIES = [
    "NGC2403",
    "NGC3198",
    "NGC5055",
]

MODEL_SPECS: Dict[str, Dict] = {
    "path_interference": {
        "variables": {
            "A": 0.6,
            "ell0": 5.0,
            "beta_sigma": 0.4,
        },
        "bounds": {
            "A": (0.0, 5.0),
            "ell0": (0.5, 50.0),
            "beta_sigma": (0.0, 2.0),
        },
        "fixed": {
            "p": 0.9,
            "n_coh": 0.8,
            "sigma_ref": 30.0,
        },
    },
    "metric_resonance": {
        "variables": {
            "A": 0.5,
            "ell0": 4.0,
            "lambda_m0": 8.0,
            "log_width": 0.8,
        },
        "bounds": {
            "A": (0.0, 5.0),
            "ell0": (0.5, 50.0),
            "lambda_m0": (0.5, 80.0),
            "log_width": (0.1, 2.0),
        },
        "fixed": {
            "p": 1.0,
            "n_coh": 1.0,
            "beta_sigma": 0.0,
        },
    },
    "entanglement": {
        "variables": {
            "A": 0.7,
            "ell0": 5.0,
            "sigma0": 25.0,
        },
        "bounds": {
            "A": (0.0, 5.0),
            "ell0": (0.5, 50.0),
            "sigma0": (5.0, 80.0),
        },
        "fixed": {
            "p": 1.0,
            "n_coh": 1.0,
        },
    },
    "vacuum_condensation": {
        "variables": {
            "A": 0.7,
            "ell0": 6.0,
            "sigma_c": 35.0,
            "alpha": 2.0,
        },
        "bounds": {
            "A": (0.0, 5.0),
            "ell0": (0.5, 60.0),
            "sigma_c": (5.0, 120.0),
            "alpha": (0.5, 6.0),
        },
        "fixed": {
            "p": 1.0,
            "n_coh": 1.0,
            "beta": 1.0,
        },
    },
    "graviton_pairing": {
        "variables": {
            "A": 0.7,
            "xi0": 5.0,
            "sigma0": 25.0,
            "gamma_xi": 0.4,
        },
        "bounds": {
            "A": (0.0, 5.0),
            "xi0": (0.5, 60.0),
            "sigma0": (5.0, 80.0),
            "gamma_xi": (0.0, 2.5),
        },
        "fixed": {
            "p": 1.0,
            "n_coh": 1.0,
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_sigma_lookup(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["galaxy_name"], df["sigma_velocity"]))


def load_profile(galaxy: str, sparc_dir: Path) -> Dict[str, np.ndarray]:
    raw_path = sparc_dir / f"{galaxy}_capacity_test.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing SPARC profile for {galaxy}: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    radii: List[float] = []
    g_bar: List[float] = []
    lambda_vals: List[float] = []
    v_obs: List[float] = []
    v_gr: List[float] = []
    multipliers: List[float] = []

    for row in payload.get("data", []):
        R_kpc = float(row["R"])
        v_obs_kms = float(row["V_obs"])
        v_gr_kms = float(row["V_GR"])
        if R_kpc <= 0 or v_gr_kms <= 1.0:
            continue
        ratio = v_obs_kms / v_gr_kms
        if not np.isfinite(ratio):
            continue
        radii.append(R_kpc)
        g_bar.append((v_gr_kms ** 2) / (R_kpc + 1e-12))
        lambda_vals.append(2.0 * np.pi * R_kpc)
        v_obs.append(v_obs_kms)
        v_gr.append(v_gr_kms)
        multipliers.append(ratio ** 2)

    return {
        "R": np.asarray(radii),
        "g_bar": np.asarray(g_bar),
        "lambda_gw": np.asarray(lambda_vals),
        "v_obs": np.asarray(v_obs),
        "v_gr": np.asarray(v_gr),
        "mult": np.asarray(multipliers),
    }


def fit_model(model_name: str, profile: Dict[str, np.ndarray], sigma_v: float):
    spec = MODEL_SPECS[model_name]
    var_names = list(spec["variables"].keys())
    x0 = np.array([spec["variables"][k] for k in var_names], dtype=float)
    lower = np.array([spec["bounds"][k][0] for k in var_names], dtype=float)
    upper = np.array([spec["bounds"][k][1] for k in var_names], dtype=float)

    def vector_to_params(vec: np.ndarray) -> Dict[str, float]:
        params = dict(spec.get("fixed", {}))
        params.update({name: float(val) for name, val in zip(var_names, vec)})
        return params

    def residuals(vec: np.ndarray) -> np.ndarray:
        params = vector_to_params(vec)
        f_model = apply_coherence_model(
            model_name,
            profile["R"],
            profile["g_bar"],
            profile["lambda_gw"],
            sigma_v,
            params,
        )
        return f_model - profile["mult"]

    result = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=5000)
    best_params = vector_to_params(result.x)
    final_residuals = residuals(result.x)
    rms_multiplier = float(np.sqrt(np.mean(final_residuals ** 2)))
    f_model = np.clip(apply_coherence_model(
        model_name,
        profile["R"],
        profile["g_bar"],
        profile["lambda_gw"],
        sigma_v,
        best_params,
    ), 0.0, None)
    v_model = np.sqrt(f_model) * profile["v_gr"]
    velocity_error = float(np.sqrt(np.mean((v_model - profile["v_obs"]) ** 2)))

    return {
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "rms_multiplier": rms_multiplier,
        "rms_velocity": velocity_error,
        "params": best_params,
    }


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main():
    repo_root = Path(__file__).resolve().parent.parent
    workspace = Path(__file__).resolve().parent
    sparc_dir = repo_root / "gravitywavebaseline" / "sparc_results"
    sparc_meta = repo_root / "data" / "sparc" / "sparc_combined.csv"
    results_dir = workspace / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sigma_lookup = load_sigma_lookup(sparc_meta)

    summary_rows = []

    for galaxy in GALAXIES:
        profile = load_profile(galaxy, sparc_dir)
        sigma_v = float(sigma_lookup[galaxy])

        for model_name in MODEL_SPECS.keys():
            fit = fit_model(model_name, profile, sigma_v)
            summary_rows.append({
                "galaxy": galaxy,
                "sigma_v": sigma_v,
                "model": model_name,
                **fit,
            })
            print(f"{galaxy:8s} | {model_name:18s} | RMS(mult)={fit['rms_multiplier']:.3f} | RMS(v)={fit['rms_velocity']:.2f} km/s")

    summary_path = results_dir / "sparc_coherence_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    md_lines = [
        "# SPARC coherence fit summary",
        "",
        "Real data: SPARC rotation curves (gravitywavebaseline/sparc_results).",
        "sigma_v source: data/sparc/sparc_combined.csv.",
        "Derived lambda: 2*pi*R (orbital circumference).",
        "",
        "| Galaxy | sigma_v (km/s) | Model | RMS multiplier | RMS velocity (km/s) |",
        "|--------|---------------:|:------|----------------:|--------------------:|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['galaxy']} | {row['sigma_v']:.2f} | {row['model']} | "
            f"{row['rms_multiplier']:.4f} | {row['rms_velocity']:.2f} |"
        )
    md_lines.append("")

    md_path = results_dir / "sparc_coherence_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
