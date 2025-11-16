"""
Run Burr-XII-based coherence models against real SPARC rotation curves.

Workflow:
  1. Load galaxy metadata from data/sparc/sparc_combined.csv
  2. Load rotation curves from data/Rotmod_LTG/*_rotmod.dat (real data)
  3. Fit each microphysics model so that g_eff = g_bar * f_model
     reproduces the observed velocities.
  4. Save metrics/parameters under coherence tests/results/.

We only READ the existing SPARC data; nothing in data/ is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from coherence_models import MODEL_REGISTRY  # noqa: E402


def softplus(x: float, floor: float = 1e-6) -> float:
    if x > 50:
        return x + floor
    return math.log1p(math.exp(x)) + floor


def inv_softplus(y: float, floor: float = 1e-6) -> float:
    y_adj = max(y - floor, 1e-9)
    if y_adj > 50:
        return y_adj
    return math.log(math.expm1(y_adj))


def compute_lambda_values(radii: np.ndarray, mode: str = "orbital") -> np.ndarray:
    R_safe = np.maximum(np.asarray(radii, dtype=np.float64), 1e-3)
    mode = mode.lower()
    if mode == "orbital":
        return 2.0 * np.pi * R_safe
    if mode == "dynamical":
        return R_safe
    raise ValueError(f"Unsupported lambda spectrum mode '{mode}'")


@dataclass
class GalaxyCurve:
    galaxy: str
    radii: np.ndarray
    v_obs: np.ndarray
    err_v: np.ndarray
    g_bar: np.ndarray
    lambda_gw: np.ndarray
    sigma_array: np.ndarray
    lambda_matter: float
    metadata: Dict[str, float]


def load_rotmod_curve(path: Path) -> pd.DataFrame:
    columns = ["R", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        names=columns,
        engine="python",
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df[df["R"] > 0].reset_index(drop=True)


def build_dataset(
    summary_path: Path,
    rotmod_dir: Path,
    lambda_mode: str = "orbital",
) -> Tuple[List[GalaxyCurve], Dict[str, object]]:
    summary_df = pd.read_csv(summary_path)
    dataset: List[GalaxyCurve] = []
    missing = []
    total_rows = 0

    for row in summary_df.itertuples():
        galaxy = row.galaxy_name
        rotmod_path = rotmod_dir / f"{galaxy}_rotmod.dat"
        if not rotmod_path.exists():
            missing.append(galaxy)
            continue

        df = load_rotmod_curve(rotmod_path)
        if df.empty:
            continue

        R = df["R"].to_numpy(dtype=np.float64)
        Vobs = df["Vobs"].to_numpy(dtype=np.float64)
        errV = df["errV"].to_numpy(dtype=np.float64)
        Vgas = df["Vgas"].to_numpy(dtype=np.float64)
        Vdisk = df["Vdisk"].to_numpy(dtype=np.float64)
        Vbul = df["Vbul"].to_numpy(dtype=np.float64)

        v_bar_sq = Vgas**2 + Vdisk**2 + Vbul**2
        R_safe = np.maximum(R, 1e-3)
        g_bar = v_bar_sq / R_safe
        lambda_gw = compute_lambda_values(R, lambda_mode)

        sigma_v = getattr(row, "sigma_velocity", np.nan)
        if not np.isfinite(sigma_v) or sigma_v <= 0:
            sigma_v = 30.0
        sigma_array = np.full_like(R, float(sigma_v), dtype=np.float64)

        R_disk = getattr(row, "R_disk", np.nan)
        if not np.isfinite(R_disk) or R_disk <= 0:
            R_disk = float(np.median(R))
        lambda_matter = 2.0 * np.pi * max(R_disk, 1e-3)

        dataset.append(
            GalaxyCurve(
                galaxy=galaxy,
                radii=R,
                v_obs=Vobs,
                err_v=np.clip(errV, 1.0, None),
                g_bar=g_bar,
                lambda_gw=lambda_gw,
                sigma_array=sigma_array,
                lambda_matter=lambda_matter,
                metadata={
                    "sigma_v": sigma_v,
                    "R_disk": R_disk,
                    "n_points": len(R),
                },
            )
        )
        total_rows += len(R)

    info = {
        "n_galaxies": len(dataset),
        "n_points": total_rows,
        "n_missing": len(missing),
        "missing_galaxies": missing,
    }
    return dataset, info


def evaluate_model(
    model_name: str,
    params: Dict[str, float],
    dataset: List[GalaxyCurve],
) -> Dict[str, object]:
    model_fn = MODEL_REGISTRY[model_name]
    residuals_all = []
    abs_all = []
    chi2_terms = []
    per_gal = []

    for entry in dataset:
        local_params = dict(params)
        if model_name == "metric_resonance":
            ratio = local_params.get("lambda_ratio", 1.0)
            local_params["lambda_m0"] = ratio * entry.lambda_matter
        if model_name == "graviton_pairing" and "ell0" in local_params:
            local_params["xi0"] = local_params["ell0"]

        f = model_fn(
            entry.radii,
            entry.g_bar,
            entry.lambda_gw,
            entry.sigma_array,
            local_params,
            xp=np,
        )
        f = np.asarray(f, dtype=np.float64)
        g_eff = entry.g_bar * f
        v_model = np.sqrt(np.clip(g_eff * entry.radii, 0.0, None))
        residuals = v_model - entry.v_obs

        residuals_all.append(residuals)
        abs_all.append(np.abs(residuals))
        chi2_terms.append(float(np.mean((residuals / entry.err_v) ** 2)))
        per_gal.append(
            {
                "galaxy": entry.galaxy,
                "rms_velocity": float(np.sqrt(np.mean(residuals**2))),
                "median_abs_error": float(np.median(np.abs(residuals))),
                "chi2": chi2_terms[-1],
                "n_points": int(len(residuals)),
            }
        )

    residuals_concat = np.concatenate(residuals_all) if residuals_all else np.array([])
    abs_concat = np.concatenate(abs_all) if abs_all else np.array([])

    return {
        "global_rms": float(np.sqrt(np.mean(residuals_concat**2))) if residuals_concat.size else float("nan"),
        "median_abs_error": float(np.median(abs_concat)) if abs_concat.size else float("nan"),
        "p95_abs_error": float(np.percentile(abs_concat, 95)) if abs_concat.size else float("nan"),
        "mean_bias": float(np.mean(residuals_concat)) if residuals_concat.size else float("nan"),
        "reduced_chi2": float(np.mean(chi2_terms)) if chi2_terms else float("nan"),
        "per_galaxy": per_gal,
    }


def build_params(model_name: str, theta: Iterable[float]) -> Dict[str, float]:
    a_raw, ell_raw, extra_raw = theta
    params: Dict[str, float] = {
        "A": softplus(a_raw),
        "ell0": softplus(ell_raw),
        "p": 0.8,
        "n_coh": 0.5,
    }
    if model_name == "path_interference":
        params["beta_sigma"] = softplus(extra_raw)
        params["sigma_ref"] = 30.0
    elif model_name == "metric_resonance":
        params["lambda_ratio"] = softplus(extra_raw)
        params["log_width"] = 0.6
        params["beta_sigma"] = 0.0
        params["sigma_ref"] = 30.0
    elif model_name == "entanglement":
        params["sigma0"] = softplus(extra_raw) + 5.0
    elif model_name == "vacuum_condensation":
        params["sigma_c"] = softplus(extra_raw) + 10.0
        params["alpha"] = 2.0
        params["beta"] = 1.0
    elif model_name == "graviton_pairing":
        params["sigma0"] = 30.0
        params["gamma_xi"] = softplus(extra_raw)
    else:
        raise ValueError(f"Unknown model '{model_name}'")
    return params


def fit_model(
    model_name: str,
    dataset: List[GalaxyCurve],
    theta0: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, object]]:
    def objective(theta: np.ndarray) -> float:
        params = build_params(model_name, theta)
        metrics = evaluate_model(model_name, params, dataset)
        value = metrics["global_rms"]
        return value if np.isfinite(value) else 1e9

    result = minimize(
        objective,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4},
    )
    best_params = build_params(model_name, result.x)
    metrics = evaluate_model(model_name, best_params, dataset)
    opt_meta = {
        "success": bool(result.success),
        "message": result.message,
        "nfev": result.nfev,
        "nit": result.nit,
        "theta_opt": result.x.tolist(),
    }
    return best_params, metrics, opt_meta


def compute_baseline(dataset: List[GalaxyCurve]) -> Dict[str, object]:
    return evaluate_model(
        "path_interference",
        {"A": 0.0, "ell0": 1.0, "p": 0.8, "n_coh": 0.5, "beta_sigma": 0.0, "sigma_ref": 30.0},
        dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit coherence models on real SPARC data.")
    parser.add_argument(
        "--sparc-summary",
        default="data/sparc/sparc_combined.csv",
        type=Path,
        help="Path to the real SPARC metadata CSV.",
    )
    parser.add_argument(
        "--rotmod-dir",
        default="data/Rotmod_LTG",
        type=Path,
        help="Directory containing the real SPARC rotmod tables.",
    )
    parser.add_argument(
        "--lambda-mode",
        default="orbital",
        choices=["orbital", "dynamical"],
        help="How to convert radius to lambda_gw.",
    )
    parser.add_argument(
        "--output-dir",
        default=CURRENT_DIR / "results",
        type=Path,
        help="Where to store result artifacts.",
    )
    args = parser.parse_args()

    dataset, info = build_dataset(args.sparc_summary, args.rotmod_dir, args.lambda_mode)
    if not dataset:
        raise RuntimeError("No SPARC galaxies loaded; check data paths.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Loaded {info['n_galaxies']} galaxies / {info['n_points']} points (real SPARC data).")
    if info["n_missing"]:
        print(f"[warn] Missing rotmod files for {info['n_missing']} galaxies (skipped).")

    baseline = compute_baseline(dataset)
    print(f"[baseline] GR-only global RMS: {baseline['global_rms']:.2f} km/s")

    model_order = [
        "path_interference",
        "metric_resonance",
        "entanglement",
        "vacuum_condensation",
        "graviton_pairing",
    ]
    initial_guesses = {
        "path_interference": np.array(
            [inv_softplus(0.6), inv_softplus(5.0), inv_softplus(0.4)]
        ),
        "metric_resonance": np.array(
            [inv_softplus(0.6), inv_softplus(5.0), inv_softplus(6.5)]
        ),
        "entanglement": np.array(
            [inv_softplus(0.6), inv_softplus(5.0), inv_softplus(30.0)]
        ),
        "vacuum_condensation": np.array(
            [inv_softplus(0.6), inv_softplus(5.0), inv_softplus(40.0)]
        ),
        "graviton_pairing": np.array(
            [inv_softplus(0.6), inv_softplus(5.0), inv_softplus(0.3)]
        ),
    }

    summary_rows = []
    results_blob = {
        "data": {
            "sparc_summary": str(args.sparc_summary),
            "rotmod_dir": str(args.rotmod_dir),
            "lambda_mode": args.lambda_mode,
            "n_galaxies": info["n_galaxies"],
            "n_points": info["n_points"],
            "missing_galaxies": info["missing_galaxies"],
        },
        "baseline": baseline,
        "models": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for model_name in model_order:
        print(f"\n[fit] {model_name}")
        params, metrics, opt_meta = fit_model(model_name, dataset, initial_guesses[model_name])
        results_blob["models"].append(
            {
                "name": model_name,
                "best_params": params,
                "metrics": metrics,
                "optimizer": opt_meta,
            }
        )
        summary_rows.append(
            {
                "model": model_name,
                "global_rms": metrics["global_rms"],
                "median_abs_error": metrics["median_abs_error"],
                "p95_abs_error": metrics["p95_abs_error"],
                "reduced_chi2": metrics["reduced_chi2"],
                "A": params.get("A"),
                "ell0": params.get("ell0"),
                "extra_param": params.get(
                    "beta_sigma",
                    params.get(
                        "lambda_ratio",
                        params.get(
                            "sigma0",
                            params.get("sigma_c", params.get("gamma_xi")),
                        ),
                    ),
                ),
            }
        )
        print(
            f"  -> RMS {metrics['global_rms']:.2f} km/s | "
            f"median |dv| {metrics['median_abs_error']:.2f} km/s"
        )

    json_path = args.output_dir / "coherence_model_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results_blob, fh, indent=2)

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "coherence_model_summary.csv",
        index=False,
    )
    print(f"\n[done] Results saved to {json_path}")


if __name__ == "__main__":
    main()
