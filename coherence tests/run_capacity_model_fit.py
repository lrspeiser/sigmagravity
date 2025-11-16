from __future__ import annotations

"""
Fit the coherence microphysics models to the Milky Way capacity profile.

Data source:
    gravitywavebaseline/capacity_profile_fit.json

We treat the published capacity curve as a stand-in for the Sigma-Gravity
enhancement kernel (K). For this first pass we:
  * restrict to shells with non-zero capacity,
  * normalize K by its peak so the target multiplier is 1 + K_norm,
  * approximate lambda_gw with the orbital circumference (2*pi*R),
  * assume a constant velocity dispersion sigma_v = 30 km/s
    (toy proxy; document whenever non-data assumptions are used).

Outputs:
  - coherence tests/results/capacity_fit_results.json
  - coherence tests/results/capacity_fit_summary.md
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from coherence_models import apply_coherence_model


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPACITY_PATH = REPO_ROOT / "gravitywavebaseline" / "capacity_profile_fit.json"
RESULTS_DIR = REPO_ROOT / "coherence tests" / "results"


@dataclass
class FitSpec:
    name: str
    param_order: Sequence[str]
    bounds: Sequence[Sequence[float]]


BASE_PARAMS: Dict[str, Dict[str, float]] = {
    "path_interference": {
        "A": 1.0,
        "ell0": 5.0,
        "p": 1.0,
        "n_coh": 1.0,
        "beta_sigma": 0.4,
        "sigma_ref": 30.0,
    },
    "metric_resonance": {
        "A": 1.0,
        "ell0": 5.0,
        "p": 1.0,
        "n_coh": 1.0,
        "lambda_m0": 5.0,
        "log_width": 0.5,
        "beta_sigma": 0.0,
        "sigma_ref": 30.0,
    },
    "entanglement": {
        "A": 1.0,
        "ell0": 5.0,
        "p": 1.0,
        "n_coh": 1.0,
        "sigma0": 30.0,
    },
    "vacuum_condensation": {
        "A": 1.0,
        "ell0": 5.0,
        "p": 1.0,
        "n_coh": 1.0,
        "sigma_c": 40.0,
        "alpha": 2.0,
        "beta": 1.0,
    },
    "graviton_pairing": {
        "A": 1.0,
        "xi0": 5.0,
        "p": 1.0,
        "n_coh": 1.0,
        "sigma0": 30.0,
        "gamma_xi": 0.0,
    },
}


FIT_SPECS: List[FitSpec] = [
    FitSpec(
        "path_interference",
        ("A", "ell0", "beta_sigma"),
        ((0.0, 5.0), (0.5, 10.0), (0.0, 2.5)),
    ),
    FitSpec(
        "metric_resonance",
        ("A", "ell0", "lambda_m0", "log_width"),
        ((0.0, 5.0), (0.5, 10.0), (0.5, 15.0), (0.1, 2.5)),
    ),
    FitSpec(
        "entanglement",
        ("A", "ell0", "sigma0"),
        ((0.0, 5.0), (0.5, 10.0), (5.0, 120.0)),
    ),
    FitSpec(
        "vacuum_condensation",
        ("A", "ell0", "sigma_c", "alpha"),
        ((0.0, 5.0), (0.5, 10.0), (5.0, 120.0), (0.5, 4.0)),
    ),
    FitSpec(
        "graviton_pairing",
        ("A", "xi0", "sigma0", "gamma_xi"),
        ((0.0, 5.0), (0.5, 10.0), (5.0, 120.0), (0.0, 2.5)),
    ),
]


def load_capacity_profile():
    if not CAPACITY_PATH.exists():
        raise FileNotFoundError(f"Missing {CAPACITY_PATH}")
    with CAPACITY_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    shells = data["shells"]
    R = np.array([row["R_center"] for row in shells], dtype=float)
    capacity = np.array([row["capacity"] for row in shells], dtype=float)
    mask = (capacity > 0.0) & (R > 0.1)
    R = R[mask]
    capacity = capacity[mask]
    if R.size == 0:
        raise RuntimeError("No capacity shells available")
    K_norm = capacity / np.max(capacity)
    target_multiplier = 1.0 + K_norm
    lambda_gw = 2.0 * np.pi * R  # orbital circumference proxy
    sigma_v = np.full_like(R, 30.0)  # toy dispersion assumption
    summary = {
        "data_points": int(R.size),
        "R_min": float(R.min()),
        "R_max": float(R.max()),
        "capacity_min": float(capacity.min()),
        "capacity_max": float(capacity.max()),
    }
    return R, target_multiplier, lambda_gw, sigma_v, summary


def fit_model(
    spec: FitSpec,
    R: np.ndarray,
    target_multiplier: np.ndarray,
    lambda_gw: np.ndarray,
    sigma_v: np.ndarray,
):
    base_params = BASE_PARAMS[spec.name].copy()

    def objective(theta):
        params = base_params.copy()
        for key, val in zip(spec.param_order, theta):
            params[key] = float(val)
        pred = apply_coherence_model(
            spec.name, R, np.ones_like(R), lambda_gw, sigma_v, params
        )
        return np.sqrt(np.mean((pred - target_multiplier) ** 2))

    result = differential_evolution(
        objective,
        bounds=spec.bounds,
        seed=3,
        maxiter=80,
        popsize=12,
        tol=1e-3,
        polish=True,
    )

    best_params = base_params.copy()
    for key, val in zip(spec.param_order, result.x):
        best_params[key] = float(val)

    pred = apply_coherence_model(
        spec.name, R, np.ones_like(R), lambda_gw, sigma_v, best_params
    )
    residual = pred - target_multiplier
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_err = float(np.max(np.abs(residual)))

    return {
        "model": spec.name,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_err,
        "best_params": best_params,
        "convergence": {
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
        },
    }


def write_outputs(results: List[Dict], summary: Dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_source": str(CAPACITY_PATH.relative_to(REPO_ROOT)),
        "normalization": "capacity normalized by max value (dimensionless K)",
        "lambda_definition": "lambda_gw = 2*pi*R (orbital circumference)",
        "sigma_v_assumption": "constant 30 km/s toy dispersion",
        "results": results,
        "capacity_stats": summary,
    }
    json_path = RESULTS_DIR / "capacity_fit_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines = [
        "# Capacity Profile -> Coherence Model Fits",
        "",
        f"- Data: `{payload['data_source']}` (real Milky Way capacity profile)",
        "- Normalization: capacity / max -> K_norm, multiplier = 1 + K_norm",
        "- lambda_gw proxy: orbital circumference (2*pi*R)",
        "- sigma_v: constant 30 km/s (toy assumption, document explicitly)",
        "",
        "## Metrics",
        "| Model | RMSE | MAE | Max | Key Params |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(results, key=lambda item: item["rmse"]):
        varied = [
            f"{key}={row['best_params'][key]:.3g}"
            for key in next(
                spec.param_order for spec in FIT_SPECS if spec.name == row["model"]
            )
        ]
        lines.append(
            f"| `{row['model']}` | {row['rmse']:.4f} | {row['mae']:.4f} | {row['max_abs_error']:.4f} | "
            + ", ".join(varied)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Only shells with non-zero capacity were used.",
            "- Fits target the shape of K(R); absolute scaling must be revisited once g_bar and lambda_gw are coupled to real galaxies.",
            "- Next iteration should replace the toy sigma_v assumption with measured dispersion data per radius.",
        ]
    )

    md_path = RESULTS_DIR / "capacity_fit_summary.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    R, target, lambda_gw, sigma_v, stats = load_capacity_profile()
    results = []
    for spec in FIT_SPECS:
        print(f"[fit] {spec.name}")
        fit = fit_model(spec, R, target, lambda_gw, sigma_v)
        results.append(fit)
        param_line = ", ".join(
            f"{key}: {fit['best_params'][key]:.3g}" for key in spec.param_order
        )
        print(
            f"    rmse={fit['rmse']:.4f}, mae={fit['mae']:.4f}, params={{ {param_line} }}"
        )
    write_outputs(results, stats)
    print(f"[done] wrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
