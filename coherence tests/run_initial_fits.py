#!/usr/bin/env python
"""
Fit the five Î£-Gravity coherence mechanisms to real SPARC rotation curves.

Data inputs (read-only):
  - gravitywavebaseline/sparc_results/*_capacity_test.json  (per-galaxy RCs)
  - data/sparc/sparc_combined.csv                           (global Ïƒ_v catalog)

Assumptions:
  - g_bar âˆ V_GR^2 and g_eff âˆ V_obs^2, so the empirical multiplier is
        f_obs = (V_obs / V_GR)^2
  - Î»_gw is not provided in the capacity test exports, so we approximate it
        with the orbital circumference: Î»_gw = 2Ï€R  (purely geometric proxy)
  - Ïƒ_v is taken from the SPARC combined table and treated as constant across
        each galaxy's radius.

Outputs:
  - coherence tests/results/initial_fit_summary.json / .md
  - results_log.md files inside each mechanism subfolder
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from coherence_models import MODEL_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
COHERENCE_ROOT = Path(__file__).resolve().parent
SPARC_TABLE = REPO_ROOT / "data" / "sparc" / "sparc_combined.csv"
SPARC_RESULTS_DIR = REPO_ROOT / "gravitywavebaseline" / "sparc_results"
RESULTS_DIR = COHERENCE_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GALAXIES = [
    "DDO154",    # cold dwarf (low Ïƒ_v)
    "NGC2403",   # intermediate spiral
    "NGC5055",   # high-mass spiral
    "NGC7331",   # massive spiral
    "UGC02953",  # LSB giant
]

MODEL_PARAM_SPACES: Dict[str, List[Tuple[str, float, float]]] = {
    "path_interference": [
        ("A", 0.05, 3.0),
        ("ell0", 0.5, 20.0),
        ("p", 0.3, 3.0),
        ("n_coh", 0.3, 5.0),
        ("beta_sigma", 0.0, 1.5),
    ],
    "metric_resonance": [
        ("A", 0.05, 3.0),
        ("ell0", 0.5, 20.0),
        ("p", 0.3, 3.0),
        ("n_coh", 0.3, 5.0),
        ("lambda_m0", 0.5, 80.0),
        ("log_width", 0.1, 2.0),
        ("beta_sigma", 0.0, 1.5),
    ],
    "entanglement": [
        ("A", 0.05, 3.0),
        ("ell0", 0.5, 20.0),
        ("p", 0.3, 3.0),
        ("n_coh", 0.3, 5.0),
        ("sigma0", 5.0, 120.0),
    ],
    "vacuum_condensation": [
        ("A", 0.05, 3.0),
        ("ell0", 0.5, 20.0),
        ("p", 0.3, 3.0),
        ("n_coh", 0.3, 5.0),
        ("sigma_c", 5.0, 120.0),
        ("alpha", 0.5, 4.0),
        ("beta", 0.5, 4.0),
    ],
    "graviton_pairing": [
        ("A", 0.05, 3.0),
        ("xi0", 0.5, 20.0),
        ("p", 0.3, 3.0),
        ("n_coh", 0.3, 5.0),
        ("sigma0", 5.0, 120.0),
        ("gamma_xi", 0.0, 1.5),
    ],
}

N_RANDOM_SAMPLES = 600
RNG = np.random.default_rng(20251116)


def load_sigma_map(csv_path: Path) -> Dict[str, float]:
    sigma_map: Dict[str, float] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["galaxy_name"].strip()
            try:
                sigma_val = float(row["sigma_velocity"])
            except (ValueError, TypeError):
                continue
            sigma_map[name] = sigma_val
    return sigma_map


def load_rotation_curve(galaxy: str) -> Dict[str, np.ndarray]:
    path = SPARC_RESULTS_DIR / f"{galaxy}_capacity_test.json"
    if not path.exists():
        raise FileNotFoundError(f"Rotation curve export not found for {galaxy}: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload["data"]
    radii: List[float] = []
    f_obs: List[float] = []
    for row in rows:
        R = row.get("R")
        v_obs = row.get("V_obs")
        v_gr = row.get("V_GR")
        if R is None or v_obs is None or v_gr is None:
            continue
        if R <= 0 or v_obs <= 0 or v_gr <= 0:
            continue
        ratio = (v_obs / v_gr) ** 2
        if not math.isfinite(ratio):
            continue
        # Trim extreme outliers that usually correspond to noisy inner points
        if ratio < 0.05 or ratio > 25.0:
            continue
        radii.append(R)
        f_obs.append(ratio)
    if not radii:
        raise ValueError(f"No clean data rows found for {galaxy}")
    radii_arr = np.array(radii, dtype=float)
    f_obs_arr = np.array(f_obs, dtype=float)
    g_bar_arr = np.ones_like(radii_arr)  # not used directly but kept for API parity
    lambda_arr = 2.0 * math.pi * radii_arr
    return {
        "R": radii_arr,
        "lambda_gw": lambda_arr,
        "g_bar": g_bar_arr,
        "f_obs": f_obs_arr,
    }


def random_params(param_defs: Iterable[Tuple[str, float, float]]) -> Dict[str, float]:
    return {
        name: float(RNG.uniform(low, high))
        for (name, low, high) in param_defs
    }


def evaluate_model(
    model_name: str,
    data: Dict[str, np.ndarray],
    sigma_v: float,
) -> Dict[str, float]:
    param_defs = MODEL_PARAM_SPACES[model_name]
    model_func = MODEL_REGISTRY[model_name]
    R = data["R"]
    lambda_gw = data["lambda_gw"]
    g_bar = data["g_bar"]
    f_obs = data["f_obs"]
    log_f_obs = np.log10(f_obs)

    best: Dict[str, float] | None = None
    for _ in range(N_RANDOM_SAMPLES):
        params = random_params(param_defs)
        try:
            f_pred = model_func(R, g_bar, lambda_gw, sigma_v, params, xp=np)
        except Exception:
            continue
        if not np.all(np.isfinite(f_pred)):
            continue
        if np.any(f_pred <= 0):
            continue
        log_pred = np.log10(f_pred)
        rmse_log = float(np.sqrt(np.mean((log_pred - log_f_obs) ** 2)))
        mae_lin = float(np.mean(np.abs(f_pred - f_obs)))
        max_abs = float(np.max(np.abs(f_pred - f_obs)))
        if np.allclose(f_pred, f_pred[0]) or np.allclose(f_obs, f_obs[0]):
            corr = float('nan')
        else:
            corr = float(np.corrcoef(f_pred, f_obs)[0, 1])
        score = rmse_log
        if best is None or score < best["rmse_log"]:
            best = {
                "params": params,
                "rmse_log": rmse_log,
                "mae_lin": mae_lin,
                "max_abs": max_abs,
                "corr": corr,
            }
    if best is None:
        raise RuntimeError(f"Failed to fit model {model_name} for sigma_v={sigma_v}")
    return best


def params_to_str(params: Dict[str, float]) -> str:
    entries = [f"{k}={params[k]:.3g}" for k in sorted(params)]
    return ", ".join(entries)


def write_model_logs(results: Dict[str, dict]) -> None:
    timestamp = results["metadata"]["timestamp_utc"]
    for model_name in MODEL_PARAM_SPACES:
        rows = []
        for galaxy, g_entry in results["galaxies"].items():
            m_entry = g_entry["models"][model_name]
            rows.append(
                (
                    galaxy,
                    g_entry["sigma_v"],
                    g_entry["n_points"],
                    m_entry["rmse_log"],
                    m_entry["mae_lin"],
                    m_entry["max_abs"],
                    params_to_str(m_entry["params"]),
                )
            )
        lines = [
            f"{model_name.replace('_', ' ').title()} Results Log",
            "========================================",
            "",
            f"- Data: Real SPARC rotation curves (`gravitywavebaseline/sparc_results/*_capacity_test.json`).",
            f"- Ïƒ_v source: `data/sparc/sparc_combined.csv`.",
            f"- Î»_gw proxy: `2Ï€R` (geometric assumption; no file edits).",
            f"- Run timestamp (UTC): {timestamp}.",
            "",
            "| Galaxy | Ïƒ_v (km/s) | Points | RMSE log10(f) | MAE(f) | Max|f_model - f_obs| | Best-fit params |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for galaxy, sigma_v, n_points, rmse_log, mae_lin, max_abs, param_str in rows:
            lines.append(
                f"| {galaxy} | {sigma_v:.2f} | {n_points} | {rmse_log:.4f} | {mae_lin:.4f} | {max_abs:.4f} | {param_str} |"
            )
        log_path = COHERENCE_ROOT / model_name / "results_log.md"
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sigma_map = load_sigma_map(SPARC_TABLE)
    summary: Dict[str, dict] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_random_samples_per_model": N_RANDOM_SAMPLES,
            "galaxies": GALAXIES,
            "data_sources": {
                "rotation_curves": str(SPARC_RESULTS_DIR),
                "sigma_table": str(SPARC_TABLE),
            },
            "lambda_proxy": "lambda_gw = 2Ï€R (derived from observed radii)",
            "data_type": "real SPARC rotation curves; no root data modified",
        },
        "galaxies": {},
    }

    for galaxy in GALAXIES:
        sigma_v = sigma_map.get(galaxy)
        if sigma_v is None:
            raise KeyError(f"No sigma_v entry for {galaxy} in {SPARC_TABLE}")
        data = load_rotation_curve(galaxy)
        models_summary = {}
        for model_name in MODEL_PARAM_SPACES:
            best = evaluate_model(model_name, data, sigma_v)
            models_summary[model_name] = best
        summary["galaxies"][galaxy] = {
            "sigma_v": float(sigma_v),
            "n_points": int(len(data["R"])),
            "models": models_summary,
        }

    json_path = RESULTS_DIR / "initial_fit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown overview
    lines = [
        "# Initial Coherence Fits",
        "",
        "- Data: Real SPARC rotation curves (`gravitywavebaseline/sparc_results/*_capacity_test.json`).",
        "- Ïƒ_v: from `data/sparc/sparc_combined.csv` (no edits).",
        "- Î»_gw proxy: `2Ï€R` (geometric circumference; needed because Î»_gw not stored in exports).",
        f"- Random search: {N_RANDOM_SAMPLES} draws per mechanism/galaxy.",
        "",
        "| Galaxy | Model | Ïƒ_v (km/s) | Points | RMSE log10(f) | MAE(f) | Max|Î”f| | Best-fit params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for galaxy, g_entry in summary["galaxies"].items():
        sigma_v = g_entry["sigma_v"]
        n_points = g_entry["n_points"]
        for model_name, m_entry in g_entry["models"].items():
            lines.append(
                f"| {galaxy} | {model_name} | {sigma_v:.2f} | {n_points} | "
                f"{m_entry['rmse_log']:.4f} | {m_entry['mae_lin']:.4f} | {m_entry['max_abs']:.4f} | "
                f"{params_to_str(m_entry['params'])} |"
            )
    md_path = RESULTS_DIR / "initial_fit_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_model_logs(summary)
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
