#!/usr/bin/env python3
"""Run the frozen CPR0 spherical cluster-endpoint proxy screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from voidscreen.basin_action import G_SI, KPC_M
from voidscreen.sigma_refracted import refracted_permittivity, sigma_enhancement


ROOT = Path(__file__).resolve().parents[1]


def load_frame(path: Path, *, folds: int, seed: int) -> pd.DataFrame:
    columns = [
        "system",
        "radius_kpc",
        "log_gbar",
        "log_gtot",
        "err_log_gbar",
        "err_log_gtot",
    ]
    frame = pd.read_csv(path, sep=r"\s+", names=columns, comment="#")
    if len(frame) != 84 or frame["system"].nunique() != 20:
        raise ValueError("expected 84 rows in 20 CLASH systems")
    radius_m = frame["radius_kpc"].to_numpy(dtype=float) * KPC_M
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(dtype=float))
    mean_density_kg_m3 = 3.0 * gbar / (4.0 * np.pi * G_SI * radius_m)
    frame["mean_density_g_cm3"] = mean_density_kg_m3 * 1.0e-3
    systems = np.asarray(sorted(frame["system"].unique()), dtype=object)
    permutation = np.random.default_rng(seed).permutation(systems)
    assignment = {
        str(system): int(index % folds) for index, system in enumerate(permutation)
    }
    frame["fold"] = frame["system"].map(assignment).astype(int)
    return frame


def equal_system_mse(frame: pd.DataFrame, predicted_log_g: np.ndarray) -> float:
    residual = predicted_log_g - frame["log_gtot"].to_numpy(dtype=float)
    scores = pd.DataFrame({"system": frame["system"], "squared": residual**2})
    return float(scores.groupby("system", sort=True)["squared"].mean().mean())


def predict_sigma(frame: pd.DataFrame, response_amplitude: float) -> np.ndarray:
    gbar = np.power(10.0, frame["log_gbar"].to_numpy(dtype=float))
    return frame["log_gbar"].to_numpy(dtype=float) + np.log10(
        sigma_enhancement(gbar, response_amplitude)
    )


def predict_constant_epsilon(frame: pd.DataFrame, epsilon: float) -> np.ndarray:
    return frame["log_gbar"].to_numpy(dtype=float) - math.log10(epsilon)


def predict_rg(frame: pd.DataFrame, vector: np.ndarray) -> np.ndarray:
    epsilon_0, log10_rho_c, sharpness = np.asarray(vector, dtype=float)
    epsilon = refracted_permittivity(
        frame["mean_density_g_cm3"].to_numpy(dtype=float),
        minimum_permittivity=float(epsilon_0),
        critical_density=10.0**float(log10_rho_c),
        rg_sharpness=float(sharpness),
    )
    return frame["log_gbar"].to_numpy(dtype=float) - np.log10(epsilon)


def fit_sigma(frame: pd.DataFrame, bounds: tuple[float, float]) -> np.ndarray:
    result = minimize_scalar(
        lambda value: equal_system_mse(frame, predict_sigma(frame, value)),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1.0e-8},
    )
    return np.asarray([result.x], dtype=float)


def fit_constant_epsilon(
    frame: pd.DataFrame, bounds: tuple[float, float]
) -> np.ndarray:
    result = minimize_scalar(
        lambda value: equal_system_mse(
            frame, predict_constant_epsilon(frame, value)
        ),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    return np.asarray([result.x], dtype=float)


def fit_rg(
    frame: pd.DataFrame,
    bounds: list[tuple[float, float]],
    *,
    seed: int,
) -> np.ndarray:
    objective = lambda vector: equal_system_mse(frame, predict_rg(frame, vector))
    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=200,
        popsize=12,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1.0e-9,
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    vector = np.asarray(local_result.x if local_result.success else global_result.x)
    return np.asarray(
        [np.clip(value, low, high) for value, (low, high) in zip(vector, bounds)],
        dtype=float,
    )


def cross_validated_predictions(
    frame: pd.DataFrame,
    *,
    model: str,
    bounds,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    fits: list[dict[str, object]] = []
    for fold in sorted(frame["fold"].unique()):
        training = frame.loc[frame["fold"] != fold]
        heldout = frame.loc[frame["fold"] == fold]
        if model == "sigma_cv":
            vector = fit_sigma(training, tuple(bounds))
            prediction = predict_sigma(heldout, vector[0])
            names = ["B"]
        elif model == "constant_epsilon_cv":
            vector = fit_constant_epsilon(training, tuple(bounds))
            prediction = predict_constant_epsilon(heldout, vector[0])
            names = ["epsilon"]
        elif model == "rg_proxy_cv":
            vector = fit_rg(training, list(map(tuple, bounds)), seed=seed + int(fold))
            prediction = predict_rg(heldout, vector)
            names = ["epsilon_0", "log10_rho_c_g_cm3", "Q"]
        else:
            raise ValueError(f"unknown cross-validation model: {model}")
        predictions[heldout.index.to_numpy()] = prediction
        fits.append(
            {
                "fold": int(fold),
                "training_systems": int(training["system"].nunique()),
                "heldout_systems": sorted(heldout["system"].unique().tolist()),
                "parameters": {
                    name: float(value) for name, value in zip(names, vector)
                },
            }
        )
    if np.any(~np.isfinite(predictions)):
        raise RuntimeError(f"{model} left non-finite held-out predictions")
    return predictions, fits


def metrics(frame: pd.DataFrame, predicted_log_g: np.ndarray) -> dict[str, float]:
    observed = frame["log_gtot"].to_numpy(dtype=float)
    residual = predicted_log_g - observed
    per_system_mse = (
        pd.DataFrame({"system": frame["system"], "squared": residual**2})
        .groupby("system", sort=True)["squared"]
        .mean()
    )
    slope = np.polyfit(
        np.log10(frame["radius_kpc"].to_numpy(dtype=float)), residual, 1
    )[0]
    return {
        "point_RMSE_dex": float(np.sqrt(np.mean(residual**2))),
        "equal_system_RMSE_dex": float(np.sqrt(per_system_mse.mean())),
        "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
        "global_radial_residual_slope_dex_per_dex": float(slope),
        "median_absolute_residual_dex": float(np.median(np.abs(residual))),
    }


def boundary_audit(
    fits: list[dict[str, object]], bounds: list[tuple[float, float]]
) -> dict[str, object]:
    names = ["epsilon_0", "log10_rho_c_g_cm3", "Q"]
    away = 0
    for fit in fits:
        parameters = fit["parameters"]
        at_bound = []
        for name, (low, high) in zip(names, bounds):
            value = float(parameters[name])
            tolerance = 0.01 * (high - low)
            at_bound.append(value <= low + tolerance or value >= high - tolerance)
        fit["parameter_within_1pct_of_bound"] = {
            name: bool(flag) for name, flag in zip(names, at_bound)
        }
        if not any(at_bound):
            away += 1
    rho_values = [float(fit["parameters"][names[1]]) for fit in fits]
    return {
        "folds_with_all_parameters_away_from_bounds": away,
        "log10_rho_c_fold_range_dex": float(max(rho_values) - min(rho_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/cpr0_sigma_refracted_protocol.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/cpr0_cluster_proxy_screen"
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    screen = protocol["cluster_proxy_screen"]
    frame = load_frame(
        ROOT / screen["data"], folds=int(screen["folds"]), seed=int(screen["fold_seed"])
    )
    bounds_config = screen["bounds"]
    rg_bounds = [
        tuple(bounds_config["epsilon_0"]),
        tuple(bounds_config["log10_rho_c_g_cm3"]),
        tuple(bounds_config["Q"]),
    ]

    predictions: dict[str, np.ndarray] = {
        "sigma_fixed_B8.446": predict_sigma(frame, 8.446),
        "rg_published_galaxy_no_refit": predict_rg(
            frame, np.asarray([0.089, -24.25, 0.47])
        ),
    }
    predictions["sigma_universal_B_cv"], sigma_fits = cross_validated_predictions(
        frame,
        model="sigma_cv",
        bounds=bounds_config["Sigma_B"],
        seed=int(screen["fold_seed"]),
    )
    predictions["constant_epsilon_cv"], epsilon_fits = cross_validated_predictions(
        frame,
        model="constant_epsilon_cv",
        bounds=(bounds_config["epsilon_0"][0], 1.0),
        seed=int(screen["fold_seed"]),
    )
    predictions["cpr0_rg_proxy_cv"], rg_fits = cross_validated_predictions(
        frame,
        model="rg_proxy_cv",
        bounds=rg_bounds,
        seed=int(screen["fold_seed"]),
    )

    model_metrics = {name: metrics(frame, values) for name, values in predictions.items()}
    audit = boundary_audit(rg_fits, rg_bounds)
    gates = screen["advance_gates"]
    baseline = model_metrics["sigma_fixed_B8.446"]["equal_system_RMSE_dex"]
    candidate = model_metrics["cpr0_rg_proxy_cv"]
    improvement = 1.0 - candidate["equal_system_RMSE_dex"] / baseline
    gate_results = {
        "equal_system_RMSE_improvement": float(improvement),
        "RMSE_improvement_gate": bool(
            improvement >= gates["equal_system_RMSE_improvement_vs_fixed_Sigma_min"]
        ),
        "median_ratio_gate": bool(
            gates["median_predicted_to_observed_range"][0]
            <= candidate["median_predicted_to_observed"]
            <= gates["median_predicted_to_observed_range"][1]
        ),
        "radial_slope_gate": bool(
            abs(candidate["global_radial_residual_slope_dex_per_dex"])
            <= gates["absolute_global_radial_residual_slope_dex_per_dex_max"]
        ),
        "parameter_boundary_gate": bool(
            audit["folds_with_all_parameters_away_from_bounds"]
            >= gates["folds_with_all_parameters_away_from_bounds_min"]
        ),
        "parameter_stability_gate": bool(
            audit["log10_rho_c_fold_range_dex"]
            <= gates["log10_rho_c_fold_range_max_dex"]
        ),
    }
    gate_results["all_cluster_proxy_gates_pass"] = bool(
        all(value for key, value in gate_results.items() if key.endswith("_gate"))
    )

    output = frame.copy()
    for name, values in predictions.items():
        output[f"predicted_log_g__{name}"] = values
        output[f"residual_dex__{name}"] = values - output["log_gtot"]
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "heldout_points.csv", index=False)
    report = {
        "protocol_version": protocol["protocol_version"],
        "data_summary": {
            "points": int(len(frame)),
            "systems": int(frame["system"].nunique()),
            "density_proxy_range_g_cm3": [
                float(frame["mean_density_g_cm3"].min()),
                float(frame["mean_density_g_cm3"].max()),
            ],
        },
        "model_metrics": model_metrics,
        "fold_fits": {
            "sigma_universal_B_cv": sigma_fits,
            "constant_epsilon_cv": epsilon_fits,
            "cpr0_rg_proxy_cv": rg_fits,
        },
        "rg_parameter_audit": audit,
        "gate_results": gate_results,
        "interpretation_boundary": protocol["claim_boundary"],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
