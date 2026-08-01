#!/usr/bin/env python3
"""Lock RAR-quality galaxy matter laws before any metric-slip lensing fit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sparc_independent_nuisance_refit import metrics as sparc_metrics  # noqa: E402
from run_vector_completion_full_test import at_boundary, json_safe  # noqa: E402

from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.phenomenology import fixed_rar_enhancement  # noqa: E402
from voidscreen.unbounded_running import (  # noqa: E402
    RUNNING_MODELS,
    predict_running_acceleration,
    solar_system_diagnostics,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_folds(frame: pd.DataFrame, folds: int, seed: int) -> pd.Series:
    names = np.asarray(sorted(frame["galaxy"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    mapping = {str(name): int(index % folds) for index, name in enumerate(names)}
    return frame["galaxy"].map(mapping).astype(int)


def predict_velocity(frame: pd.DataFrame, model: str, parameters) -> np.ndarray:
    result = predict_running_acceleration(
        frame["g_bar_m_s2"].to_numpy(float),
        frame["radius_adjusted_kpc"].to_numpy(float),
        model,
        parameters,
    )
    return np.sqrt(
        result["predicted_acceleration_m_s2"]
        * frame["radius_adjusted_kpc"].to_numpy(float)
        * KPC_M
        / 1.0e6
    )


def equal_galaxy_rar_loss(frame: pd.DataFrame, predicted: np.ndarray) -> float:
    residual = np.log10(predicted / frame["velocity_RAR_same_nuisance_km_s"].to_numpy(float))
    table = frame[["galaxy"]].copy()
    table["residual_squared"] = np.square(residual)
    return float(table.groupby("galaxy")["residual_squared"].mean().mean())


def solar_penalty(model: str, parameters, protocol: dict) -> tuple[float, dict]:
    gates = protocol["solar_gates"]
    diagnostic = solar_system_diagnostics(
        model,
        parameters,
        cassini_limit=float(gates["maximum_fractional_coupling_change_limb_to_Saturn"]),
    )
    cassini_limit = float(gates["maximum_fractional_coupling_change_limb_to_Saturn"])
    earth_limit = float(gates["maximum_Earth_orbit_fractional_change"])
    excess = max(
        0.0,
        diagnostic["maximum_fractional_change_limb_to_Saturn"] / cassini_limit - 1.0,
    )
    earth = max(
        0.0,
        abs(diagnostic["Earth_orbit_fractional_change"]) / earth_limit - 1.0,
    )
    return excess**2 + earth**2, diagnostic


def fit_model(
    frame: pd.DataFrame,
    model: str,
    specification: dict,
    protocol: dict,
    seed: int,
) -> np.ndarray:
    bounds = list(map(tuple, specification["bounds"]))
    coefficient = float(protocol["optimization"]["Cassini_violation_penalty"])

    def objective(values) -> float:
        try:
            penalty, _ = solar_penalty(model, values, protocol)
            return equal_galaxy_rar_loss(frame, predict_velocity(frame, model, values)) + (
                coefficient * penalty
            )
        except (FloatingPointError, OverflowError, ValueError):
            return 1.0e100

    settings = protocol["optimization"]
    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=int(settings["differential_evolution_maxiter"]),
        popsize=int(settings["differential_evolution_popsize"]),
        tol=1.0e-10,
        polish=False,
        workers=1,
    )
    local = minimize(
        objective,
        global_fit.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 8000, "ftol": 1.0e-15, "gtol": 1.0e-10},
    )
    return np.asarray(local.x if local.success else global_fit.x, dtype=float)


def add_prediction_columns(frame: pd.DataFrame, model: str, velocity: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "matter_model", model)
    output["velocity_predicted_km_s"] = velocity
    inclination_factor = (
        output["velocity_observed_adjusted_km_s"].to_numpy(float)
        / output["velocity_observed_catalog_kms"].to_numpy(float)
    )
    output["velocity_predicted_catalog_km_s"] = velocity / inclination_factor
    output["coherence"] = np.nan
    return output


def bcg_metrics(model: str, parameters, sample: pd.DataFrame) -> dict:
    selected = sample[sample["domain"].eq("BCG")].copy()
    gbar = np.power(10.0, selected["log_gbar"].to_numpy(float))
    if model == "fixed_RAR":
        predicted = gbar * fixed_rar_enhancement(gbar, 1.2e-10)
    else:
        predicted = predict_running_acceleration(
            gbar,
            selected["radius_kpc"].to_numpy(float),
            model,
            parameters,
        )["predicted_acceleration_m_s2"]
    residual = np.log10(predicted) - selected["log_gobs"].to_numpy(float)
    return {
        "points": len(selected),
        "RMSE_dex": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_residual_dex": float(np.mean(residual)),
        "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
    }


def make_figure(report: dict, output: Path) -> None:
    names = list(report["models"])
    labels = [name.replace("curvature_", "c:") for name in names]
    outer = [report["models"][name]["SPARC"]["outer_holdout"]["RMSE_km_s"] for name in names]
    bcg = [report["models"][name]["BCG_dynamics"]["RMSE_dex"] for name in names]
    rar = report["references"]["fixed_RAR_outer_RMSE_km_s"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(labels, outer)
    axes[0].axhline(rar, color="black", linestyle="--", label="fixed RAR")
    axes[0].axhline(1.1 * rar, color="gray", linestyle=":", label="advance gate")
    axes[0].set(title="Untouched SPARC outer radii", ylabel="RMSE (km/s)")
    axes[0].legend()
    axes[1].bar(labels, bcg)
    axes[1].set(title="Unchanged matter law on cluster-central BCGs", ylabel="RMSE (dex)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    config_path = ROOT / "configs/metric_slip_galaxy_matter_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_galaxy_matter_scores":
        raise RuntimeError("galaxy matter protocol was not frozen before scoring")
    unknown = set(protocol["models"]) - RUNNING_MODELS
    if unknown:
        raise RuntimeError(f"unknown running models: {sorted(unknown)}")

    source_path = ROOT / protocol["inputs"]["SPARC_points"]
    source = pd.read_csv(source_path)
    frame = source[
        source["model"].eq("fixed_RAR") & source["scenario"].eq("invariant")
    ].copy().reset_index(drop=True)
    frame["fold"] = assign_folds(
        frame,
        int(protocol["sample"]["galaxy_folds"]),
        int(protocol["sample"]["fold_seed"]),
    )
    reference_report = json.loads((ROOT / protocol["inputs"]["SPARC_report"]).read_text())
    rar_reference = reference_report["scores"]["fixed_RAR:invariant"]
    bcg_sample = pd.read_csv(ROOT / protocol["inputs"]["BCG_cluster_sample"])

    results = {
        "fixed_RAR": {
            "full_fit_parameters": {},
            "solar": {"Cassini_pass": True, "Earth_orbit_fractional_change": 0.0},
            "SPARC": rar_reference,
            "BCG_dynamics": bcg_metrics("fixed_RAR", None, bcg_sample),
            "advance": True,
        }
    }
    predictions = [add_prediction_columns(frame, "fixed_RAR", frame["velocity_RAR_same_nuisance_km_s"].to_numpy(float))]
    fold_predictions = []
    tolerance = float(protocol["optimization"]["boundary_fraction_tolerance"])

    for model, specification in protocol["models"].items():
        print(f"galaxy matter model={model}", flush=True)
        heldout_velocity = np.full(len(frame), np.nan)
        fold_fits = []
        for fold in range(int(protocol["sample"]["galaxy_folds"])):
            training = frame[frame["fold"].ne(fold)]
            heldout = frame[frame["fold"].eq(fold)]
            parameters = fit_model(
                training[training["split"].eq("inner_train")],
                model,
                specification,
                protocol,
                int(protocol["sample"]["fold_seed"]) + fold,
            )
            velocity = predict_velocity(heldout, model, parameters)
            heldout_velocity[heldout.index.to_numpy()] = velocity
            table = add_prediction_columns(heldout, model, velocity)
            table.insert(1, "fit_fold", fold)
            fold_predictions.append(table)
            fold_fits.append(
                dict(zip(specification["parameters"], map(float, parameters), strict=True))
            )
        if np.any(~np.isfinite(heldout_velocity)):
            raise RuntimeError(f"{model} left missing galaxy-fold predictions")

        full = fit_model(
            frame[frame["split"].eq("inner_train")],
            model,
            specification,
            protocol,
            int(protocol["sample"]["fold_seed"]) + 100,
        )
        full_velocity = predict_velocity(frame, model, full)
        points = add_prediction_columns(frame, model, full_velocity)
        predictions.append(points)
        _, solar = solar_penalty(model, full, protocol)
        boundary = dict(
            zip(
                specification["parameters"],
                at_boundary(full, specification["bounds"], tolerance),
                strict=True,
            )
        )
        inner_metrics = sparc_metrics(points, "inner_train")
        outer_metrics = sparc_metrics(points, "outer_holdout")
        gates = protocol["advance_gates"]
        advance = (
            outer_metrics["RMSE_km_s"]
            <= float(gates["outer_RMSE_relative_to_fixed_RAR_max"])
            * rar_reference["outer_holdout"]["RMSE_km_s"]
            and outer_metrics["equal_galaxy_RMSE_km_s"]
            <= float(gates["outer_equal_galaxy_RMSE_relative_to_fixed_RAR_max"])
            * rar_reference["outer_holdout"]["equal_galaxy_RMSE_km_s"]
            and solar["Cassini_pass"]
            and abs(solar["Earth_orbit_fractional_change"])
            <= float(protocol["solar_gates"]["maximum_Earth_orbit_fractional_change"])
            and not any(boundary.values())
        )
        results[model] = {
            "full_fit_parameters": dict(
                zip(specification["parameters"], map(float, full), strict=True)
            ),
            "full_fit_parameter_vector": list(map(float, full)),
            "full_fit_at_boundary": boundary,
            "fold_fits": fold_fits,
            "galaxy_fold_RAR_log_velocity_RMSE_dex": float(
                np.sqrt(equal_galaxy_rar_loss(frame, heldout_velocity))
            ),
            "SPARC": {"inner_train": inner_metrics, "outer_holdout": outer_metrics},
            "BCG_dynamics": bcg_metrics(model, full, bcg_sample),
            "solar": solar,
            "advance": bool(advance),
        }

    advanced = [name for name, result in results.items() if result["advance"]]
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed galaxy-first matter-law lock",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hash": sha256(source_path),
        "models": results,
        "advanced_to_slip": advanced,
        "references": {
            "fixed_RAR_outer_RMSE_km_s": rar_reference["outer_holdout"]["RMSE_km_s"],
            "fixed_RAR_outer_equal_galaxy_RMSE_km_s": rar_reference["outer_holdout"]["equal_galaxy_RMSE_km_s"],
        },
        "claim_boundary": [
            "The fixed-RAR nuisance parameters were inferred from each galaxy's inner radii and then held fixed.",
            "The curvature laws were calibrated to the fixed-RAR inner response, not directly to lensing or observed outer velocities.",
            "The outer score is a galaxy radial-transfer check; selection among passing laws spends it for model choice.",
            "BCG dynamics are a cluster-environment matter-tracer check but not same-object gas/lensing closure."
        ],
    }
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["predictions"], index=False
    )
    pd.concat(fold_predictions, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["fold_predictions"], index=False
    )
    make_figure(report, ROOT / protocol["outputs"]["figure"])
    lines = [
        "# Galaxy-first matter-law lock",
        "",
        "| model | outer RMSE (km/s) | BCG dynamics (dex) | advance |",
        "|---|---:|---:|---|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['SPARC']['outer_holdout']['RMSE_km_s']:.3f} | "
            f"{result['BCG_dynamics']['RMSE_dex']:.4f} | {result['advance']} |"
        )
    lines.extend(["", f"Advanced: **{', '.join(advanced)}**."])
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
