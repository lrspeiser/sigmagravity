#!/usr/bin/env python3
"""Test a source-distribution gate on held-out SPARC and CLASH profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.source_gated_metric import (  # noqa: E402
    gated_extra_acceleration,
    radial_source_concentration,
    source_gate,
    source_gated_metric_eta,
)

KPC_M = 3.085677581491367e19
G_SI = 6.67430e-11


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_rar_acceleration(gbar: np.ndarray, a_dagger: float) -> np.ndarray:
    return gbar / -np.expm1(-np.sqrt(gbar / a_dagger))


def add_profile_statistics(
    frame: pd.DataFrame,
    *,
    system_column: str,
    radius_column: str,
    gbar_column: str,
    a_dagger: float,
    maximum_mass_slope: float,
    diffuseness_power: float,
    concentration_power: float,
    density_column: str | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    output["source_concentration"] = np.nan
    for _, block in output.groupby(system_column, sort=True):
        ordered = block.sort_values(radius_column)
        if density_column is None:
            concentration = radial_source_concentration(
                ordered[radius_column].to_numpy(float),
                ordered[gbar_column].to_numpy(float),
                maximum_mass_slope=maximum_mass_slope,
            )
        else:
            radius_m = ordered[radius_column].to_numpy(float) * KPC_M
            gbar = ordered[gbar_column].to_numpy(float)
            density_kg_m3 = ordered[density_column].to_numpy(float) * 1000.0
            enclosed_mass_kg = gbar * np.square(radius_m) / G_SI
            shell_mass_kg = 4.0 * np.pi * np.power(radius_m, 3) * density_kg_m3
            mass_slope = np.clip(
                shell_mass_kg / enclosed_mass_kg,
                0.0,
                maximum_mass_slope,
            )
            concentration = 1.0 / (1.0 + mass_slope)
        output.loc[ordered.index, "source_concentration"] = concentration
    output["source_gate"] = source_gate(
        output[gbar_column].to_numpy(float),
        output["source_concentration"].to_numpy(float),
        acceleration_scale_m_s2=a_dagger,
        diffuseness_power=diffuseness_power,
        concentration_power=concentration_power,
    )
    return output


def balanced_folds(systems, folds: int, seed: int) -> dict[str, int]:
    names = np.asarray(sorted(set(map(str, systems))), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    return {str(name): int(index % folds) for index, name in enumerate(names)}


def load_galaxies(path: Path, constants: dict, split: dict) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = raw[
        (raw["model"] == "fixed_RAR") & (raw["scenario"] == "invariant")
    ].copy()
    expected = split["galaxy_inner_training_points"] + split["galaxy_outer_test_points"]
    if len(frame) != expected or frame["galaxy"].nunique() != split["galaxies"]:
        raise RuntimeError("SPARC fixed-RAR sample does not match the frozen protocol")
    frame["domain"] = "galaxy"
    frame["system"] = frame["galaxy"].astype(str)
    frame["fold"] = frame["system"].map(
        balanced_folds(
            frame["system"],
            int(split["folds"]),
            int(split["fold_seed"]),
        )
    )
    frame = add_profile_statistics(
        frame,
        system_column="system",
        radius_column="radius_adjusted_kpc",
        gbar_column="g_bar_m_s2",
        a_dagger=float(constants["a_dagger_m_s2"]),
        maximum_mass_slope=float(constants["maximum_enclosed_mass_slope"]),
        diffuseness_power=float(constants["diffuseness_power"]),
        concentration_power=float(constants["concentration_power"]),
    )
    radius_m = frame["radius_adjusted_kpc"].to_numpy(float) * KPC_M
    velocity = frame["velocity_predicted_km_s"].to_numpy(float) * 1000.0
    frame["g_RAR_m_s2"] = np.square(velocity) / radius_m
    if np.any(frame["g_RAR_m_s2"] < frame["g_bar_m_s2"] * (1.0 - 1.0e-8)):
        raise RuntimeError("inferred SPARC RAR acceleration fell below gbar")
    return frame


def load_clusters(path: Path, constants: dict, split: dict) -> pd.DataFrame:
    raw = pd.read_csv(path)
    columns = [
        "system",
        "radius_kpc",
        "fold",
        "log_gbar",
        "log_gobs",
        "err_log_gbar",
        "err_log_gobs",
        "local_density_g_cm3",
    ]
    frame = raw[columns].copy()
    if len(frame) != split["cluster_radial_points"]:
        raise RuntimeError("CLASH point count does not match the frozen protocol")
    if frame["system"].nunique() != split["clusters"]:
        raise RuntimeError("CLASH system count does not match the frozen protocol")
    frame["domain"] = "cluster"
    frame["g_bar_m_s2"] = np.power(10.0, frame["log_gbar"].to_numpy(float))
    frame["g_obs_m_s2"] = np.power(10.0, frame["log_gobs"].to_numpy(float))
    frame["g_RAR_m_s2"] = fixed_rar_acceleration(
        frame["g_bar_m_s2"].to_numpy(float),
        float(constants["a_dagger_m_s2"]),
    )
    frame["sigma_dex"] = np.hypot(
        frame["err_log_gbar"].to_numpy(float),
        frame["err_log_gobs"].to_numpy(float),
    )
    return add_profile_statistics(
        frame,
        system_column="system",
        radius_column="radius_kpc",
        gbar_column="g_bar_m_s2",
        a_dagger=float(constants["a_dagger_m_s2"]),
        maximum_mass_slope=float(constants["maximum_enclosed_mass_slope"]),
        diffuseness_power=float(constants["diffuseness_power"]),
        concentration_power=float(constants["concentration_power"]),
        density_column="local_density_g_cm3",
    )


def predict_cluster(frame: pd.DataFrame, kappa: float, *, gated: bool) -> np.ndarray:
    gate = frame["source_gate"].to_numpy(float) if gated else np.ones(len(frame))
    acceleration = gated_extra_acceleration(
        frame["g_bar_m_s2"].to_numpy(float),
        frame["g_RAR_m_s2"].to_numpy(float),
        gate,
        kappa,
    )
    return np.log10(acceleration)


def predict_galaxy_velocity(frame: pd.DataFrame, kappa: float) -> np.ndarray:
    acceleration = gated_extra_acceleration(
        frame["g_bar_m_s2"].to_numpy(float),
        frame["g_RAR_m_s2"].to_numpy(float),
        frame["source_gate"].to_numpy(float),
        kappa,
    )
    return frame["velocity_predicted_km_s"].to_numpy(float) * np.sqrt(
        acceleration / frame["g_RAR_m_s2"].to_numpy(float)
    )


def equal_system_mse(system, residual) -> float:
    table = pd.DataFrame({"system": np.asarray(system), "squared": np.square(residual)})
    return float(table.groupby("system")["squared"].mean().mean())


def cluster_normalized_mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    residual = (prediction - frame["log_gobs"].to_numpy(float)) / frame[
        "sigma_dex"
    ].to_numpy(float)
    return equal_system_mse(frame["system"], residual)


def galaxy_normalized_mse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    residual = (
        prediction - frame["velocity_observed_adjusted_km_s"].to_numpy(float)
    ) / frame["velocity_error_total_km_s"].to_numpy(float)
    return equal_system_mse(frame["system"], residual)


def bounded_fit(function, bounds: list[float]) -> tuple[float, float]:
    low, high = map(float, bounds)
    result = minimize_scalar(
        function,
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1.0e-8, "maxiter": 1000},
    )
    candidates = [(low, function(low)), (high, function(high))]
    if result.success and math.isfinite(float(result.fun)):
        candidates.append((float(result.x), float(result.fun)))
    return min(candidates, key=lambda item: (item[1], item[0]))


def fit_cluster_kappa(frame: pd.DataFrame, bounds: list[float], *, gated: bool):
    return bounded_fit(
        lambda kappa: cluster_normalized_mse(
            frame, predict_cluster(frame, kappa, gated=gated)
        ),
        bounds,
    )


def fit_joint_kappa(
    galaxy: pd.DataFrame,
    cluster: pd.DataFrame,
    bounds: list[float],
) -> tuple[float, float]:
    galaxy_null = galaxy_normalized_mse(galaxy, predict_galaxy_velocity(galaxy, 0.0))
    cluster_null = cluster_normalized_mse(
        cluster, predict_cluster(cluster, 0.0, gated=True)
    )

    def objective(kappa: float) -> float:
        galaxy_score = galaxy_normalized_mse(
            galaxy, predict_galaxy_velocity(galaxy, kappa)
        )
        cluster_score = cluster_normalized_mse(
            cluster, predict_cluster(cluster, kappa, gated=True)
        )
        return 0.5 * (galaxy_score / galaxy_null + cluster_score / cluster_null)

    return bounded_fit(objective, bounds)


def cluster_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    residual = prediction - frame["log_gobs"].to_numpy(float)
    per_system = (
        pd.DataFrame({"system": frame["system"], "squared": np.square(residual)})
        .groupby("system")["squared"]
        .mean()
    )
    normalized = residual / frame["sigma_dex"].to_numpy(float)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return {
        "systems": int(frame["system"].nunique()),
        "points": int(len(frame)),
        "equal_system_RMSE_dex": float(np.sqrt(per_system.mean())),
        "point_RMSE_dex": rmse,
        "RMSE_factor": float(10.0**rmse),
        "mean_residual_dex": float(np.mean(residual)),
        "median_predicted_to_observed": float(np.median(np.power(10.0, residual))),
        "error_normalized_RMS": float(np.sqrt(np.mean(np.square(normalized)))),
        "fraction_within_25_percent": float(
            np.mean(np.abs(residual) <= np.log10(1.25))
        ),
    }


def galaxy_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    residual = prediction - frame["velocity_observed_adjusted_km_s"].to_numpy(float)
    per_system = (
        pd.DataFrame({"system": frame["system"], "squared": np.square(residual)})
        .groupby("system")["squared"]
        .mean()
    )
    normalized = residual / frame["velocity_error_total_km_s"].to_numpy(float)
    return {
        "systems": int(frame["system"].nunique()),
        "points": int(len(frame)),
        "RMSE_km_s": float(np.sqrt(np.mean(np.square(residual)))),
        "equal_system_RMSE_km_s": float(np.sqrt(per_system.mean())),
        "MAE_km_s": float(np.mean(np.abs(residual))),
        "mean_residual_km_s": float(np.mean(residual)),
        "error_normalized_RMS": float(np.sqrt(np.mean(np.square(normalized)))),
    }


def paired_system_bootstrap(
    frame: pd.DataFrame,
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    table = pd.DataFrame(
        {
            "system": frame["system"].to_numpy(),
            "squared_a": np.square(residual_a),
            "squared_b": np.square(residual_b),
        }
    )
    paired = table.groupby("system")[["squared_a", "squared_b"]].mean().to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(draws, len(paired)))
    sampled = paired[indices].mean(axis=1)
    delta = np.sqrt(sampled[:, 0]) - np.sqrt(sampled[:, 1])
    observed = float(np.sqrt(paired[:, 0].mean()) - np.sqrt(paired[:, 1].mean()))
    return {
        "definition": "equal-system RMSE A minus B; negative favors A",
        "observed_delta": observed,
        "percentile_95_interval": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "probability_A_better": float(np.mean(delta < 0.0)),
        "draws": int(draws),
    }


def cross_validate(
    galaxy: pd.DataFrame,
    cluster: pd.DataFrame,
    protocol: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = int(protocol["sample_and_splits"]["folds"])
    bounds = protocol["parameter_rule"]["bounds"]
    predictions = []
    fit_rows = []
    for fold in range(folds):
        galaxy_train = galaxy[
            (galaxy["fold"] != fold) & (galaxy["split"] == "inner_train")
        ]
        galaxy_test = galaxy[
            (galaxy["fold"] == fold) & (galaxy["split"] == "outer_holdout")
        ]
        cluster_train = cluster[cluster["fold"] != fold]
        cluster_test = cluster[cluster["fold"] == fold]
        fit_specs = {
            "metric_slip_gated": fit_cluster_kappa(
                cluster_train, bounds, gated=True
            ),
            "metric_slip_constant": fit_cluster_kappa(
                cluster_train, bounds, gated=False
            ),
            "single_potential_gated": fit_joint_kappa(
                galaxy_train, cluster_train, bounds
            ),
        }
        for model, (kappa, objective) in fit_specs.items():
            fit_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "kappa": kappa,
                    "training_objective": objective,
                    "heldout_galaxies": int(galaxy_test["system"].nunique()),
                    "heldout_clusters": int(cluster_test["system"].nunique()),
                }
            )
            cluster_prediction = predict_cluster(
                cluster_test,
                kappa,
                gated=model != "metric_slip_constant",
            )
            block = cluster_test.copy()
            block["model"] = model
            block["kappa"] = kappa
            block["predicted"] = cluster_prediction
            block["residual"] = cluster_prediction - block["log_gobs"]
            predictions.append(block)

            galaxy_prediction = (
                predict_galaxy_velocity(galaxy_test, kappa)
                if model == "single_potential_gated"
                else galaxy_test["velocity_predicted_km_s"].to_numpy(float)
            )
            block = galaxy_test.copy()
            block["model"] = model
            block["kappa"] = kappa
            block["predicted"] = galaxy_prediction
            block["residual"] = (
                galaxy_prediction - block["velocity_observed_adjusted_km_s"]
            )
            predictions.append(block)
    return pd.concat(predictions, ignore_index=True, sort=False), pd.DataFrame(fit_rows)


def gate_summary(frame: pd.DataFrame) -> dict:
    return {
        "minimum": float(frame["source_gate"].min()),
        "median": float(frame["source_gate"].median()),
        "maximum": float(frame["source_gate"].max()),
        "concentration_median": float(frame["source_concentration"].median()),
        "fraction_gate_above_0_1": float(np.mean(frame["source_gate"] > 0.1)),
    }


def make_figure(predictions: pd.DataFrame, fits: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    galaxy = predictions[predictions["domain"] == "galaxy"]
    cluster = predictions[predictions["domain"] == "cluster"]
    axes[0].hist(
        galaxy.drop_duplicates(["system", "radius_adjusted_kpc"])["source_gate"],
        bins=25,
        alpha=0.65,
        label="SPARC",
    )
    axes[0].hist(
        cluster.drop_duplicates(["system", "radius_kpc"])["source_gate"],
        bins=25,
        alpha=0.65,
        label="CLASH",
    )
    axes[0].set(xlabel="source-distribution gate F", ylabel="points", title="Measured gate")
    axes[0].legend(frameon=False)

    for model, block in cluster.groupby("model", sort=True):
        axes[1].scatter(
            block["log_gobs"],
            block["residual"],
            s=20,
            alpha=0.65,
            label=model.replace("_", " "),
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set(
        xlabel="lensing-derived log acceleration",
        ylabel="held-out residual (dex)",
        title="Cluster radial transfer",
    )
    axes[1].legend(frameon=False, fontsize=8)

    for model, block in fits.groupby("model", sort=True):
        axes[2].plot(block["fold"], block["kappa"], marker="o", label=model.replace("_", " "))
    axes[2].set(
        xlabel="held-out fold",
        ylabel="training-selected kappa",
        title="One setting per training fold",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "source_gated_metric_protocol.json",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_source_gated_scores":
        raise RuntimeError("protocol was not frozen before source-gated scoring")
    inputs = {key: ROOT / value for key, value in protocol["inputs"].items()}
    constants = protocol["fixed_constants"]
    split = protocol["sample_and_splits"]
    galaxy = load_galaxies(
        inputs["SPARC_fixed_RAR_predictions"], constants, split
    )
    cluster = load_clusters(inputs["CLASH_radial_points"], constants, split)
    predictions, fold_fits = cross_validate(galaxy, cluster, protocol)

    scores = {}
    for model, block in predictions.groupby("model", sort=True):
        scores[model] = {
            "galaxy_outer": galaxy_metrics(
                block[block["domain"] == "galaxy"],
                block.loc[block["domain"] == "galaxy", "predicted"].to_numpy(float),
            ),
            "cluster_heldout": cluster_metrics(
                block[block["domain"] == "cluster"],
                block.loc[block["domain"] == "cluster", "predicted"].to_numpy(float),
            ),
        }
    metric_cluster = predictions[
        (predictions["domain"] == "cluster")
        & (predictions["model"] == "metric_slip_gated")
    ]
    constant_cluster = predictions[
        (predictions["domain"] == "cluster")
        & (predictions["model"] == "metric_slip_constant")
    ]
    single_galaxy = predictions[
        (predictions["domain"] == "galaxy")
        & (predictions["model"] == "single_potential_gated")
    ]
    baseline_galaxy = predictions[
        (predictions["domain"] == "galaxy")
        & (predictions["model"] == "metric_slip_gated")
    ]
    bootstrap = {
        "gated_metric_slip_minus_constant_slip_cluster_RMSE_dex": paired_system_bootstrap(
            metric_cluster,
            metric_cluster["residual"].to_numpy(float),
            constant_cluster["residual"].to_numpy(float),
            draws=int(protocol["primary_metrics"]["bootstrap_draws"]),
            seed=int(protocol["primary_metrics"]["bootstrap_seed"]),
        ),
        "single_potential_minus_fixed_RAR_galaxy_RMSE_km_s": paired_system_bootstrap(
            single_galaxy,
            single_galaxy["residual"].to_numpy(float),
            baseline_galaxy["residual"].to_numpy(float),
            draws=int(protocol["primary_metrics"]["bootstrap_draws"]),
            seed=int(protocol["primary_metrics"]["bootstrap_seed"]) + 1,
        ),
    }

    bounds = protocol["parameter_rule"]["bounds"]
    galaxy_inner = galaxy[galaxy["split"] == "inner_train"]
    galaxy_preference = bounded_fit(
        lambda kappa: galaxy_normalized_mse(
            galaxy_inner, predict_galaxy_velocity(galaxy_inner, kappa)
        ),
        bounds,
    )
    cluster_preference = fit_cluster_kappa(cluster, bounds, gated=True)

    a_dagger = float(constants["a_dagger_m_s2"])
    full_kappa = float(fold_fits[fold_fits["model"] == "metric_slip_gated"]["kappa"].median())
    gm_sun = 1.32712440018e20
    saturn_radius_m = 8.43 * 149597870700.0
    saturn_gbar = gm_sun / saturn_radius_m**2
    saturn_gdyn = fixed_rar_acceleration(np.asarray([saturn_gbar]), a_dagger)
    worst_solar_gate = source_gate(
        [saturn_gbar],
        [0.0],
        acceleration_scale_m_s2=a_dagger,
        diffuseness_power=float(constants["diffuseness_power"]),
        concentration_power=float(constants["concentration_power"]),
    )
    solar_eta = source_gated_metric_eta(
        [saturn_gbar], saturn_gdyn, worst_solar_gate, full_kappa
    )

    gated_rmse = scores["metric_slip_gated"]["cluster_heldout"][
        "equal_system_RMSE_dex"
    ]
    constant_rmse = scores["metric_slip_constant"]["cluster_heldout"][
        "equal_system_RMSE_dex"
    ]
    raw_ratio = gated_rmse / constant_rmse
    raw_advance = raw_ratio <= float(
        protocol["raw_lensing_advancement"]["maximum_ratio_to_constant_slip"]
    )
    cluster_required_denominator = (
        cluster["source_gate"].to_numpy(float)
        * (
            cluster["g_RAR_m_s2"].to_numpy(float)
            - cluster["g_bar_m_s2"].to_numpy(float)
        )
    )
    cluster_required_kappa = (
        cluster["g_obs_m_s2"].to_numpy(float)
        - cluster["g_RAR_m_s2"].to_numpy(float)
    ) / cluster_required_denominator
    per_cluster_model_rmse = (
        predictions[predictions["domain"] == "cluster"]
        .assign(squared=lambda values: np.square(values["residual"]))
        .groupby(["system", "model"])["squared"]
        .mean()
        .unstack("model")
        .apply(np.sqrt)
    )
    gated_cluster_wins = int(
        (
            per_cluster_model_rmse["metric_slip_gated"]
            < per_cluster_model_rmse["metric_slip_constant"]
        ).sum()
    )
    previous = json.loads(
        inputs["CLASH_comparison_report"].read_text(encoding="utf-8")
    )
    sparc = json.loads(inputs["SPARC_report"].read_text(encoding="utf-8"))
    raw_metric = json.loads(
        inputs["raw_metric_slip_report"].read_text(encoding="utf-8")
    )
    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed held-out source-gated galaxy and cluster test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(protocol_path),
        },
        "input_hashes": {
            key: sha256(path) for key, path in inputs.items()
        },
        "formula": protocol["equations"],
        "sample": {
            "galaxies": int(galaxy["system"].nunique()),
            "galaxy_inner_points": int((galaxy["split"] == "inner_train").sum()),
            "galaxy_outer_points": int((galaxy["split"] == "outer_holdout").sum()),
            "clusters": int(cluster["system"].nunique()),
            "cluster_points": int(len(cluster)),
            "galaxy_gate": gate_summary(galaxy),
            "cluster_gate": gate_summary(cluster),
        },
        "scores": scores,
        "parameter_stability": {
            model: {
                "fold_values": list(map(float, block.sort_values("fold")["kappa"])),
                "median": float(block["kappa"].median()),
                "minimum": float(block["kappa"].min()),
                "maximum": float(block["kappa"].max()),
            }
            for model, block in fold_fits.groupby("model", sort=True)
        },
        "separate_domain_preference": {
            "galaxy_inner_best_kappa": float(galaxy_preference[0]),
            "galaxy_inner_objective": float(galaxy_preference[1]),
            "cluster_full_sample_best_kappa": float(cluster_preference[0]),
            "cluster_full_sample_objective": float(cluster_preference[1]),
        },
        "paired_complete_system_bootstrap": bootstrap,
        "raw_lensing_advancement": {
            "gated_to_constant_slip_RMSE_ratio": float(raw_ratio),
            "threshold": float(
                protocol["raw_lensing_advancement"][
                    "maximum_ratio_to_constant_slip"
                ]
            ),
            "advance": bool(raw_advance),
        },
        "failure_diagnostic": {
            "correlation_source_gate_with_pointwise_required_kappa": float(
                np.corrcoef(
                    cluster["source_gate"].to_numpy(float),
                    cluster_required_kappa,
                )[0, 1]
            ),
            "pointwise_required_kappa_median": float(
                np.median(cluster_required_kappa)
            ),
            "pointwise_required_kappa_10_to_90_percentile": [
                float(np.quantile(cluster_required_kappa, 0.1)),
                float(np.quantile(cluster_required_kappa, 0.9)),
            ],
            "clusters_where_gated_beats_constant": gated_cluster_wins,
            "clusters_compared": int(len(per_cluster_model_rmse)),
            "interpretation": "The gate is largest where the NFW-derived target requires a smaller added response, so one kappa over-boosts some radii while leaving others short.",
        },
        "Solar_System_worst_case_at_Saturn": {
            "assumed_concentration": 0.0,
            "gate": float(worst_solar_gate[0]),
            "eta_minus_one": float(solar_eta[0] - 1.0),
            "note": "A central point-mass profile has C=1 and exactly zero gate; this deliberately uses C=0 as a conservative bound.",
        },
        "context_comparators": {
            "fixed_RAR_SPARC_outer_RMSE_km_s": sparc["scores"][
                "fixed_RAR:invariant"
            ]["outer_holdout"]["RMSE_km_s"],
            "simple_MOND_SPARC_outer_RMSE_km_s": sparc["scores"][
                "simple_MOND:invariant"
            ]["outer_holdout"]["RMSE_km_s"],
            "previous_coherence_RG_cluster_RMSE_dex": previous[
                "cluster_lensing_metrics"
            ]["candidate"]["equal_system_RMSE_dex"],
            "cluster_retuned_RAR_RMSE_dex": previous[
                "cluster_lensing_metrics"
            ]["cluster_retuned_RAR"]["equal_system_RMSE_dex"],
            "NFW_derived_target_RMSE_dex": 0.0,
            "NFW_zero_warning": previous["cluster_lensing_metrics"][
                "NFW_construction"
            ]["independence_warning"]
            if "independence_warning"
            in previous["cluster_lensing_metrics"]["NFW_construction"]
            else "zero by construction on this NFW-derived target",
            "previous_raw_constant_slip_validation_RMSE_arcsec": raw_metric[
                "cross_cluster_validation"
            ]["selected_slip"]["equal_system_radial_RMS_arcsec"],
            "previous_raw_compact_halo_validation_RMSE_arcsec": raw_metric[
                "comparators"
            ]["compact_halo_validation"]["equal_system_radial_RMS_arcsec"],
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(ROOT / protocol["outputs"]["predictions"], index=False)
    fold_fits.to_csv(ROOT / protocol["outputs"]["fold_fits"], index=False)
    make_figure(predictions, fold_fits, ROOT / protocol["outputs"]["figure"])
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Source-gated metric test",
        "",
        "| formula | galaxy outer RMSE (km/s) | cluster held-out RMSE (dex) | median kappa |",
        "|---|---:|---:|---:|",
    ]
    for model in (
        "metric_slip_gated",
        "metric_slip_constant",
        "single_potential_gated",
    ):
        lines.append(
            f"| {model} | {scores[model]['galaxy_outer']['RMSE_km_s']:.3f} | "
            f"{scores[model]['cluster_heldout']['equal_system_RMSE_dex']:.3f} | "
            f"{report['parameter_stability'][model]['median']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Gated/constant cluster RMSE ratio: **{raw_ratio:.3f}**.",
            f"Advance to a new raw image-plane run: **{raw_advance}**.",
        ]
    )
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
