"""Noncircular CLASH validation and grouped parameter diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .datasets import normalize_cluster_name
from .model import (
    DEFAULT_A0,
    DEFAULT_G_DAGGER,
    amplitude_from_path_length,
    infer_B,
    predict_acceleration,
)


@dataclass(frozen=True)
class FitResult:
    parameters: dict
    chi2: float
    dof: int
    rms_dex: float
    median_abs_dex: float


def submitted_cluster_amplitude() -> float:
    return amplitude_from_path_length(600.0, A0=DEFAULT_A0, L0_kpc=0.4, n=0.27)


def _prediction_log(frame: pd.DataFrame, B) -> np.ndarray:
    return np.log10(predict_acceleration(frame["gbar"].to_numpy(), B))


def _model_slope_log_gbar(gbar: np.ndarray, B) -> np.ndarray:
    epsilon = 1e-5
    upper = np.log10(predict_acceleration(gbar * np.exp(epsilon), B))
    lower = np.log10(predict_acceleration(gbar * np.exp(-epsilon), B))
    return (upper - lower) / (2.0 * epsilon / np.log(10.0))


def residual_sigma(frame: pd.DataFrame, B) -> np.ndarray:
    slope = _model_slope_log_gbar(frame["gbar"].to_numpy(), B)
    return np.sqrt(
        frame["err_log_gtot"].to_numpy() ** 2
        + (slope * frame["err_log_gbar"].to_numpy()) ** 2
    )


def score_prediction(frame: pd.DataFrame, B) -> tuple[FitResult, pd.DataFrame]:
    prediction = _prediction_log(frame, B)
    residual = prediction - frame["log_gtot"].to_numpy()
    sigma = residual_sigma(frame, B)
    output = frame.copy()
    output["B_model"] = np.asarray(B) if np.ndim(B) else float(B)
    output["log_gtot_pred"] = prediction
    output["residual_dex"] = residual
    output["sigma_residual_dex"] = sigma
    output["ratio_predicted_observed"] = np.power(10.0, residual)
    result = FitResult(
        parameters={},
        chi2=float(np.sum((residual / sigma) ** 2)),
        dof=int(len(frame)),
        rms_dex=float(np.sqrt(np.mean(residual**2))),
        median_abs_dex=float(np.median(np.abs(residual))),
    )
    return result, output


def fit_constant_B(frame: pd.DataFrame) -> FitResult:
    def objective(log_B):
        B = float(np.exp(log_B[0]))
        residual = _prediction_log(frame, B) - frame["log_gtot"].to_numpy()
        return residual / residual_sigma(frame, B)

    fit = least_squares(objective, np.log([5.0]), bounds=(np.log([0.05]), np.log([100.0])))
    B = float(np.exp(fit.x[0]))
    scored, _ = score_prediction(frame, B)
    return FitResult(
        parameters={"B": B},
        chi2=scored.chi2,
        dof=max(1, len(frame) - 1),
        rms_dex=scored.rms_dex,
        median_abs_dex=scored.median_abs_dex,
    )


def radial_B(radius_kpc, B200: float, exponent: float):
    return B200 * np.power(np.asarray(radius_kpc, dtype=float) / 200.0, exponent)


def fit_radial_B(frame: pd.DataFrame) -> FitResult:
    radii = frame["radius_kpc"].to_numpy()

    def objective(parameters):
        B = radial_B(radii, np.exp(parameters[0]), parameters[1])
        residual = _prediction_log(frame, B) - frame["log_gtot"].to_numpy()
        return residual / residual_sigma(frame, B)

    fit = least_squares(
        objective,
        [np.log(5.0), -0.1],
        bounds=([np.log(0.05), -2.0], [np.log(100.0), 2.0]),
    )
    B200, exponent = float(np.exp(fit.x[0])), float(fit.x[1])
    B = radial_B(radii, B200, exponent)
    scored, _ = score_prediction(frame, B)
    return FitResult(
        parameters={"B200": B200, "radial_exponent": exponent},
        chi2=scored.chi2,
        dof=max(1, len(frame) - 2),
        rms_dex=scored.rms_dex,
        median_abs_dex=scored.median_abs_dex,
    )


def leave_one_cluster_out(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    predictions = []
    fit_rows = []
    for cluster in sorted(frame["cluster"].unique()):
        train = frame[frame["cluster"] != cluster]
        test = frame[frame["cluster"] == cluster]
        constant = fit_constant_B(train)
        radial = fit_radial_B(train)
        for model, fit in (("constant_B", constant), ("radial_B_diagnostic", radial)):
            if model == "constant_B":
                B = fit.parameters["B"]
            else:
                B = radial_B(
                    test["radius_kpc"],
                    fit.parameters["B200"],
                    fit.parameters["radial_exponent"],
                )
            _, scored = score_prediction(test, B)
            scored["held_out_cluster"] = cluster
            scored["model"] = model
            predictions.append(scored)
            fit_rows.append({"held_out_cluster": cluster, "model": model, **fit.parameters})
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary = {}
    for model, group in prediction_frame.groupby("model"):
        residual = group["residual_dex"].to_numpy()
        summary[model] = {
            "n_points": int(len(group)),
            "n_clusters": int(group["cluster"].nunique()),
            "rms_dex": float(np.sqrt(np.mean(residual**2))),
            "median_abs_dex": float(np.median(np.abs(residual))),
            "mean_residual_dex": float(np.mean(residual)),
        }
    return prediction_frame, {"summary": summary, "fits": fit_rows}


def grouped_bootstrap_posteriors(frame: pd.DataFrame, draws: int = 500, seed: int = 20260718):
    """Cluster bootstrap for empirical parameter uncertainty, preserving grouping."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(sorted(frame["cluster"].unique()))
    rows = []
    for draw in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for occurrence, cluster in enumerate(sampled):
            piece = frame[frame["cluster"] == cluster].copy()
            piece["bootstrap_group"] = f"{cluster}_{occurrence}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        constant = fit_constant_B(sample)
        radial = fit_radial_B(sample)
        rows.append(
            {
                "draw": draw,
                "B_constant": constant.parameters["B"],
                "B200_radial": radial.parameters["B200"],
                "radial_exponent": radial.parameters["radial_exponent"],
            }
        )
    return pd.DataFrame(rows)


def _weighted_radial_residual_slope(frame: pd.DataFrame) -> float:
    """Weighted slope of log residual against log10(radius / 200 kpc)."""
    x = np.log10(frame["radius_kpc"].to_numpy(dtype=float) / 200.0)
    y = frame["residual_dex"].to_numpy(dtype=float)
    sigma = frame["sigma_residual_dex"].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(sigma, 1e-12) ** 2
    design = np.column_stack([np.ones(len(x)), x])
    normal = design.T @ (weights[:, None] * design)
    target = design.T @ (weights * y)
    return float(np.linalg.solve(normal, target)[1])


def cluster_bootstrap_radial_trend(
    frame: pd.DataFrame,
    draws: int = 5000,
    seed: int = 20260718,
) -> dict:
    """Quantify the no-refit radial bias while resampling complete clusters."""
    clusters = np.asarray(sorted(frame["cluster"].unique()))
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for occurrence, cluster in enumerate(sampled):
            piece = frame[frame["cluster"] == cluster].copy()
            piece["bootstrap_group"] = f"{cluster}_{occurrence}"
            pieces.append(piece)
        slopes.append(_weighted_radial_residual_slope(pd.concat(pieces, ignore_index=True)))
    slopes = np.asarray(slopes)
    return {
        "definition": "weighted residual_dex versus log10(radius_kpc/200), cluster bootstrap",
        "slope_dex_per_radius_dex": _weighted_radial_residual_slope(frame),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "bootstrap_95_percent_interval": [
            float(np.quantile(slopes, 0.025)),
            float(np.quantile(slopes, 0.975)),
        ],
    }


def audit_tian(frame: pd.DataFrame, fox_names: set[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["normalized_cluster"] = data["cluster"].map(normalize_cluster_name)
    data["overlaps_fox_calibration"] = data["normalized_cluster"].isin(fox_names)
    disjoint = data[~data["overlaps_fox_calibration"]].copy()
    submitted_B = submitted_cluster_amplitude()
    submitted_score, submitted_rows = score_prediction(disjoint, submitted_B)
    submitted_rows["B_obs"] = infer_B(submitted_rows["gbar"], submitted_rows["gtot"])
    constant = fit_constant_B(data)
    radial = fit_radial_B(data)
    loco_rows, loco = leave_one_cluster_out(data)
    by_radius = []
    _, all_submitted_rows = score_prediction(data, submitted_B)
    all_submitted_rows["B_obs"] = infer_B(all_submitted_rows["gbar"], all_submitted_rows["gtot"])
    for radius, group in submitted_rows.groupby("radius_kpc"):
        positive = group[group["B_obs"] > 0]
        by_radius.append(
            {
                "radius_kpc": float(radius),
                "n": int(len(group)),
                "median_predicted_observed": float(group["ratio_predicted_observed"].median()),
                "median_B_obs_positive": float(positive["B_obs"].median()) if len(positive) else None,
            }
        )
    equivalent_n = np.log(constant.parameters["B"] / DEFAULT_A0) / np.log(600.0 / 0.4)
    summary = {
        "submitted": {
            "B": submitted_B,
            "disjoint_points": int(len(disjoint)),
            "disjoint_clusters": int(disjoint["cluster"].nunique()),
            "fox_overlap_clusters": sorted(data.loc[data["overlaps_fox_calibration"], "cluster"].unique()),
            "overlap_rule": (
                "lowercase cluster names, remove non-alphanumeric characters, normalize "
                "Abell and MACS aliases, then require exact equality with a Fox calibration name"
            ),
            "median_predicted_observed": float(submitted_rows["ratio_predicted_observed"].median()),
            "mean_residual_dex": float(submitted_rows["residual_dex"].mean()),
            "rms_dex": submitted_score.rms_dex,
            "median_abs_dex": submitted_score.median_abs_dex,
            "radial_residual_trend": cluster_bootstrap_radial_trend(submitted_rows),
        },
        "all_points_constant_refit": {
            **constant.parameters,
            "equivalent_n_at_600_kpc": float(equivalent_n),
            "chi2_per_dof": constant.chi2 / constant.dof,
            "rms_dex": constant.rms_dex,
            "median_abs_dex": constant.median_abs_dex,
        },
        "all_points_radial_diagnostic": {
            **radial.parameters,
            "chi2_per_dof": radial.chi2 / radial.dof,
            "rms_dex": radial.rms_dex,
            "median_abs_dex": radial.median_abs_dex,
        },
        "leave_one_cluster_out": loco["summary"],
        "by_radius": by_radius,
    }
    return summary, all_submitted_rows, loco_rows
