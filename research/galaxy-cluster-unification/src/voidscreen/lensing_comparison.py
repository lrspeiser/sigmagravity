"""Metrics for grouped radial-lensing model comparisons.

Residuals use ``predicted log10(acceleration) - observed log10(acceleration)``.
The helpers deliberately keep complete lens systems together when estimating
uncertainty; radial points from one cluster are not independent bootstrap units.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def equal_system_rmse(system, residual_dex) -> float:
    """Return RMSE after assigning equal weight to each complete system."""
    residual = _finite_vector(residual_dex, "residual_dex")
    labels = np.asarray(system)
    if labels.ndim != 1 or len(labels) != len(residual):
        raise ValueError("system must be one-dimensional and match residual_dex")
    scored = pd.DataFrame({"system": labels, "squared": np.square(residual)})
    return float(np.sqrt(scored.groupby("system", sort=True)["squared"].mean().mean()))


def lensing_metrics(
    system,
    residual_dex,
    *,
    sigma_dex=None,
    radius_kpc=None,
    coverage_factors=(1.10, 1.25, 1.50, 2.00),
) -> dict[str, object]:
    """Summarize signed log residuals without fitting a calibration offset."""
    residual = _finite_vector(residual_dex, "residual_dex")
    labels = np.asarray(system)
    if labels.ndim != 1 or len(labels) != len(residual):
        raise ValueError("system must be one-dimensional and match residual_dex")

    mean = float(np.mean(residual))
    point_rmse = float(np.sqrt(np.mean(np.square(residual))))
    median_absolute = float(np.median(np.abs(residual)))
    output: dict[str, object] = {
        "systems": int(len(np.unique(labels))),
        "points": int(len(residual)),
        "equal_system_RMSE_dex": equal_system_rmse(labels, residual),
        "point_RMSE_dex": point_rmse,
        "mean_residual_dex": mean,
        "median_absolute_residual_dex": median_absolute,
        "geometric_mean_predicted_to_observed": float(10.0**mean),
        "RMSE_expressed_as_multiplicative_factor": float(10.0**point_rmse),
        "median_absolute_error_factor": float(10.0**median_absolute),
        "bias_corrected_point_RMSE_dex": float(
            np.sqrt(np.mean(np.square(residual - mean)))
        ),
        "posthoc_multiplier_to_remove_mean_bias": float(10.0 ** (-mean)),
        "coverage_within_symmetric_factor": {
            f"{factor:.2f}": float(np.mean(np.abs(residual) <= math.log10(factor)))
            for factor in coverage_factors
        },
    }

    if sigma_dex is not None:
        sigma = _finite_vector(sigma_dex, "sigma_dex")
        if len(sigma) != len(residual) or np.any(sigma <= 0.0):
            raise ValueError("sigma_dex must be positive and match residual_dex")
        output["diagonal_error_normalized_RMS"] = float(
            np.sqrt(np.mean(np.square(residual / sigma)))
        )

    if radius_kpc is not None:
        radius = _finite_vector(radius_kpc, "radius_kpc")
        if len(radius) != len(residual) or np.any(radius <= 0.0):
            raise ValueError("radius_kpc must be positive and match residual_dex")
        output["radial_residual_slope_dex_per_dex"] = float(
            np.polyfit(np.log10(radius), residual, 1)[0]
        )
        slopes = []
        frame = pd.DataFrame({"system": labels, "radius": radius, "residual": residual})
        for _, block in frame.groupby("system", sort=True):
            if len(block) >= 2:
                slopes.append(
                    float(np.polyfit(np.log10(block["radius"]), block["residual"], 1)[0])
                )
        output["median_system_radial_residual_slope_dex_per_dex"] = (
            float(np.median(slopes)) if slopes else None
        )
    return output


def paired_system_bootstrap(
    system,
    candidate_residual_dex,
    reference_residual_dex,
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap equal-system RMSE differences using systems as sampling units."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    candidate = _finite_vector(candidate_residual_dex, "candidate_residual_dex")
    reference = _finite_vector(reference_residual_dex, "reference_residual_dex")
    labels = np.asarray(system)
    if len(candidate) != len(reference) or len(labels) != len(candidate):
        raise ValueError("system and both residual vectors must have the same length")

    groups = [
        np.flatnonzero(labels == label)
        for label in sorted(np.unique(labels).tolist(), key=str)
    ]
    candidate_mse = np.asarray([np.mean(np.square(candidate[index])) for index in groups])
    reference_mse = np.asarray([np.mean(np.square(reference[index])) for index in groups])
    observed = float(np.sqrt(candidate_mse.mean()) - np.sqrt(reference_mse.mean()))

    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        picked = rng.integers(0, len(groups), size=len(groups))
        deltas[draw] = np.sqrt(candidate_mse[picked].mean()) - np.sqrt(
            reference_mse[picked].mean()
        )
    return {
        "definition": "candidate minus reference equal-system RMSE; negative favors candidate",
        "observed_delta_dex": observed,
        "percentile_95_interval_dex": list(map(float, np.percentile(deltas, [2.5, 97.5]))),
        "probability_candidate_better": float(np.mean(deltas < 0.0)),
        "draws": int(draws),
        "sampling_unit": "complete cluster",
    }
