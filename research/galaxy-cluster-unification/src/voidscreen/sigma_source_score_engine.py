"""Posterior decision engine for the frozen V19BL source-only gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from voidscreen.sigma_source_invariants import (
    analytic_press_unexplained_fraction,
    axial_angle_deg,
    axial_difference_deg,
    axial_interval_summary_deg,
    gradient_detection_sigma,
    robust_detection_sigma,
    symmetric_fractional_change,
)

FloatArray = NDArray[np.float64]


def gradient_support_mask(
    gradients: Sequence[tuple[ArrayLike, ArrayLike]],
    *,
    minimum_detection_sigma: float,
) -> NDArray[np.bool_]:
    """Require every named two-component gradient to be detected per region."""

    threshold = float(minimum_detection_sigma)
    if not math.isfinite(threshold) or threshold <= 0.0 or not gradients:
        raise ValueError("a positive threshold and at least one gradient are required")
    shapes = [np.asarray(east).shape for east, _ in gradients]
    if len(set(shapes)) != 1 or len(shapes[0]) != 2:
        raise ValueError("gradient components must share a draw-by-region shape")
    draws, regions = shapes[0]
    if draws < 3 or regions < 1:
        raise ValueError("gradient posterior is undersized")
    support = np.ones(regions, dtype=bool)
    for east, north in gradients:
        east_values = np.asarray(east, dtype=float)
        north_values = np.asarray(north, dtype=float)
        if north_values.shape != east_values.shape:
            raise ValueError("gradient east/north components differ in shape")
        for region in range(regions):
            support[region] &= (
                gradient_detection_sigma(
                    east_values[:, region], north_values[:, region]
                )
                >= threshold
            )
    return support


def i4_draw_summary(
    q_plus: ArrayLike, q_cross: ArrayLike, support: ArrayLike
) -> dict[str, FloatArray]:
    plus = np.asarray(q_plus, dtype=float)
    cross = np.asarray(q_cross, dtype=float)
    mask = np.asarray(support, dtype=bool)
    if plus.ndim != 2 or cross.shape != plus.shape or mask.shape != (plus.shape[1],):
        raise ValueError("I4 inputs must be draw-by-region components and a region mask")
    if not np.any(mask) or not np.all(np.isfinite(plus[:, mask])) or not np.all(np.isfinite(cross[:, mask])):
        raise ValueError("I4 support must contain finite regions")
    activation = np.sqrt(np.mean(2.0 * (plus[:, mask] ** 2 + cross[:, mask] ** 2), axis=1))
    mean_plus = np.mean(plus[:, mask], axis=1)
    mean_cross = np.mean(cross[:, mask], axis=1)
    resultant = np.hypot(mean_plus, mean_cross)
    if np.any(resultant <= np.finfo(float).tiny):
        raise ValueError("an I4 draw has no global axial direction")
    return {
        "activation": activation,
        "axis_deg": axial_angle_deg(mean_plus, mean_cross),
        "mean_q_plus": mean_plus,
        "mean_q_cross": mean_cross,
    }


def i5_draw_summary(baroclinicity: ArrayLike, support: ArrayLike) -> dict[str, FloatArray]:
    values = np.asarray(baroclinicity, dtype=float)
    mask = np.asarray(support, dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[1],) or not np.any(mask):
        raise ValueError("I5 inputs must be draw-by-region values and a nonempty mask")
    selected = values[:, mask]
    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0) or np.any(selected > 1.0):
        raise ValueError("supported I5 values must lie in [0,1]")
    return {"activation": np.mean(selected, axis=1)}


def posterior_novelty_scores(
    controls: ArrayLike,
    response: ArrayLike,
    support: ArrayLike,
    *,
    minimum_unexplained_fraction: float,
) -> dict[str, Any]:
    """Apply the exact quadratic PRESS null independently to every draw."""

    predictors = np.asarray(controls, dtype=float)
    target = np.asarray(response, dtype=float)
    mask = np.asarray(support, dtype=bool)
    if predictors.ndim != 3 or predictors.shape[2] != 5:
        raise ValueError("controls must have draw-by-region-by-five shape")
    if target.ndim == 2:
        target = target[..., None]
    if target.ndim != 3 or target.shape[:2] != predictors.shape[:2]:
        raise ValueError("response must share draw and region axes with controls")
    if mask.shape != (predictors.shape[1],) or np.count_nonzero(mask) <= 21:
        raise ValueError("support must contain more than 21 regions")
    fractions = np.empty(predictors.shape[0], dtype=float)
    maximum_leverage = np.empty_like(fractions)
    for draw in range(predictors.shape[0]):
        score = analytic_press_unexplained_fraction(
            predictors[draw, mask], target[draw, mask]
        )
        fractions[draw] = float(score["joint_unexplained_fraction"])
        maximum_leverage[draw] = float(score["maximum_leverage"])
    threshold = float(minimum_unexplained_fraction)
    return {
        "unexplained_fraction": fractions,
        "maximum_leverage": maximum_leverage,
        "pass_fraction": float(np.mean(fractions >= threshold)),
    }


def leave_one_region_out_stability(
    response: ArrayLike,
    support: ArrayLike,
    *,
    candidate: str,
    maximum_activation_change_fraction: float,
    maximum_axis_change_deg: float,
) -> dict[str, Any]:
    """Score omissions on posterior-median regional candidate features."""

    values = np.asarray(response, dtype=float)
    mask = np.asarray(support, dtype=bool)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3 or mask.shape != (values.shape[1],):
        raise ValueError("response and support shapes differ")
    indices = np.flatnonzero(mask)
    median = np.median(values[:, indices], axis=0)
    if candidate == "I4":
        if median.shape[1] != 2:
            raise ValueError("I4 requires q_plus/q_cross response components")
        base_activation = math.sqrt(float(np.mean(2.0 * np.sum(median**2, axis=1))))
        base_axis = float(axial_angle_deg(np.mean(median[:, 0]), np.mean(median[:, 1])))
    elif candidate == "I5":
        if median.shape[1] != 1:
            raise ValueError("I5 requires one response component")
        base_activation = float(np.mean(median[:, 0]))
        base_axis = math.nan
    else:
        raise ValueError("candidate must be I4 or I5")
    passes: list[bool] = []
    activation_changes: list[float] = []
    axis_changes: list[float] = []
    for omitted in range(indices.size):
        retained = np.ones(indices.size, dtype=bool)
        retained[omitted] = False
        subset = median[retained]
        if candidate == "I4":
            activation = math.sqrt(float(np.mean(2.0 * np.sum(subset**2, axis=1))))
            axis = float(axial_angle_deg(np.mean(subset[:, 0]), np.mean(subset[:, 1])))
            axis_change = float(axial_difference_deg(axis, base_axis))
        else:
            activation = float(np.mean(subset[:, 0]))
            axis_change = 0.0
        activation_change = float(symmetric_fractional_change(activation, base_activation))
        activation_changes.append(activation_change)
        axis_changes.append(axis_change)
        passes.append(
            activation_change <= maximum_activation_change_fraction
            and axis_change <= maximum_axis_change_deg
        )
    return {
        "pass_fraction": float(np.mean(passes)),
        "maximum_activation_change_fraction": max(activation_changes),
        "maximum_axis_change_deg": max(axis_changes),
    }


def posterior_feature_summary(
    draw_summary: Mapping[str, ArrayLike],
) -> dict[str, Any]:
    activation = np.asarray(draw_summary["activation"], dtype=float)
    result: dict[str, Any] = {
        "activation_percentiles": {
            token: float(value)
            for token, value in zip(
                ("q05", "q16", "median", "q84", "q95"),
                np.percentile(activation, [5.0, 16.0, 50.0, 84.0, 95.0]),
                strict=True,
            )
        },
        "detection_sigma": robust_detection_sigma(activation),
    }
    if "axis_deg" in draw_summary:
        result["axial_posterior"] = axial_interval_summary_deg(
            np.asarray(draw_summary["axis_deg"], dtype=float)
        )
    return result


def joint_variant_draw_pass_fraction(
    primary: Mapping[str, ArrayLike],
    variants: Sequence[Mapping[str, ArrayLike]],
    *,
    maximum_activation_change_fraction: float,
    maximum_axis_change_deg: float,
) -> float:
    activation = np.asarray(primary["activation"], dtype=float)
    passes = np.ones(activation.shape, dtype=bool)
    for variant in variants:
        changed = symmetric_fractional_change(
            activation, np.asarray(variant["activation"], dtype=float)
        )
        passes &= changed <= maximum_activation_change_fraction
        if "axis_deg" in primary:
            if "axis_deg" not in variant:
                raise ValueError("a directional primary requires directional variants")
            passes &= axial_difference_deg(
                np.asarray(primary["axis_deg"], dtype=float),
                np.asarray(variant["axis_deg"], dtype=float),
            ) <= maximum_axis_change_deg
    return float(np.mean(passes))
