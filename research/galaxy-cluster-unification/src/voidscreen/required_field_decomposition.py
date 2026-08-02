"""Diagnostics for decomposing a spent required lens-deflection field."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class RadialVectorDecomposition:
    monopole_x: np.ndarray
    monopole_y: np.ndarray
    angular_x: np.ndarray
    angular_y: np.ndarray
    radial_component: np.ndarray
    tangential_component: np.ndarray
    table: pd.DataFrame


def vector_rms(x_values, y_values, mask=None) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    selected = np.ones(x.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    return float(np.sqrt(np.mean(x[selected] ** 2 + y[selected] ** 2)))


def radial_vector_decomposition(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radial_edges: np.ndarray,
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> RadialVectorDecomposition:
    """Split a vector field into a binned radial monopole and exact residual."""
    field_x, field_y, grid_x, grid_y = np.broadcast_arrays(
        np.asarray(alpha_x, dtype=float),
        np.asarray(alpha_y, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    edges = np.asarray(radial_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("radial_edges must be a strictly increasing vector")
    dx = grid_x - float(center_x)
    dy = grid_y - float(center_y)
    radius = np.hypot(dx, dy)
    safe = np.maximum(radius, np.finfo(float).tiny)
    radial_x = dx / safe
    radial_y = dy / safe
    radial_x = np.where(radius > 0.0, radial_x, 1.0)
    radial_y = np.where(radius > 0.0, radial_y, 0.0)
    tangential_x = -radial_y
    tangential_y = radial_x
    radial_component = field_x * radial_x + field_y * radial_y
    tangential_component = field_x * tangential_x + field_y * tangential_y
    monopole_radial = np.zeros_like(field_x)
    rows = []
    for index, (lower, upper) in enumerate(pairwise(edges)):
        use = (radius >= lower) & (radius < upper if index < len(edges) - 2 else radius <= upper)
        count = int(np.sum(use))
        mean_radial = float(np.mean(radial_component[use])) if count else float("nan")
        mean_tangential = float(np.mean(tangential_component[use])) if count else float("nan")
        if count:
            monopole_radial[use] = mean_radial
        rows.append(
            {
                "radial_bin": index,
                "radius_minimum": float(lower),
                "radius_maximum": float(upper),
                "radius_midpoint": 0.5 * float(lower + upper),
                "samples": count,
                "mean_radial_deflection": mean_radial,
                "mean_tangential_deflection": mean_tangential,
            }
        )
    monopole_x = monopole_radial * radial_x
    monopole_y = monopole_radial * radial_y
    return RadialVectorDecomposition(
        monopole_x=monopole_x,
        monopole_y=monopole_y,
        angular_x=field_x - monopole_x,
        angular_y=field_y - monopole_y,
        radial_component=radial_component,
        tangential_component=tangential_component,
        table=pd.DataFrame(rows),
    )


def convergence_and_jacobian_determinant(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return effective convergence, curl, and lens-Jacobian determinant."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    x_values = np.asarray(alpha_x, dtype=float)
    y_values = np.asarray(alpha_y, dtype=float)
    dax_dx = np.gradient(x_values, float(spacing), axis=0, edge_order=2)
    dax_dy = np.gradient(x_values, float(spacing), axis=1, edge_order=2)
    day_dx = np.gradient(y_values, float(spacing), axis=0, edge_order=2)
    day_dy = np.gradient(y_values, float(spacing), axis=1, edge_order=2)
    convergence = 0.5 * (dax_dx + day_dy)
    curl = day_dx - dax_dy
    determinant = (1.0 - dax_dx) * (1.0 - day_dy) - dax_dy * day_dx
    return convergence, curl, determinant


def sign_change_cells(values: np.ndarray) -> int:
    data = np.asarray(values, dtype=float)
    horizontal = data[:-1, :] * data[1:, :] <= 0.0
    vertical = data[:, :-1] * data[:, 1:] <= 0.0
    return int(np.sum(horizontal) + np.sum(vertical))


def angular_harmonics(
    radial_component: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    modes: list[int],
    *,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> pd.DataFrame:
    angle = np.arctan2(np.asarray(y) - center_y, np.asarray(x) - center_x)
    radial = np.asarray(radial_component, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    denominator = max(float(np.sum(np.abs(radial[selected]))), np.finfo(float).tiny)
    rows = []
    for mode in modes:
        cosine = float(np.sum(radial[selected] * np.cos(mode * angle[selected])))
        sine = float(np.sum(radial[selected] * np.sin(mode * angle[selected])))
        rows.append(
            {
                "mode": int(mode),
                "cosine_moment": cosine / denominator,
                "sine_moment": sine / denominator,
                "amplitude": float(np.hypot(cosine, sine)) / denominator,
                "phase_radian": float(np.arctan2(sine, cosine) / float(mode)),
            }
        )
    return pd.DataFrame(rows)


def predictor_correlations(
    predictors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    mask: np.ndarray,
) -> pd.DataFrame:
    selected = np.asarray(mask, dtype=bool)
    rows = []
    for predictor_name, predictor in predictors.items():
        x_values = np.asarray(predictor, dtype=float)[selected]
        for target_name, target in targets.items():
            y_values = np.asarray(target, dtype=float)[selected]
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            correlation = spearmanr(x_values[finite], y_values[finite])
            rows.append(
                {
                    "predictor": predictor_name,
                    "target": target_name,
                    "samples": int(np.sum(finite)),
                    "spearman_rho": float(correlation.statistic),
                    "p_value": float(correlation.pvalue),
                }
            )
    return pd.DataFrame(rows)


def positive_weight_radius_quantile(
    radius: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0,1]")
    radii = np.asarray(radius, dtype=float).ravel()
    values = np.maximum(np.asarray(weights, dtype=float).ravel(), 0.0)
    if float(np.sum(values)) <= 0.0:
        return float("nan")
    order = np.argsort(radii)
    cumulative = np.cumsum(values[order]) / np.sum(values)
    return float(radii[order][np.searchsorted(cumulative, quantile, side="left")])
