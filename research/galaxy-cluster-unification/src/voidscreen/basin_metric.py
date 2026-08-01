"""Weak-field response of the NBM0 conformal-disformal physical metric."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


C_M_S = 299_792_458.0


@dataclass(frozen=True)
class BasinMetricCoefficients:
    dynamics: float
    spatial_curvature: float
    weyl_half: float


def basin_metric_coefficients(alpha: float, beta: float) -> BasinMetricCoefficients:
    """Return coefficients multiplying c^2 X in Psi, Phi, and (Phi+Psi)/2."""
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("alpha and beta must be finite")
    return BasinMetricCoefficients(
        dynamics=alpha - beta,
        spatial_curvature=-alpha,
        weyl_half=-0.5 * beta,
    )


def weak_field_potentials(
    newtonian_potential_m2_s2: np.ndarray | float,
    basin_field: np.ndarray | float,
    *,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Psi, Phi, and the half-Weyl potential for the physical metric."""
    newtonian = np.asarray(newtonian_potential_m2_s2, dtype=np.float64)
    field = np.asarray(basin_field, dtype=np.float64)
    newtonian, field = np.broadcast_arrays(newtonian, field)
    if not np.isfinite(newtonian).all() or not np.isfinite(field).all():
        raise ValueError("potentials and basin field must be finite")
    coefficients = basin_metric_coefficients(alpha, beta)
    scale = C_M_S**2 * field
    psi = newtonian + coefficients.dynamics * scale
    phi = newtonian + coefficients.spatial_curvature * scale
    return psi, phi, 0.5 * (psi + phi)


def lensing_to_dynamics_extra_ratio(alpha: float, beta: float) -> float:
    """Return the metric-derived X response ratio; infinity means dynamics blind."""
    coefficients = basin_metric_coefficients(alpha, beta)
    if coefficients.dynamics == 0.0:
        if coefficients.weyl_half == 0.0:
            return math.nan
        return math.copysign(math.inf, coefficients.weyl_half)
    return coefficients.weyl_half / coefficients.dynamics


def beta_for_response_ratio(alpha: float, target_ratio: float) -> float:
    """Solve q=-beta/[2(alpha-beta)] for beta when a finite solution exists."""
    if not math.isfinite(alpha) or not math.isfinite(target_ratio):
        raise ValueError("alpha and target_ratio must be finite")
    denominator = 2.0 * target_ratio - 1.0
    if denominator == 0.0:
        if alpha == 0.0:
            raise ValueError("q=1/2 with alpha=0 does not identify beta")
        raise ValueError("q=1/2 requires the pure-disformal alpha=0 limit")
    return 2.0 * target_ratio * alpha / denominator
