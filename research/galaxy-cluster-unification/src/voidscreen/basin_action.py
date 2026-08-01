"""Reciprocal weak-field controls for the nonlocal basin-metric program.

The module deliberately stops short of fitting astronomical data.  It turns the
minimal canonical action into its point-source response, exposes the combinations
that observations can identify, and supplies structural tests for radial shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19
M_SUN_KG = 1.98847e30


@dataclass(frozen=True)
class ReciprocalDustCouplings:
    """Observable combinations implied by a canonical reciprocal dust source."""

    alpha: float
    beta: float
    source_d: float
    dynamics_amplitude: float
    lensing_amplitude: float
    lensing_to_dynamics_ratio: float


@dataclass(frozen=True)
class YukawaRecovery:
    """Effective-parameter result for a theory-free synthetic recovery check."""

    dynamics_amplitude: float
    range_m: float
    lensing_to_dynamics_ratio: float
    cost: float
    maximum_absolute_log_residual: float
    jacobian_condition_number: float
    success: bool


def reciprocal_dust_couplings(alpha: float, beta: float) -> ReciprocalDustCouplings:
    """Return the source and metric responses fixed by action reciprocity.

    For the physical metric

        g_tilde = exp(2 alpha X) (g + 2 beta X U U),

    nonrelativistic matter sources the canonical scalar through
    ``d = alpha - beta``.  A point mass then gives a dynamical Yukawa amplitude
    ``A_dyn = 2 d**2`` relative to Newtonian gravity and a Weyl/lensing amplitude
    ``A_lens = -beta*d``.  The simultaneous sign flip of ``d, alpha, beta`` is a
    field-redefinition degeneracy, not a second observable solution.
    """
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("alpha and beta must be finite")
    source_d = alpha - beta
    dynamics_amplitude = 2.0 * source_d**2
    lensing_amplitude = -beta * source_d
    if dynamics_amplitude == 0.0:
        ratio = math.nan
    else:
        ratio = lensing_amplitude / dynamics_amplitude
    return ReciprocalDustCouplings(
        alpha=alpha,
        beta=beta,
        source_d=source_d,
        dynamics_amplitude=dynamics_amplitude,
        lensing_amplitude=lensing_amplitude,
        lensing_to_dynamics_ratio=ratio,
    )


def metric_couplings_from_effective(
    dynamics_amplitude: float,
    lensing_to_dynamics_ratio: float,
    *,
    field_sign: int = 1,
) -> ReciprocalDustCouplings:
    """Reconstruct one of the two field-sign-equivalent metric couplings."""
    if not math.isfinite(dynamics_amplitude) or dynamics_amplitude <= 0.0:
        raise ValueError("dynamics_amplitude must be finite and positive")
    if not math.isfinite(lensing_to_dynamics_ratio):
        raise ValueError("lensing_to_dynamics_ratio must be finite")
    if field_sign not in (-1, 1):
        raise ValueError("field_sign must be -1 or 1")
    source_d = field_sign * math.sqrt(0.5 * dynamics_amplitude)
    beta = -2.0 * lensing_to_dynamics_ratio * source_d
    alpha = source_d + beta
    return reciprocal_dust_couplings(alpha, beta)


def point_mass_yukawa_acceleration_m_s2(
    radius_m: np.ndarray | float,
    mass_kg: float,
    range_m: float,
) -> np.ndarray:
    """Inward acceleration magnitude of a unit-amplitude attractive Yukawa field."""
    radius = np.asarray(radius_m, dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_m must be finite and positive")
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    if not math.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("range_m must be finite and positive")
    x = radius / range_m
    return G_SI * mass_kg * (1.0 + x) * np.exp(-x) / np.square(radius)


def point_mass_observable_accelerations_m_s2(
    radius_m: np.ndarray | float,
    mass_kg: float,
    range_m: float,
    *,
    dynamics_amplitude: float,
    lensing_to_dynamics_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return baryonic, dynamical, and Weyl-equivalent acceleration magnitudes."""
    if not math.isfinite(dynamics_amplitude) or dynamics_amplitude < 0.0:
        raise ValueError("dynamics_amplitude must be finite and nonnegative")
    if not math.isfinite(lensing_to_dynamics_ratio):
        raise ValueError("lensing_to_dynamics_ratio must be finite")
    radius = np.asarray(radius_m, dtype=np.float64)
    baryonic = G_SI * mass_kg / np.square(radius)
    unit = point_mass_yukawa_acceleration_m_s2(radius, mass_kg, range_m)
    dynamics = baryonic + dynamics_amplitude * unit
    lensing = baryonic + lensing_to_dynamics_ratio * dynamics_amplitude * unit
    return baryonic, dynamics, lensing


def point_mass_circular_speed_log_slope(
    radius_over_range: np.ndarray | float,
    dynamics_amplitude: float,
) -> np.ndarray:
    """Return d ln(v_c)/d ln(r) for Newton plus an attractive Yukawa scalar."""
    x = np.asarray(radius_over_range, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("radius_over_range must be finite and positive")
    if not math.isfinite(dynamics_amplitude) or dynamics_amplitude < 0.0:
        raise ValueError("dynamics_amplitude must be finite and nonnegative")
    extra_shape = (1.0 + x) * np.exp(-x)
    enhancement = 1.0 + dynamics_amplitude * extra_shape
    enhancement_slope = (
        -dynamics_amplitude * np.square(x) * np.exp(-x) / enhancement
    )
    return -0.5 + 0.5 * enhancement_slope


def positive_spectral_circular_speed_log_slope(
    radius_m: np.ndarray | float,
    ranges_m: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Slope for a nonnegative superposition of attractive Yukawa exchanges.

    Positive spectral weight is the weak-field signature expected when the
    exchanged scalar modes have positive norm.  Every term makes the force
    enhancement decrease with radius, so this function is also an executable
    version of the linear-response no-flat-curve theorem used by the project.
    """
    radius = np.asarray(radius_m, dtype=np.float64)
    ranges = np.asarray(ranges_m, dtype=np.float64)
    weights = np.asarray(amplitudes, dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_m must be finite and positive")
    if ranges.ndim != 1 or weights.ndim != 1 or ranges.shape != weights.shape:
        raise ValueError("ranges_m and amplitudes must be matching one-dimensional arrays")
    if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
        raise ValueError("ranges_m must be finite and positive")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("amplitudes must be finite and nonnegative")
    x = np.expand_dims(radius, axis=-1) / ranges
    enhancement = 1.0 + np.sum(weights * (1.0 + x) * np.exp(-x), axis=-1)
    derivative = -np.sum(weights * np.square(x) * np.exp(-x), axis=-1)
    return -0.5 + 0.5 * derivative / enhancement


def fractional_linear_point_source_scaling(operator_power: float) -> dict[str, float | bool]:
    """Return point-source scaling for ``(-nabla^2)^p Phi proportional rho`` in 3D."""
    if not math.isfinite(operator_power) or operator_power <= 0.0:
        raise ValueError("operator_power must be finite and positive")
    return {
        "operator_power": operator_power,
        "potential_radial_exponent": 2.0 * operator_power - 3.0,
        "acceleration_radial_exponent": 2.0 * operator_power - 4.0,
        "circular_speed_squared_radial_exponent": 2.0 * operator_power - 3.0,
        "flat_rotation_curve": math.isclose(operator_power, 1.5),
        "circular_speed_fourth_power_mass_exponent": 2.0,
    }


def fit_effective_yukawa_from_extras(
    radius_m: np.ndarray,
    mass_kg: float,
    dynamics_extra_m_s2: np.ndarray,
    lensing_extra_m_s2: np.ndarray,
    *,
    initial_dynamics_amplitude: float = 1.0,
    initial_range_m: float | None = None,
    initial_lensing_ratio: float = 1.0,
) -> YukawaRecovery:
    """Recover the three identifiable effective parameters from ideal joint data."""
    radius = np.asarray(radius_m, dtype=np.float64)
    dynamics = np.asarray(dynamics_extra_m_s2, dtype=np.float64)
    lensing = np.asarray(lensing_extra_m_s2, dtype=np.float64)
    if radius.ndim != 1 or dynamics.shape != radius.shape or lensing.shape != radius.shape:
        raise ValueError("radius and extra-acceleration arrays must be matching vectors")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_m must be finite and positive")
    if np.any(~np.isfinite(dynamics)) or np.any(dynamics <= 0.0):
        raise ValueError("dynamics extras must be finite and positive")
    if np.any(~np.isfinite(lensing)) or np.any(lensing <= 0.0):
        raise ValueError("lensing extras must be finite and positive")
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    if initial_range_m is None:
        initial_range_m = float(np.sqrt(radius.min() * radius.max()))
    if (
        initial_dynamics_amplitude <= 0.0
        or initial_range_m <= 0.0
        or initial_lensing_ratio <= 0.0
    ):
        raise ValueError("initial effective parameters must be positive")

    def residual(vector: np.ndarray) -> np.ndarray:
        amplitude, range_value, ratio = np.exp(vector)
        unit = point_mass_yukawa_acceleration_m_s2(radius, mass_kg, range_value)
        return np.concatenate(
            [
                np.log(amplitude * unit / dynamics),
                np.log(ratio * amplitude * unit / lensing),
            ]
        )

    lower_range = float(radius.min() * 1.0e-4)
    upper_range = float(radius.max() * 1.0e4)
    result = least_squares(
        residual,
        np.log(
            [
                initial_dynamics_amplitude,
                initial_range_m,
                initial_lensing_ratio,
            ]
        ),
        bounds=(
            np.log([1.0e-12, lower_range, 1.0e-12]),
            np.log([1.0e12, upper_range, 1.0e12]),
        ),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=20_000,
    )
    amplitude, range_value, ratio = np.exp(result.x)
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])
    residuals = residual(result.x)
    return YukawaRecovery(
        dynamics_amplitude=float(amplitude),
        range_m=float(range_value),
        lensing_to_dynamics_ratio=float(ratio),
        cost=float(2.0 * result.cost),
        maximum_absolute_log_residual=float(np.max(np.abs(residuals))),
        jacobian_condition_number=condition,
        success=bool(result.success),
    )
