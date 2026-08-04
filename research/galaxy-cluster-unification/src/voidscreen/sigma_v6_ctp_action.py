"""Minimal localization audits for the Sigma v6D closed-time-path action.

The functions here deliberately test the smallest variational localization of
inverse wave operators.  They do not model galaxy data.  A constraint pair
``A(Box U-J)`` yields the mixed kinetic block carried by ``A`` and ``U`` after
integration by parts; the same construction applies component by component to
the spatial symmetric trace-free tensor pair ``B_ij`` and ``Q_ij``.
"""

from __future__ import annotations

import numpy as np


def constraint_pair_kinetic_hessian(
    component_counts: tuple[int, ...] | list[int], mixing: float = 1.0
) -> np.ndarray:
    """Return the kinetic Hessian for multiplier/response constraint pairs."""

    counts = tuple(int(value) for value in component_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("component counts must be positive")
    if not np.isfinite(mixing) or mixing == 0.0:
        raise ValueError("mixing must be finite and nonzero")
    component_total = sum(counts)
    hessian = np.zeros((2 * component_total, 2 * component_total), dtype=float)
    for index in range(component_total):
        hessian[2 * index, 2 * index + 1] = mixing
        hessian[2 * index + 1, 2 * index] = mixing
    return hessian


def kinetic_signature(matrix: np.ndarray | list[list[float]], tolerance: float = 1.0e-12) -> dict:
    """Count positive, negative, and null eigenvalues of a symmetric Hessian."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)) or not np.allclose(values, values.T):
        raise ValueError("matrix must be finite and symmetric")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    eigenvalues = np.linalg.eigvalsh(values)
    return {
        "positive": int(np.sum(eigenvalues > tolerance)),
        "negative": int(np.sum(eigenvalues < -tolerance)),
        "null": int(np.sum(np.abs(eigenvalues) <= tolerance)),
        "rank": int(np.sum(np.abs(eigenvalues) > tolerance)),
        "eigenvalues": eigenvalues,
    }


def localized_initial_data_count(component_counts: tuple[int, ...] | list[int]) -> dict[str, int]:
    """Count configuration variables and second-order Cauchy data in the localization."""

    counts = tuple(int(value) for value in component_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("component counts must be positive")
    response_components = sum(counts)
    configurations = 2 * response_components
    return {
        "desired_retarded_response_components": response_components,
        "localized_configuration_components": configurations,
        "localized_second_order_initial_data": 2 * configurations,
        "extra_multiplier_configuration_components": response_components,
    }


def retarded_impulse_response(
    time: np.ndarray | float,
    impulse_time: float,
    frequency: float,
) -> np.ndarray:
    """Return theta(t-ti) sin[omega(t-ti)]/omega."""

    values = np.asarray(time, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("time must be finite")
    if not np.isfinite(impulse_time):
        raise ValueError("impulse time must be finite")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be finite and positive")
    lag = values - impulse_time
    return np.where(lag >= 0.0, np.sin(frequency * lag) / frequency, 0.0)


def advanced_impulse_response(
    time: np.ndarray | float,
    impulse_time: float,
    frequency: float,
) -> np.ndarray:
    """Return theta(ti-t) sin[omega(ti-t)]/omega."""

    values = np.asarray(time, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("time must be finite")
    if not np.isfinite(impulse_time):
        raise ValueError("impulse time must be finite")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be finite and positive")
    lag = impulse_time - values
    return np.where(lag >= 0.0, np.sin(frequency * lag) / frequency, 0.0)
