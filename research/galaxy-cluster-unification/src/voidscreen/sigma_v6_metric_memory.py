"""Theory-only utilities for the Sigma v6A causal metric-memory audit.

These routines do not implement an astronomical gravity fit.  They encode the
minimal mathematical distinctions needed before a retarded nonlocal action is
allowed to see galaxy or cluster data: retarded versus time-symmetric response,
zero response for zero source and fixed initial data, and a rotationally
covariant nonlinear response to a trace-free spatial Hessian.
"""

from __future__ import annotations

import numpy as np


def _finite_vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def retarded_convolution(
    source: np.ndarray | list[float], kernel_by_nonnegative_lag: np.ndarray | list[float]
) -> np.ndarray:
    """Return a discrete response with support only in the source's causal future."""

    source_values = _finite_vector(source, "source")
    kernel = _finite_vector(kernel_by_nonnegative_lag, "kernel")
    return np.convolve(source_values, kernel, mode="full")[: source_values.size]


def time_symmetric_convolution(
    source: np.ndarray | list[float], kernel_by_absolute_lag: np.ndarray | list[float]
) -> np.ndarray:
    """Return the symmetric kernel produced by a traditional bilinear variation.

    A kernel specified only for nonnegative lag is mirrored around zero.  This
    deliberately exposes the advanced response that a closed-time-path or
    equivalent causal prescription must remove.
    """

    source_values = _finite_vector(source, "source")
    kernel = _finite_vector(kernel_by_absolute_lag, "kernel")
    indices = np.arange(source_values.size)
    lag = np.abs(indices[:, None] - indices[None, :])
    weights = np.where(lag < kernel.size, kernel[np.minimum(lag, kernel.size - 1)], 0.0)
    return weights @ source_values


def trace_free_symmetric(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return the symmetric trace-free part of a 3x3 spatial Hessian."""

    values = np.asarray(matrix, dtype=float)
    if values.shape != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be a finite 3x3 array")
    symmetric = 0.5 * (values + values.T)
    return symmetric - np.eye(3) * np.trace(symmetric) / 3.0


def hessian_response(
    matrix: np.ndarray | list[list[float]], saturation: float = 1.0
) -> np.ndarray:
    """A bounded covariant toy response used only to audit superposition order."""

    if not np.isfinite(saturation) or saturation <= 0.0:
        raise ValueError("saturation must be finite and positive")
    tensor = trace_free_symmetric(matrix)
    norm_squared = float(np.sum(tensor * tensor))
    return tensor / np.sqrt(1.0 + saturation * norm_squared)


def nonlinear_superposition_residual(
    first: np.ndarray,
    second: np.ndarray,
    saturation: float = 1.0,
) -> float:
    """Measure N(H1+H2) != N(H1)+N(H2), normalized to the separate response."""

    together = hessian_response(np.asarray(first) + np.asarray(second), saturation)
    separate = hessian_response(first, saturation) + hessian_response(second, saturation)
    scale = max(float(np.linalg.norm(separate)), np.finfo(float).eps)
    return float(np.linalg.norm(together - separate) / scale)


def rotation_covariance_residual(
    matrix: np.ndarray, rotation: np.ndarray, saturation: float = 1.0
) -> float:
    """Return ||N(RHR^T)-R N(H) R^T|| normalized to ||N(H)||."""

    rotation_values = np.asarray(rotation, dtype=float)
    if rotation_values.shape != (3, 3) or not np.all(np.isfinite(rotation_values)):
        raise ValueError("rotation must be a finite 3x3 array")
    identity_residual = np.linalg.norm(rotation_values @ rotation_values.T - np.eye(3))
    determinant_residual = abs(np.linalg.det(rotation_values) - 1.0)
    if identity_residual > 1.0e-10 or determinant_residual > 1.0e-10:
        raise ValueError("rotation must be a proper orthogonal matrix")
    tensor = trace_free_symmetric(matrix)
    rotated = rotation_values @ tensor @ rotation_values.T
    left = hessian_response(rotated, saturation)
    right = rotation_values @ hessian_response(tensor, saturation) @ rotation_values.T
    scale = max(float(np.linalg.norm(right)), np.finfo(float).eps)
    return float(np.linalg.norm(left - right) / scale)


def metric_memory_activation(chi: np.ndarray | float) -> np.ndarray:
    """Return chi*exp(-sqrt(chi)), the frozen v6A action-envelope activation.

    The low-chi expansion starts with chi - chi**(3/2), permitting cancellation
    of the Einstein-Hilbert quadratic weak-field term and a MOND-order cubic
    residue.  The high-chi correction is exponentially suppressed.
    """

    values = np.asarray(chi, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("chi must be finite and nonnegative")
    return values * np.exp(-np.sqrt(values))


def metric_memory_chi(
    gradient_ratio_squared: np.ndarray | float,
    trace_free_hessian_ratio_squared: np.ndarray | float,
    orientation_coupling: float,
) -> np.ndarray:
    """Combine scalar amplitude and Hessian orientation without object labels."""

    gradient = np.asarray(gradient_ratio_squared, dtype=float)
    hessian = np.asarray(trace_free_hessian_ratio_squared, dtype=float)
    if np.any(~np.isfinite(gradient)) or np.any(gradient < 0.0):
        raise ValueError("gradient invariant must be finite and nonnegative")
    if np.any(~np.isfinite(hessian)) or np.any(hessian < 0.0):
        raise ValueError("hessian invariant must be finite and nonnegative")
    if not np.isfinite(orientation_coupling) or orientation_coupling < 0.0:
        raise ValueError("orientation coupling must be finite and nonnegative")
    return gradient + orientation_coupling * np.sqrt(hessian)
