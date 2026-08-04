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


def v6a_perturbation_action(
    amplitude: np.ndarray | float,
    gradient_coefficient: float,
    hessian_coefficient: float,
    orientation_coupling: float,
) -> np.ndarray:
    """Evaluate the frozen v6A envelope along a one-parameter perturbation.

    For a metric perturbation ``h = amplitude * h0``, the gradient and squared
    Hessian invariants begin as ``X=A*amplitude**2`` and
    ``Z=B*amplitude**2``.  The frozen ``sqrt(Z)`` term therefore exposes its
    absolute-value cusp directly.
    """

    values = np.asarray(amplitude, dtype=float)
    coefficients = (gradient_coefficient, hessian_coefficient, orientation_coupling)
    if np.any(~np.isfinite(values)):
        raise ValueError("amplitude must be finite")
    if any(not np.isfinite(value) or value < 0.0 for value in coefficients):
        raise ValueError("perturbation coefficients must be finite and nonnegative")
    chi = (
        gradient_coefficient * values**2
        + orientation_coupling * np.sqrt(hessian_coefficient * values**2)
    )
    return metric_memory_activation(chi)


def hessian_power_regularities(power: float) -> dict[str, float | bool | str]:
    """Classify ``Z**power`` when ``Z`` starts at second perturbative order."""

    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be finite and positive")
    perturbation_order = 2.0 * power
    first_variation_exists = perturbation_order > 1.0
    finite_quadratic_variation = perturbation_order >= 2.0
    if perturbation_order < 2.0:
        spectrum_role = "singular_or_undefined_quadratic_variation"
    elif perturbation_order == 2.0:
        spectrum_role = "changes_quadratic_operator"
    else:
        spectrum_role = "nonlinear_only_about_zero_background"
    return {
        "power": float(power),
        "perturbation_order": perturbation_order,
        "first_variation_exists": first_variation_exists,
        "finite_quadratic_variation": finite_quadratic_variation,
        "spectrum_role": spectrum_role,
    }


def bounded_tensor_coherence(
    tensor_invariant: np.ndarray | float, potential_scale: float
) -> np.ndarray:
    """Return the analytic bounded v6B orientation coherence ``Z/(Z+phi^2)``."""

    values = np.asarray(tensor_invariant, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("tensor invariant must be finite and nonnegative")
    if not np.isfinite(potential_scale) or potential_scale <= 0.0:
        raise ValueError("potential scale must be finite and positive")
    return values / (values + potential_scale**2)


def v6b_metric_memory_chi(
    gradient_ratio_squared: np.ndarray | float,
    tensor_invariant: np.ndarray | float,
    orientation_coupling: float,
    potential_scale: float,
) -> np.ndarray:
    """Return the differentiable v6B scalar-times-orientation invariant."""

    gradient = np.asarray(gradient_ratio_squared, dtype=float)
    if np.any(~np.isfinite(gradient)) or np.any(gradient < 0.0):
        raise ValueError("gradient invariant must be finite and nonnegative")
    if not np.isfinite(orientation_coupling) or orientation_coupling < 0.0:
        raise ValueError("orientation coupling must be finite and nonnegative")
    coherence = bounded_tensor_coherence(tensor_invariant, potential_scale)
    return gradient * (1.0 + orientation_coupling * coherence)


def static_trace_free_projector(direction: np.ndarray | list[float]) -> np.ndarray:
    """Return ``n_i n_j-delta_ij/3`` for the twice-retarded static memory."""

    values = np.asarray(direction, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("direction must be a finite three-vector")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    unit = values / norm
    return np.outer(unit, unit) - np.eye(3) / 3.0


def repeated_massless_memory_step_response(
    time: np.ndarray | float, wavenumber: float, source_amplitude: float = 1.0
) -> np.ndarray:
    """Response of two identical retarded wave inverses to a switched-on mode.

    The first response is ``S/k^2 * (1-cos(k t))``.  Feeding it through the
    same operator produces the resonant ``t*sin(k t)`` term returned here.
    """

    values = np.asarray(time, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("time must be finite and nonnegative")
    if not np.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be finite and positive")
    if not np.isfinite(source_amplitude):
        raise ValueError("source amplitude must be finite")
    phase = wavenumber * values
    return source_amplitude * (
        (1.0 - np.cos(phase)) / wavenumber**2
        - values * np.sin(phase) / (2.0 * wavenumber)
    )


def detuned_massive_memory_step_response(
    time: np.ndarray | float,
    wavenumber: float,
    memory_mass: float,
    source_amplitude: float = 1.0,
) -> np.ndarray:
    """Response of a massive retarded tensor memory driven by the first mode."""

    values = np.asarray(time, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("time must be finite and nonnegative")
    if not np.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("wavenumber must be finite and positive")
    if not np.isfinite(memory_mass) or memory_mass <= 0.0:
        raise ValueError("memory mass must be finite and positive")
    if not np.isfinite(source_amplitude):
        raise ValueError("source amplitude must be finite")
    omega = np.sqrt(wavenumber**2 + memory_mass**2)
    constant_response = (1.0 - np.cos(omega * values)) / omega**2
    oscillatory_response = (
        np.cos(wavenumber * values) - np.cos(omega * values)
    ) / memory_mass**2
    return source_amplitude * (constant_response - oscillatory_response)


def detuned_static_tensor_transfer(
    wavenumber: np.ndarray | float, memory_mass: float
) -> np.ndarray:
    """Return the static v6C transfer k^2/(k^2+m^2)."""

    values = np.asarray(wavenumber, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("wavenumber must be finite and nonnegative")
    if not np.isfinite(memory_mass) or memory_mass <= 0.0:
        raise ValueError("memory mass must be finite and positive")
    return values**2 / (values**2 + memory_mass**2)


def v6c_total_constitutive_coefficient(
    gradient_ratio_squared: np.ndarray | float, orientation_strength: float
) -> np.ndarray:
    """Return d[X-f((1+q)X)]/dX for the retired v6C placement."""

    values = np.asarray(gradient_ratio_squared, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("gradient invariant must be finite and nonnegative")
    if not np.isfinite(orientation_strength) or orientation_strength < 0.0:
        raise ValueError("orientation strength must be finite and nonnegative")
    factor = 1.0 + orientation_strength
    root = np.sqrt(factor * values)
    correction_derivative = factor * np.exp(-root) * (1.0 - 0.5 * root)
    return 1.0 - correction_derivative


def v6d_cubic_orientation_correction(
    gradient_ratio_squared: np.ndarray | float, orientation_strength: float
) -> np.ndarray:
    """Return exp(-sqrt(X)) * [X + q X**(3/2)] for the v6D placement."""

    values = np.asarray(gradient_ratio_squared, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("gradient invariant must be finite and nonnegative")
    if not np.isfinite(orientation_strength) or orientation_strength < 0.0:
        raise ValueError("orientation strength must be finite and nonnegative")
    root = np.sqrt(values)
    return np.exp(-root) * (values + orientation_strength * values * root)


def v6d_total_constitutive_coefficient(
    gradient_ratio_squared: np.ndarray | float, orientation_strength: float
) -> np.ndarray:
    """Return d[X-f_D(X,q)]/dX for the v6D scalar weak-field surrogate."""

    values = np.asarray(gradient_ratio_squared, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("gradient invariant must be finite and nonnegative")
    if not np.isfinite(orientation_strength) or not 0.0 <= orientation_strength <= 1.0:
        raise ValueError("orientation strength must lie between zero and one")
    root = np.sqrt(values)
    bracket = (
        1.0
        + 0.5 * (3.0 * orientation_strength - 1.0) * root
        - 0.5 * orientation_strength * values
    )
    return 1.0 - np.exp(-root) * bracket


def v6d_parallel_ellipticity_coefficient(
    gradient_ratio_squared: np.ndarray | float, orientation_strength: float
) -> np.ndarray:
    """Return mu+2X*mu_X, the radial/parallel ellipticity coefficient."""

    values = np.asarray(gradient_ratio_squared, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("gradient invariant must be finite and positive")
    if not np.isfinite(orientation_strength) or not 0.0 <= orientation_strength <= 1.0:
        raise ValueError("orientation strength must lie between zero and one")
    root = np.sqrt(values)
    bracket = (
        1.0
        + 0.5 * (3.0 * orientation_strength - 1.0) * root
        - 0.5 * orientation_strength * values
    )
    bracket_derivative = (
        0.5 * (3.0 * orientation_strength - 1.0)
        - orientation_strength * root
    )
    mu = 1.0 - np.exp(-root) * bracket
    derivative_by_root = np.exp(-root) * (bracket - bracket_derivative)
    return mu + root * derivative_by_root


def v6d_deep_acceleration_enhancement(orientation_strength: float) -> float:
    """Return the deep-field acceleration ratio relative to q=0 at fixed source."""

    if not np.isfinite(orientation_strength) or not 0.0 <= orientation_strength < 1.0:
        raise ValueError("orientation strength must lie in [0, 1)")
    return float(1.0 / np.sqrt(1.0 - orientation_strength))
