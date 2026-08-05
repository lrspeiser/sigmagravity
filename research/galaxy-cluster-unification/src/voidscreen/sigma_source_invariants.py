"""Target-independent baryonic source-state invariants for Sigma V19BJ.

These functions commission the Euclidean, projected-map algebra only.  They do
not turn a two-dimensional observable into a four-dimensional action source.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_field(value: ArrayLike, *, name: str) -> FloatArray:
    array = _float_array(value, name=name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def component_overlap(gas_density: ArrayLike, stellar_density: ArrayLike) -> FloatArray:
    """Return ``4 f_g (1-f_g)`` with zero assigned where both densities vanish."""

    gas = _float_array(gas_density, name="gas_density")
    stars = _float_array(stellar_density, name="stellar_density")
    if gas.shape != stars.shape:
        raise ValueError("gas_density and stellar_density must have the same shape")
    if np.any(gas < 0.0) or np.any(stars < 0.0):
        raise ValueError("component densities must be non-negative")
    total = gas + stars
    fraction = np.divide(gas, total, out=np.zeros_like(total), where=total > 0.0)
    return 4.0 * fraction * (1.0 - fraction)


def relative_current(
    gas_velocity: ArrayLike,
    stellar_velocity: ArrayLike,
    sound_speed: ArrayLike,
    stellar_dispersion: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return the projected relative-current vector and dimensionless norm.

    The final velocity axis can contain two or three spatial components.  The
    norm is divided by ``c_s^2 + sigma_star^2`` and is invariant under a common
    velocity boost.
    """

    gas = _float_array(gas_velocity, name="gas_velocity")
    stars = _float_array(stellar_velocity, name="stellar_velocity")
    if gas.shape != stars.shape or gas.ndim < 1:
        raise ValueError("gas_velocity and stellar_velocity must share a vector shape")
    sound = _float_array(sound_speed, name="sound_speed")
    dispersion = _float_array(stellar_dispersion, name="stellar_dispersion")
    try:
        scale_squared = np.broadcast_to(sound**2 + dispersion**2, gas.shape[:-1])
    except ValueError as exc:
        raise ValueError("speed scales must broadcast over the velocity field") from exc
    if np.any(scale_squared <= 0.0):
        raise ValueError("c_s^2 + sigma_star^2 must be positive")
    vector = gas - stars
    norm = np.sum(vector**2, axis=-1) / scale_squared
    return vector, norm


def anisotropic_stress(
    spatial_stress: ArrayLike, enthalpy_density: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return the symmetric trace-free stress and its enthalpy-normalized norm."""

    stress = _float_array(spatial_stress, name="spatial_stress")
    if stress.ndim < 2 or stress.shape[-1] != stress.shape[-2]:
        raise ValueError("spatial_stress must end in a square matrix")
    dimension = stress.shape[-1]
    symmetric = 0.5 * (stress + np.swapaxes(stress, -1, -2))
    trace = np.trace(symmetric, axis1=-2, axis2=-1)
    identity = np.eye(dimension, dtype=float)
    trace_free = symmetric - trace[..., None, None] * identity / float(dimension)
    enthalpy = _positive_field(enthalpy_density, name="enthalpy_density")
    try:
        enthalpy = np.broadcast_to(enthalpy, stress.shape[:-2])
    except ValueError as exc:
        raise ValueError("enthalpy_density must broadcast over the stress field") from exc
    normalized_norm = np.sum(trace_free**2, axis=(-2, -1)) / enthalpy**2
    return trace_free, normalized_norm


def _two_dimensional_gradients(
    field: ArrayLike, spacing: Sequence[float], *, name: str
) -> FloatArray:
    array = _positive_field(field, name=name)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional map")
    if len(spacing) != 2 or any(step <= 0.0 for step in spacing):
        raise ValueError("spacing must contain two positive values")
    gradients = np.gradient(np.log(array), *spacing, edge_order=2)
    return np.stack(gradients, axis=-1)


def thermodynamic_gradient_stress(
    gas_density: ArrayLike,
    gas_entropy: ArrayLike,
    *,
    spacing: Sequence[float] = (1.0, 1.0),
) -> FloatArray:
    """Return the projected STF product of log-density and log-entropy gradients."""

    density_gradient = _two_dimensional_gradients(
        gas_density, spacing, name="gas_density"
    )
    entropy_gradient = _two_dimensional_gradients(
        gas_entropy, spacing, name="gas_entropy"
    )
    outer = 0.5 * (
        np.einsum("...i,...j->...ij", density_gradient, entropy_gradient)
        + np.einsum("...i,...j->...ij", entropy_gradient, density_gradient)
    )
    trace = np.trace(outer, axis1=-2, axis2=-1)
    return outer - 0.5 * trace[..., None, None] * np.eye(2, dtype=float)


def projected_baroclinicity(
    gas_density: ArrayLike,
    gas_pressure: ArrayLike,
    *,
    spacing: Sequence[float] = (1.0, 1.0),
) -> tuple[FloatArray, FloatArray]:
    """Return signed and squared normalized 2D pressure-density misalignment.

    The signed value is the line-of-sight component of the cross product of
    the two normalized projected gradients.  Pixels with a zero gradient are
    assigned zero and must be excluded by the later significance mask.
    """

    density_gradient = _two_dimensional_gradients(
        gas_density, spacing, name="gas_density"
    )
    pressure_gradient = _two_dimensional_gradients(
        gas_pressure, spacing, name="gas_pressure"
    )
    cross = (
        density_gradient[..., 0] * pressure_gradient[..., 1]
        - density_gradient[..., 1] * pressure_gradient[..., 0]
    )
    denominator = np.linalg.norm(density_gradient, axis=-1) * np.linalg.norm(
        pressure_gradient, axis=-1
    )
    signed = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    signed = np.clip(signed, -1.0, 1.0)
    return signed, signed**2


def axial_orientation_deg(
    tensor_field: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """Return the principal axial orientation of a 2D symmetric tensor field."""

    tensor = _float_array(tensor_field, name="tensor_field")
    if tensor.shape[-2:] != (2, 2):
        raise ValueError("tensor_field must end in a 2 by 2 matrix")
    symmetric = 0.5 * (tensor + np.swapaxes(tensor, -1, -2))
    leading_shape = tensor.shape[:-2]
    if weights is None:
        weight = np.ones(leading_shape, dtype=float)
    else:
        weight = _float_array(weights, name="weights")
        try:
            weight = np.broadcast_to(weight, leading_shape)
        except ValueError as exc:
            raise ValueError("weights must broadcast over the tensor field") from exc
        if np.any(weight < 0.0):
            raise ValueError("weights must be non-negative")
    total_weight = float(np.sum(weight))
    if total_weight <= 0.0:
        raise ValueError("at least one tensor weight must be positive")
    mean = np.sum(symmetric * weight[..., None, None], axis=tuple(range(len(leading_shape))))
    mean /= total_weight
    numerator = 2.0 * mean[0, 1]
    denominator = mean[0, 0] - mean[1, 1]
    if abs(numerator) + abs(denominator) <= np.finfo(float).eps:
        raise ValueError("the weighted tensor is isotropic and has no axial orientation")
    return float(np.degrees(0.5 * np.arctan2(numerator, denominator)) % 180.0)
