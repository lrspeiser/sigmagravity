"""Target-independent baryonic source-state invariants for Sigma V19BJ/V19BL.

These functions commission the Euclidean, projected-map algebra only. They do
not turn a two-dimensional observable into a four-dimensional action source.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any

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

    The final velocity axis can contain two or three spatial components. The
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
    the two normalized projected gradients. Pixels with a zero gradient are
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
    signed = np.divide(
        cross, denominator, out=np.zeros_like(cross), where=denominator > 0
    )
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
    mean = np.sum(
        symmetric * weight[..., None, None], axis=tuple(range(len(leading_shape)))
    )
    mean /= total_weight
    numerator = 2.0 * mean[0, 1]
    denominator = mean[0, 0] - mean[1, 1]
    if abs(numerator) + abs(denominator) <= np.finfo(float).eps:
        raise ValueError("the weighted tensor is isotropic and has no axial orientation")
    return float(np.degrees(0.5 * np.arctan2(numerator, denominator)) % 180.0)


def _positive_spacing(value: float) -> float:
    spacing = float(value)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    return spacing


def _as_log_field(values: ArrayLike, name: str) -> FloatArray:
    field = np.asarray(values, dtype=float)
    if field.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    finite = np.isfinite(field)
    if np.any(field[finite] <= 0.0):
        raise ValueError(f"finite {name} values must be positive")
    output = np.full(field.shape, np.nan, dtype=float)
    output[finite] = np.log(field[finite])
    return output


def central_gradient(values: ArrayLike, spacing: float) -> tuple[FloatArray, FloatArray]:
    """Return east and north central derivatives, preserving invalid boundaries."""

    field = np.asarray(values, dtype=float)
    if field.ndim < 2 or field.shape[-2] < 3 or field.shape[-1] < 3:
        raise ValueError("field must end in two axes of length at least three")
    step = _positive_spacing(spacing)
    east = np.full(field.shape, np.nan, dtype=float)
    north = np.full(field.shape, np.nan, dtype=float)
    center = field[..., 1:-1, 1:-1]
    west = field[..., 1:-1, :-2]
    east_neighbor = field[..., 1:-1, 2:]
    south = field[..., :-2, 1:-1]
    north_neighbor = field[..., 2:, 1:-1]
    valid = (
        np.isfinite(center)
        & np.isfinite(west)
        & np.isfinite(east_neighbor)
        & np.isfinite(south)
        & np.isfinite(north_neighbor)
    )
    east_center = np.where(valid, (east_neighbor - west) / (2.0 * step), np.nan)
    north_center = np.where(valid, (north_neighbor - south) / (2.0 * step), np.nan)
    east[..., 1:-1, 1:-1] = east_center
    north[..., 1:-1, 1:-1] = north_center
    return east, north


def central_hessian(
    values: ArrayLike, spacing: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return east-east, north-north, and east-north central derivatives."""

    field = np.asarray(values, dtype=float)
    if field.ndim < 2 or field.shape[-2] < 3 or field.shape[-1] < 3:
        raise ValueError("field must end in two axes of length at least three")
    step = _positive_spacing(spacing)
    center = field[..., 1:-1, 1:-1]
    west = field[..., 1:-1, :-2]
    east = field[..., 1:-1, 2:]
    south = field[..., :-2, 1:-1]
    north = field[..., 2:, 1:-1]
    southwest = field[..., :-2, :-2]
    southeast = field[..., :-2, 2:]
    northwest = field[..., 2:, :-2]
    northeast = field[..., 2:, 2:]
    valid = np.logical_and.reduce(
        [
            np.isfinite(center),
            np.isfinite(west),
            np.isfinite(east),
            np.isfinite(south),
            np.isfinite(north),
            np.isfinite(southwest),
            np.isfinite(southeast),
            np.isfinite(northwest),
            np.isfinite(northeast),
        ]
    )
    factor = step * step
    d_ee_center = np.where(valid, (east - 2.0 * center + west) / factor, np.nan)
    d_nn_center = np.where(valid, (north - 2.0 * center + south) / factor, np.nan)
    d_en_center = np.where(
        valid,
        (northeast - northwest - southeast + southwest) / (4.0 * factor),
        np.nan,
    )
    outputs = [np.full(field.shape, np.nan, dtype=float) for _ in range(3)]
    for output, interior in zip(
        outputs, (d_ee_center, d_nn_center, d_en_center), strict=True
    ):
        output[..., 1:-1, 1:-1] = interior
    return outputs[0], outputs[1], outputs[2]


def projected_source_maps(
    electron_density: ArrayLike,
    entropy_proxy: ArrayLike,
    thermal_pressure: ArrayLike,
    gas_surface_density: ArrayLike,
    *,
    spacing_kpc: float,
    resolution_fwhm_kpc: float,
    log_gradient_floor: float = 1.0e-12,
) -> dict[str, FloatArray]:
    """Build I4, I5, and density-control maps from positive smoothed fields.

    The final two axes are north and east. Leading axes, when present, are
    independent posterior draws. I4 is the dimensionless projected STF tensor
    l^2 D_<i ln(n_e) D_j> ln(K). I5 is the squared sine of the projected angle
    between density and pressure gradients.
    """

    spacing = _positive_spacing(spacing_kpc)
    resolution = _positive_spacing(resolution_fwhm_kpc)
    floor = float(log_gradient_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("log_gradient_floor must be finite and positive")
    log_ne = _as_log_field(electron_density, "electron_density")
    log_entropy = _as_log_field(entropy_proxy, "entropy_proxy")
    log_pressure = _as_log_field(thermal_pressure, "thermal_pressure")
    log_surface = _as_log_field(gas_surface_density, "gas_surface_density")
    if not (
        log_ne.shape
        == log_entropy.shape
        == log_pressure.shape
        == log_surface.shape
    ):
        raise ValueError("all source fields must have identical shapes")

    ne_e, ne_n = central_gradient(log_ne, spacing)
    entropy_e, entropy_n = central_gradient(log_entropy, spacing)
    pressure_e, pressure_n = central_gradient(log_pressure, spacing)
    surface_e, surface_n = central_gradient(log_surface, spacing)
    surface_ee, surface_nn, surface_en = central_hessian(log_surface, spacing)
    length_squared = resolution * resolution

    q_plus = 0.5 * length_squared * (
        ne_e * entropy_e - ne_n * entropy_n
    )
    q_cross = 0.5 * length_squared * (
        ne_e * entropy_n + ne_n * entropy_e
    )
    q_amplitude = np.sqrt(2.0 * (q_plus * q_plus + q_cross * q_cross))

    cross = ne_e * pressure_n - ne_n * pressure_e
    denominator = (ne_e * ne_e + ne_n * ne_n) * (
        pressure_e * pressure_e + pressure_n * pressure_n
    )
    baroclinicity = np.full(denominator.shape, np.nan, dtype=float)
    valid_baro = np.isfinite(cross) & np.isfinite(denominator) & (denominator > 0.0)
    baroclinicity[valid_baro] = np.clip(
        cross[valid_baro] * cross[valid_baro] / denominator[valid_baro],
        0.0,
        1.0,
    )

    surface_gradient = resolution * np.sqrt(
        surface_e * surface_e + surface_n * surface_n
    )
    surface_trace = length_squared * (surface_ee + surface_nn)
    surface_anisotropy = length_squared * np.sqrt(
        (surface_ee - surface_nn) ** 2 + 4.0 * surface_en * surface_en
    )
    return {
        "electron_density_gradient_east_kpc_inv": ne_e,
        "electron_density_gradient_north_kpc_inv": ne_n,
        "entropy_gradient_east_kpc_inv": entropy_e,
        "entropy_gradient_north_kpc_inv": entropy_n,
        "pressure_gradient_east_kpc_inv": pressure_e,
        "pressure_gradient_north_kpc_inv": pressure_n,
        "i4_q_plus": q_plus,
        "i4_q_cross": q_cross,
        "i4_amplitude": q_amplitude,
        "i5_baroclinicity": baroclinicity,
        "control_log_gas_surface_density": log_surface,
        "control_log_surface_gradient": np.log(
            np.maximum(surface_gradient, floor)
        ),
        "control_surface_hessian_trace": surface_trace,
        "control_surface_hessian_anisotropy": surface_anisotropy,
    }


def region_means(
    values: ArrayLike,
    labels: ArrayLike,
    region_ids: ArrayLike,
    *,
    radial_mask: ArrayLike | None = None,
) -> FloatArray:
    """Average a map or draw stack over fixed labeled adaptive regions."""

    data = np.asarray(values, dtype=float)
    label_grid = np.asarray(labels, dtype=np.int64)
    identifiers = np.asarray(region_ids, dtype=np.int64)
    if data.shape[-2:] != label_grid.shape:
        raise ValueError("map and label-grid shapes differ")
    if identifiers.ndim != 1 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("region_ids must be a unique vector")
    if radial_mask is None:
        admitted = np.ones(label_grid.shape, dtype=bool)
    else:
        admitted = np.asarray(radial_mask, dtype=bool)
        if admitted.shape != label_grid.shape:
            raise ValueError("radial_mask shape differs from label grid")
    flat = data.reshape((-1, label_grid.size))
    labels_flat = label_grid.ravel()
    admitted_flat = admitted.ravel()
    result = np.full((flat.shape[0], identifiers.size), np.nan, dtype=float)
    for index, region_id in enumerate(identifiers):
        pixels = admitted_flat & (labels_flat == int(region_id))
        if not np.any(pixels):
            continue
        selected = flat[:, pixels]
        counts = np.count_nonzero(np.isfinite(selected), axis=1)
        totals = np.nansum(selected, axis=1)
        result[:, index] = np.divide(
            totals,
            counts,
            out=np.full(flat.shape[0], np.nan, dtype=float),
            where=counts > 0,
        )
    leading = data.shape[:-2]
    return result.reshape((*leading, identifiers.size))


def quadratic_design(predictors: ArrayLike) -> FloatArray:
    """Return the fixed intercept + five linear + five square + ten cross basis."""

    values = np.asarray(predictors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5 or not np.all(np.isfinite(values)):
        raise ValueError("predictors must be a finite N by 5 matrix")
    if values.shape[0] <= 21:
        raise ValueError("quadratic PRESS requires more regions than coefficients")
    deviations = values - np.mean(values, axis=0)
    scales = np.std(deviations, axis=0)
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("each density-control predictor must vary")
    standardized = deviations / scales
    columns = [np.ones(values.shape[0])]
    columns.extend(standardized[:, index] for index in range(5))
    columns.extend(standardized[:, index] ** 2 for index in range(5))
    columns.extend(
        standardized[:, left] * standardized[:, right]
        for left, right in combinations(range(5), 2)
    )
    design = np.column_stack(columns)
    if design.shape[1] != 21:
        raise AssertionError("fixed quadratic basis must contain 21 columns")
    return design


def analytic_press_unexplained_fraction(
    predictors: ArrayLike,
    response: ArrayLike,
    *,
    minimum_one_minus_leverage: float = 1.0e-6,
) -> dict[str, Any]:
    """Return rotation-independent analytic leave-one-region-out PRESS ratios."""

    design = quadratic_design(predictors)
    target = np.asarray(response, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    if (
        target.ndim != 2
        or target.shape[0] != design.shape[0]
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("response must be finite and share the predictor rows")
    leverage_floor = float(minimum_one_minus_leverage)
    if not math.isfinite(leverage_floor) or leverage_floor <= 0.0:
        raise ValueError("minimum_one_minus_leverage must be positive")
    gram_inverse = np.linalg.pinv(design.T @ design, rcond=1.0e-12)
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank != design.shape[1]:
        raise ValueError("density-control quadratic design is rank deficient")
    coefficients = gram_inverse @ design.T @ target
    fitted = design @ coefficients
    residual = target - fitted
    leverage = np.einsum("ij,jk,ik->i", design, gram_inverse, design)
    one_minus = 1.0 - leverage
    if np.any(one_minus <= leverage_floor):
        raise ValueError("density-control PRESS has an unstable leverage point")
    press_residual = residual / one_minus[:, None]
    centered = target - np.mean(target, axis=0)
    total = np.sum(centered * centered, axis=0)
    press = np.sum(press_residual * press_residual, axis=0)
    component_fraction = np.divide(
        press,
        total,
        out=np.zeros_like(press),
        where=total > np.finfo(float).tiny,
    )
    joint_total = float(np.sum(total))
    joint_fraction = (
        float(np.sum(press) / joint_total)
        if joint_total > np.finfo(float).tiny
        else 0.0
    )
    return {
        "component_unexplained_fractions": component_fraction,
        "joint_unexplained_fraction": joint_fraction,
        "maximum_leverage": float(np.max(leverage)),
        "design_rank": design_rank,
        "coefficient_count": int(design.shape[1]),
    }


def axial_angle_deg(q_plus: ArrayLike, q_cross: ArrayLike) -> FloatArray:
    plus = np.asarray(q_plus, dtype=float)
    cross = np.asarray(q_cross, dtype=float)
    if plus.shape != cross.shape:
        raise ValueError("tensor components must share a shape")
    return np.mod(np.degrees(0.5 * np.arctan2(cross, plus)), 180.0)


def axial_difference_deg(first: ArrayLike, second: ArrayLike) -> FloatArray:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    difference = np.mod(left - right + 90.0, 180.0) - 90.0
    return np.abs(difference)


def axial_interval_summary_deg(angles: ArrayLike) -> dict[str, float]:
    values = np.asarray(angles, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("angles must be a finite vector with at least two draws")
    doubled = np.radians(2.0 * values)
    resultant = np.mean(np.exp(1j * doubled))
    if abs(resultant) <= np.finfo(float).eps:
        return {
            "median_axis_deg": math.nan,
            "width_95_deg": 180.0,
            "resultant_length": 0.0,
        }
    center = 0.5 * math.degrees(math.atan2(resultant.imag, resultant.real))
    center = center % 180.0
    signed = np.mod(values - center + 90.0, 180.0) - 90.0
    low, median, high = np.percentile(signed, [2.5, 50.0, 97.5])
    return {
        "median_axis_deg": float((center + median) % 180.0),
        "width_95_deg": float(high - low),
        "resultant_length": float(abs(resultant)),
    }


def robust_detection_sigma(values: ArrayLike) -> float:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("samples must be a finite vector with at least three draws")
    q16, median, q84 = np.percentile(samples, [16.0, 50.0, 84.0])
    scale = 0.5 * (q84 - q16)
    if scale <= np.finfo(float).tiny:
        return math.inf if median > 0.0 else 0.0
    return float(median / scale)


def symmetric_fractional_change(first: ArrayLike, second: ArrayLike) -> FloatArray:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    denominator = np.abs(left) + np.abs(right)
    return np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros(np.broadcast_shapes(left.shape, right.shape), dtype=float),
        where=denominator > np.finfo(float).tiny,
    )


def gradient_detection_sigma(east: ArrayLike, north: ArrayLike) -> float:
    """Mahalanobis significance of a two-component posterior gradient mean."""

    east_draws = np.asarray(east, dtype=float)
    north_draws = np.asarray(north, dtype=float)
    if (
        east_draws.ndim != 1
        or north_draws.shape != east_draws.shape
        or east_draws.size < 3
        or not np.all(np.isfinite(east_draws))
        or not np.all(np.isfinite(north_draws))
    ):
        raise ValueError("gradient components must be matching finite draw vectors")
    samples = np.column_stack([east_draws, north_draws])
    mean = np.mean(samples, axis=0)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    if np.max(np.abs(covariance)) <= np.finfo(float).tiny:
        return math.inf if float(np.linalg.norm(mean)) > 0.0 else 0.0
    inverse = np.linalg.pinv(covariance, rcond=1.0e-12)
    squared = float(mean @ inverse @ mean)
    return math.sqrt(max(squared, 0.0))
