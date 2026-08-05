"""Target-independent regional sampling and common-grid gas reconstruction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

FloatArray = NDArray[np.float64]


def _positive_scalar(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _unit_uniforms(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0) or np.any(array >= 1.0):
        raise ValueError(f"{name} must lie strictly between zero and one")
    return array


def cluster_sobol_uniforms(
    region_count: int,
    draws: int,
    seed: int,
    *,
    rank_correlation: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return region-specific temperature/norm uniforms and one shared depth draw.

    A common depth-factor draw represents the cluster-wide deprojection scale.
    Temperature and normalization are region-specific. Their latent Gaussian
    copula has the declared correlation and uses the same Sobol base points for
    every sensitivity branch.
    """

    if region_count <= 0:
        raise ValueError("region_count must be positive")
    if draws <= 0 or draws & (draws - 1):
        raise ValueError("draws must be a positive power of two")
    correlation = float(rank_correlation)
    if not math.isfinite(correlation) or abs(correlation) >= 1.0:
        raise ValueError("rank_correlation must lie strictly between -1 and 1")
    dimension = 2 * region_count + 1
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    base = sampler.random_base2(int(math.log2(draws))).T
    epsilon = np.finfo(float).eps
    base = np.clip(base, epsilon, 1.0 - epsilon)
    temperature = base[:region_count]
    independent_norm = base[region_count : 2 * region_count]
    z_temperature = ndtri(temperature)
    z_independent = ndtri(independent_norm)
    z_norm = correlation * z_temperature + math.sqrt(1.0 - correlation**2) * z_independent
    normalization = np.clip(ndtr(z_norm), epsilon, 1.0 - epsilon)
    depth = base[-1]
    return temperature, normalization, depth


def positive_profile_draws(
    best: Any,
    lower: Any,
    upper: Any,
    bounds: tuple[float, float],
    uniforms: ArrayLike,
) -> tuple[FloatArray, str]:
    """Draw a positive parameter from an asymmetric log profile or full bound.

    An ordered 68-percent interval is interpreted as the minus/plus one-sigma
    scale in log space. If the interval failed, the predeclared full log-uniform
    fit bound is used instead of dropping or selectively refitting the region.
    """

    value = _positive_scalar(best, name="best")
    minimum = _positive_scalar(bounds[0], name="minimum bound")
    maximum = _positive_scalar(bounds[1], name="maximum bound")
    if not minimum < value < maximum:
        raise ValueError("best must lie strictly inside its positive bounds")
    u = _unit_uniforms(uniforms, name="uniforms")
    try:
        low = float(lower)
        high = float(upper)
    except (TypeError, ValueError):
        low = math.nan
        high = math.nan
    ordered = (
        math.isfinite(low)
        and math.isfinite(high)
        and minimum < low < value < high < maximum
    )
    if not ordered:
        draws = np.exp(math.log(minimum) + u * math.log(maximum / minimum))
        return draws, "full_frozen_log_bound_fallback"

    z = ndtri(u)
    log_best = math.log(value)
    lower_sigma = log_best - math.log(low)
    upper_sigma = math.log(high) - log_best
    scale = np.where(z < 0.0, lower_sigma, upper_sigma)
    draws = np.exp(log_best + z * scale)
    draws = np.clip(draws, np.nextafter(minimum, maximum), np.nextafter(maximum, minimum))
    return draws, "asymmetric_log_profile"


def log_uniform_depth_factors(
    uniforms: ArrayLike, minimum: float, maximum: float
) -> FloatArray:
    u = _unit_uniforms(uniforms, name="depth uniforms")
    low = _positive_scalar(minimum, name="minimum depth factor")
    high = _positive_scalar(maximum, name="maximum depth factor")
    if low >= high:
        raise ValueError("depth-factor bounds must be ordered")
    return np.exp(math.log(low) + u * math.log(high / low))


def common_grid_axis(half_width_kpc: float, spacing_kpc: float) -> FloatArray:
    half_width = _positive_scalar(half_width_kpc, name="half_width_kpc")
    spacing = _positive_scalar(spacing_kpc, name="spacing_kpc")
    intervals = 2.0 * half_width / spacing
    rounded = round(intervals)
    if not math.isclose(intervals, rounded, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("twice the half width must be divisible by spacing")
    return np.linspace(-half_width, half_width, rounded + 1, dtype=float)


def resample_bin_labels_to_physical_grid(
    binmap: ArrayLike,
    *,
    center_logical_x: float,
    center_logical_y: float,
    native_pixel_kpc: float,
    common_axis_kpc: ArrayLike,
) -> NDArray[np.int64]:
    """Nearest-neighbor sample labels onto east-positive/north-positive axes."""

    labels = np.asarray(binmap)
    if labels.ndim != 2 or not np.all(np.isfinite(labels)):
        raise ValueError("binmap must be a finite two-dimensional array")
    pixel = _positive_scalar(native_pixel_kpc, name="native_pixel_kpc")
    center_x = _positive_scalar(center_logical_x, name="center_logical_x")
    center_y = _positive_scalar(center_logical_y, name="center_logical_y")
    axis = np.asarray(common_axis_kpc, dtype=float)
    if axis.ndim != 1 or len(axis) < 3 or not np.all(np.isfinite(axis)):
        raise ValueError("common_axis_kpc must be a finite one-dimensional axis")
    east, north = np.meshgrid(axis, axis)
    native_column = center_x - east / pixel - 1.0
    native_row = center_y + north / pixel - 1.0
    sampled = map_coordinates(
        labels.astype(float),
        [native_row, native_column],
        order=0,
        mode="constant",
        cval=-1.0,
        prefilter=False,
    )
    return np.rint(sampled).astype(np.int64)


def map_region_values(
    label_grid: ArrayLike, region_ids: ArrayLike, region_values: ArrayLike
) -> FloatArray:
    labels = np.asarray(label_grid, dtype=np.int64)
    identifiers = np.asarray(region_ids, dtype=np.int64)
    values = np.asarray(region_values, dtype=float)
    if labels.ndim != 2:
        raise ValueError("label_grid must be two-dimensional")
    if identifiers.ndim != 1 or values.shape != identifiers.shape:
        raise ValueError("region_ids and region_values must be matching vectors")
    if len(np.unique(identifiers)) != len(identifiers) or np.any(identifiers < 0):
        raise ValueError("region_ids must be unique and non-negative")
    if not np.all(np.isfinite(values)):
        raise ValueError("region_values must be finite")
    maximum = max(int(np.max(labels)), int(np.max(identifiers)), 0)
    lookup = np.full(maximum + 1, np.nan, dtype=float)
    lookup[identifiers] = values
    output = np.full(labels.shape, np.nan, dtype=float)
    valid = (labels >= 0) & (labels <= maximum)
    output[valid] = lookup[labels[valid]]
    return output


def smooth_masked_field(
    field: ArrayLike,
    *,
    sigma_pixels: float,
    conserve_integral: bool,
) -> FloatArray:
    """Gaussian smooth a masked field without treating missing pixels as zero."""

    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    sigma = _positive_scalar(sigma_pixels, name="sigma_pixels")
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError("field must contain at least one finite pixel")
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma, mode="constant")
    weight = gaussian_filter(valid.astype(float), sigma, mode="constant")
    output = np.full(values.shape, np.nan, dtype=float)
    supported = weight > 1e-8
    output[supported] = numerator[supported] / weight[supported]
    output[~valid] = np.nan
    if conserve_integral:
        before = float(np.sum(values[valid]))
        after = float(np.sum(output[valid]))
        if not math.isfinite(before) or not math.isfinite(after) or after == 0.0:
            raise ValueError("cannot conserve a non-finite or zero field integral")
        output[valid] *= before / after
    return output


def quantile_summary(draws: ArrayLike) -> dict[str, FloatArray]:
    array = np.asarray(draws, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("draws must be a finite region-by-draw matrix")
    quantiles = np.quantile(array, [0.05, 0.16, 0.5, 0.84, 0.95], axis=1)
    return {
        "q05": quantiles[0],
        "q16": quantiles[1],
        "median": quantiles[2],
        "q84": quantiles[3],
        "q95": quantiles[4],
    }
