"""Common-grid stellar-light nuisance controls for Sigma source scoring."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]


def logical_pixels_to_common_kpc(
    x_zero_based: ArrayLike,
    y_zero_based: ArrayLike,
    *,
    center_logical_x: float,
    center_logical_y: float,
    native_pixel_kpc: float,
) -> tuple[FloatArray, FloatArray]:
    """Invert the exact V19X4 east/north-to-native-pixel convention."""

    x = np.asarray(x_zero_based, dtype=float)
    y = np.asarray(y_zero_based, dtype=float)
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("logical pixel coordinates must be matching and finite")
    center_x = float(center_logical_x)
    center_y = float(center_logical_y)
    pixel = float(native_pixel_kpc)
    if not all(math.isfinite(value) for value in (center_x, center_y, pixel)):
        raise ValueError("center and pixel scale must be finite")
    if center_x <= 0.0 or center_y <= 0.0 or pixel <= 0.0:
        raise ValueError("center and pixel scale must be positive")
    east = (center_x - 1.0 - x) * pixel
    north = (y - center_y + 1.0) * pixel
    return east, north


def cloud_in_cell_light_map(
    east_kpc: ArrayLike,
    north_kpc: ArrayLike,
    luminosity: ArrayLike,
    common_axis_kpc: ArrayLike,
) -> FloatArray:
    """Deposit one normalized member-light draw on a regular physical grid."""

    east = np.asarray(east_kpc, dtype=float)
    north = np.asarray(north_kpc, dtype=float)
    light = np.asarray(luminosity, dtype=float)
    axis = np.asarray(common_axis_kpc, dtype=float)
    if east.ndim != 1 or east.shape != north.shape or east.shape != light.shape:
        raise ValueError("member coordinates and luminosities must be matching vectors")
    if east.size == 0 or not np.all(np.isfinite(east)) or not np.all(np.isfinite(north)):
        raise ValueError("member coordinates must be nonempty and finite")
    if not np.all(np.isfinite(light)) or np.any(light < 0.0) or np.sum(light) <= 0.0:
        raise ValueError("member luminosities must be finite, nonnegative, and nonzero")
    if axis.ndim != 1 or axis.size < 3 or not np.all(np.isfinite(axis)):
        raise ValueError("common axis must be a finite vector")
    differences = np.diff(axis)
    if not np.allclose(differences, differences[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("common axis must be uniformly spaced")
    spacing = float(differences[0])
    if spacing <= 0.0:
        raise ValueError("common axis must increase")

    normalized = light / float(np.sum(light))
    x = (east - axis[0]) / spacing
    y = (north - axis[0]) / spacing
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    if np.any(ix < 0) or np.any(iy < 0) or np.any(ix + 1 >= axis.size) or np.any(iy + 1 >= axis.size):
        raise ValueError("a member lies outside the common-grid CIC support")
    dx = x - ix
    dy = y - iy
    output = np.zeros((axis.size, axis.size), dtype=float)
    contributions = (
        (iy, ix, (1.0 - dx) * (1.0 - dy)),
        (iy, ix + 1, dx * (1.0 - dy)),
        (iy + 1, ix, (1.0 - dx) * dy),
        (iy + 1, ix + 1, dx * dy),
    )
    for row, column, fraction in contributions:
        np.add.at(output, (row, column), normalized * fraction)
    if not math.isclose(float(np.sum(output)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("cloud-in-cell deposition did not conserve normalized light")
    return output


def smooth_light_draws(
    light_maps: ArrayLike,
    *,
    sigma_pixels: float,
) -> FloatArray:
    """Convolve independent point-light maps without mixing posterior draws."""

    maps = np.asarray(light_maps, dtype=float)
    sigma = float(sigma_pixels)
    if maps.ndim != 3 or not np.all(np.isfinite(maps)) or np.any(maps < 0.0):
        raise ValueError("light_maps must be a finite nonnegative draw stack")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_pixels must be finite and positive")
    before = np.sum(maps, axis=(-2, -1))
    if np.any(before <= 0.0):
        raise ValueError("every light draw must have positive total light")
    smoothed = gaussian_filter(maps, sigma=(0.0, sigma, sigma), mode="constant")
    after = np.sum(smoothed, axis=(-2, -1))
    if np.any(after <= 0.0):
        raise RuntimeError("smoothing erased a light draw")
    smoothed *= (before / after)[:, None, None]
    return smoothed


def region_light_percentile_ranks(
    smoothed_light: ArrayLike,
    label_grid: ArrayLike,
    region_ids: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return region-mean normalized light and within-draw percentile ranks."""

    maps = np.asarray(smoothed_light, dtype=float)
    labels = np.asarray(label_grid, dtype=np.int64)
    identifiers = np.asarray(region_ids, dtype=np.int64)
    if maps.ndim != 3 or maps.shape[-2:] != labels.shape:
        raise ValueError("light stack and label-grid shapes differ")
    if identifiers.ndim != 1 or len(np.unique(identifiers)) != identifiers.size:
        raise ValueError("region_ids must be a unique vector")
    means = np.full((maps.shape[0], identifiers.size), np.nan, dtype=float)
    for index, region_id in enumerate(identifiers):
        pixels = labels == int(region_id)
        if np.any(pixels):
            means[:, index] = np.mean(maps[:, pixels], axis=1)
    if not np.all(np.isfinite(means)):
        raise ValueError("every registered region must have common-grid pixels")
    ranks = np.empty_like(means)
    for draw in range(means.shape[0]):
        ranks[draw] = (rankdata(means[draw], method="average") - 0.5) / identifiers.size
    return means, ranks
