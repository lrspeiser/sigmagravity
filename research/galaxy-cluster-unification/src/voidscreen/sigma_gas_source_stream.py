"""Memory-bounded V19X4 gas-draw projection into V19BL regional features."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

from voidscreen.sigma_source_invariants import projected_source_maps, region_means

FloatArray = NDArray[np.float64]

REQUIRED_GAS_FIELDS = (
    "electron_density_cm3",
    "entropy_proxy_keV_cm2",
    "thermal_pressure_erg_cm3",
    "gas_surface_density_msun_kpc2",
)


def region_draws_to_grid(
    region_values: ArrayLike,
    region_ids: ArrayLike,
    label_grid: ArrayLike,
) -> FloatArray:
    """Map a draw-by-region batch to draw-by-north-by-east common grids."""

    values = np.asarray(region_values, dtype=float)
    identifiers = np.asarray(region_ids, dtype=np.int64)
    labels = np.asarray(label_grid, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != identifiers.size:
        raise ValueError("region_values must be draw by registered region")
    if labels.ndim != 2 or identifiers.ndim != 1:
        raise ValueError("labels must be a map and region_ids a vector")
    if len(np.unique(identifiers)) != identifiers.size or np.any(identifiers < 0):
        raise ValueError("region_ids must be unique and nonnegative")
    if not np.all(np.isfinite(values)):
        raise ValueError("regional draws must be finite")
    maximum = max(int(np.max(labels)), int(np.max(identifiers)), 0)
    lookup = np.full(maximum + 1, -1, dtype=np.int64)
    lookup[identifiers] = np.arange(identifiers.size)
    valid = (labels >= 0) & (labels <= maximum)
    index = np.full(labels.shape, -1, dtype=np.int64)
    index[valid] = lookup[labels[valid]]
    valid &= index >= 0
    output = np.full((values.shape[0], *labels.shape), np.nan, dtype=float)
    output[:, valid] = values[:, index[valid]]
    return output


def smooth_masked_draws(
    draw_maps: ArrayLike,
    *,
    sigma_pixels: float,
    conserve_integral: bool,
) -> FloatArray:
    """Apply the V19X4 mask-normalized Gaussian rule independently by draw."""

    maps = np.asarray(draw_maps, dtype=float)
    sigma = float(sigma_pixels)
    if maps.ndim != 3 or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("draw maps must be 3D and sigma positive")
    valid = np.isfinite(maps)
    if not np.all(valid == valid[0]):
        raise ValueError("all draws must share one validity mask")
    if not np.all(np.isfinite(maps[valid])):
        raise ValueError("valid map values must be finite")
    numerator = gaussian_filter(
        np.where(valid, maps, 0.0), sigma=(0.0, sigma, sigma), mode="constant"
    )
    weight = gaussian_filter(
        valid.astype(float), sigma=(0.0, sigma, sigma), mode="constant"
    )
    output = np.full(maps.shape, np.nan, dtype=float)
    supported = weight > 1.0e-8
    output[supported] = numerator[supported] / weight[supported]
    output[~valid] = np.nan
    if conserve_integral:
        before = np.nansum(maps, axis=(-2, -1))
        after = np.nansum(output, axis=(-2, -1))
        if np.any(before <= 0.0) or np.any(after <= 0.0):
            raise ValueError("cannot conserve a nonpositive field")
        output *= (before / after)[:, None, None]
    return output


def radial_mask(
    east_axis_kpc: ArrayLike, north_axis_kpc: ArrayLike, radius_kpc: float
) -> NDArray[np.bool_]:
    east = np.asarray(east_axis_kpc, dtype=float)
    north = np.asarray(north_axis_kpc, dtype=float)
    radius = float(radius_kpc)
    if east.ndim != 1 or north.ndim != 1 or not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("axes must be vectors and radius positive")
    grid_east, grid_north = np.meshgrid(east, north)
    return grid_east * grid_east + grid_north * grid_north <= radius * radius


def gas_feature_batch(
    regional_fields: Mapping[str, ArrayLike],
    *,
    region_ids: ArrayLike,
    label_grid: ArrayLike,
    east_axis_kpc: ArrayLike,
    north_axis_kpc: ArrayLike,
    spacing_kpc: float,
    smoothing_fwhm_kpc: Sequence[float],
    radii_kpc: Sequence[float],
) -> dict[str, FloatArray]:
    """Return every regional I4/I5 and gas-control quantity for one draw batch."""

    identifiers = np.asarray(region_ids, dtype=np.int64)
    missing = [name for name in REQUIRED_GAS_FIELDS if name not in regional_fields]
    if missing:
        raise ValueError(f"missing gas fields: {missing}")
    mapped = {
        name: region_draws_to_grid(regional_fields[name], identifiers, label_grid)
        for name in REQUIRED_GAS_FIELDS
    }
    result: dict[str, FloatArray] = {}
    for fwhm in smoothing_fwhm_kpc:
        token_scale = f"{float(fwhm):g}kpc"
        sigma_pixels = float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0))) / float(
            spacing_kpc
        )
        smoothed = {
            name: smooth_masked_draws(
                maps,
                sigma_pixels=sigma_pixels,
                conserve_integral=name == "gas_surface_density_msun_kpc2",
            )
            for name, maps in mapped.items()
        }
        sources = projected_source_maps(
            smoothed["electron_density_cm3"],
            smoothed["entropy_proxy_keV_cm2"],
            smoothed["thermal_pressure_erg_cm3"],
            smoothed["gas_surface_density_msun_kpc2"],
            spacing_kpc=float(spacing_kpc),
            resolution_fwhm_kpc=float(fwhm),
        )
        for radius in radii_kpc:
            token = f"{token_scale}_r{float(radius):g}kpc"
            admitted = radial_mask(east_axis_kpc, north_axis_kpc, float(radius))
            for name, values in sources.items():
                result[f"{name}_{token}"] = region_means(
                    values, label_grid, identifiers, radial_mask=admitted
                )
    return result


def append_feature_batch(
    accumulated: dict[str, list[FloatArray]], batch: Mapping[str, ArrayLike]
) -> None:
    """Append a batch while refusing schema drift between streaming chunks."""

    if accumulated and set(accumulated) != set(batch):
        raise ValueError("gas feature batch schema changed")
    for name, values in batch.items():
        accumulated.setdefault(name, []).append(np.asarray(values, dtype=float))


def concatenate_feature_batches(
    accumulated: Mapping[str, Sequence[ArrayLike]], expected_draws: int
) -> dict[str, FloatArray]:
    result = {
        name: np.concatenate([np.asarray(value, dtype=float) for value in values], axis=0)
        for name, values in accumulated.items()
    }
    if not result or any(values.shape[0] != expected_draws for values in result.values()):
        raise ValueError("streamed gas features do not contain the expected draws")
    return result
