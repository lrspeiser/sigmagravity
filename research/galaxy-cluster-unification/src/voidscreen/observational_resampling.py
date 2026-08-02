"""Mass-conservative common-resolution operators for map validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CommonResolutionMaps:
    filtered_native: np.ndarray
    coarse: np.ndarray
    downsampling_ratio: float
    added_native_gaussian_sigma_pixels: float
    filtered_mass_relative_error: float
    coarse_mass_relative_error: float


def _square(values: np.ndarray) -> np.ndarray:
    surface = np.asarray(values, dtype=float)
    if (
        surface.ndim != 2
        or surface.shape[0] != surface.shape[1]
        or surface.shape[0] < 17
        or not np.all(np.isfinite(surface))
        or np.any(surface < 0.0)
    ):
        raise ValueError("surface must be a finite nonnegative square map")
    return surface


def common_resolution_surface_density(
    surface_density: np.ndarray,
    target_cells: int,
) -> CommonResolutionMaps:
    """Filter native and coarse maps to one effective Gaussian resolution.

    Surface-density integrals are conserved using the implied physical pixel
    areas. The Gaussian width is a measurement operator, not a gravity-law
    parameter.
    """

    surface = _square(surface_density)
    target = int(target_cells)
    if target < 17 or target % 2 == 0 or target >= surface.shape[0]:
        raise ValueError("target_cells must be a smaller odd grid of at least 17")
    ratio = float((surface.shape[0] - 1) / (target - 1))
    added_sigma = 0.5 * np.sqrt(max(ratio**2 - 1.0, 0.0))
    filtered = ndimage.gaussian_filter(
        surface,
        sigma=added_sigma,
        mode="constant",
        cval=0.0,
    )
    native_sum = float(np.sum(surface))
    filtered_sum = float(np.sum(filtered))
    if native_sum > 0.0 and filtered_sum > 0.0:
        filtered *= native_sum / filtered_sum
    coordinates = np.linspace(0.0, surface.shape[0] - 1.0, target)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    coarse = ndimage.map_coordinates(
        filtered,
        [yy, xx],
        order=1,
        mode="constant",
        cval=0.0,
    )
    coarse_physical_sum = float(np.sum(coarse)) * ratio**2
    filtered_physical_sum = float(np.sum(filtered))
    if filtered_physical_sum > 0.0 and coarse_physical_sum > 0.0:
        coarse *= filtered_physical_sum / coarse_physical_sum
    tiny = np.finfo(float).tiny
    return CommonResolutionMaps(
        filtered_native=filtered,
        coarse=coarse,
        downsampling_ratio=ratio,
        added_native_gaussian_sigma_pixels=float(added_sigma),
        filtered_mass_relative_error=float(
            abs(float(np.sum(filtered)) / max(native_sum, tiny) - 1.0)
        ),
        coarse_mass_relative_error=float(
            abs(
                float(np.sum(coarse)) * ratio**2
                / max(float(np.sum(filtered)), tiny)
                - 1.0
            )
        ),
    )
