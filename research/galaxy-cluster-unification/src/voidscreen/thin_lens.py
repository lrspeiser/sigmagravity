"""Coordinate-safe thin-lens deflection from a resolved baryonic surface map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19
RAD_TO_ARCSEC = 206_264.80624709636


@dataclass(frozen=True)
class ThinLensDeflection:
    """Physical deflection at distance ratio one on ``(north, east)`` rows."""

    alpha_east_radian: np.ndarray
    alpha_north_radian: np.ndarray
    alpha_east_arcsec: np.ndarray
    alpha_north_arcsec: np.ndarray
    input_mass_msun: float


def thin_lens_deflection_from_surface_density(
    surface_density_msun_kpc2,
    cell_kpc: float,
    *,
    gravitational_constant: float = G_SI,
    light_speed: float = C_SI,
) -> ThinLensDeflection:
    r"""Convolve a resolved surface density with the exact thin-lens kernel.

    At distance ratio one,

    .. math::

       \boldsymbol\alpha(\boldsymbol\xi)=\frac{4G}{c^2}
       \int \Sigma(\boldsymbol\xi')
       \frac{\boldsymbol\xi-\boldsymbol\xi'}
       {|\boldsymbol\xi-\boldsymbol\xi'|^2}\,d^2\xi'.

    Array axis 0 is north and axis 1 is east.  Zero padding is implicit in the
    linear FFT convolution; no periodic mass copies are introduced.
    """
    surface = np.asarray(surface_density_msun_kpc2, dtype=float)
    if surface.ndim != 2 or min(surface.shape) < 5:
        raise ValueError("surface density must be a two-dimensional grid")
    if np.any(~np.isfinite(surface)) or np.any(surface < 0.0):
        raise ValueError("surface density must be finite and nonnegative")
    if not np.isfinite(cell_kpc) or cell_kpc <= 0.0:
        raise ValueError("cell_kpc must be finite and positive")
    if gravitational_constant <= 0.0 or light_speed <= 0.0:
        raise ValueError("physical constants must be positive")
    cell_m = float(cell_kpc) * KPC_M
    surface_si = surface * M_SUN_KG / KPC_M**2
    north_offsets = (
        np.arange(-(surface.shape[0] - 1), surface.shape[0], dtype=float) * cell_m
    )
    east_offsets = (
        np.arange(-(surface.shape[1] - 1), surface.shape[1], dtype=float) * cell_m
    )
    east_grid, north_grid = np.meshgrid(east_offsets, north_offsets, indexing="xy")
    radius_squared = east_grid * east_grid + north_grid * north_grid
    kernel_east = np.divide(
        east_grid,
        radius_squared,
        out=np.zeros_like(east_grid),
        where=radius_squared > 0.0,
    )
    kernel_north = np.divide(
        north_grid,
        radius_squared,
        out=np.zeros_like(north_grid),
        where=radius_squared > 0.0,
    )
    multiplier = (
        4.0
        * float(gravitational_constant)
        * cell_m**2
        / float(light_speed) ** 2
    )
    alpha_east = multiplier * fftconvolve(
        surface_si, kernel_east, mode="same"
    )
    alpha_north = multiplier * fftconvolve(
        surface_si, kernel_north, mode="same"
    )
    return ThinLensDeflection(
        alpha_east_radian=alpha_east,
        alpha_north_radian=alpha_north,
        alpha_east_arcsec=alpha_east * RAD_TO_ARCSEC,
        alpha_north_arcsec=alpha_north * RAD_TO_ARCSEC,
        input_mass_msun=float(np.sum(surface) * float(cell_kpc) ** 2),
    )
