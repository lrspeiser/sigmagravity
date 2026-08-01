"""Registered projected baryon maps for cluster field experiments.

The helpers in this module deliberately stop at a two-dimensional projected
mass description.  They do not infer a dark component and they do not use a
lensing residual.  This makes the resulting maps suitable as frozen inputs to
later inverse-routing or lensing tests.
"""

from __future__ import annotations

import math

import numpy as np


G_KPC_KM2_S2_MSUN = 4.30091e-6


def sky_to_lens_offsets(
    ra_deg,
    dec_deg,
    *,
    reference_ra_deg: float,
    reference_dec_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sky coordinates to Lenstool-style west/north offsets.

    Positive ``x`` is west and positive ``y`` is north.  The tangent-plane
    approximation is more than adequate for a cluster core a few arcminutes
    across.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    if np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)):
        raise ValueError("sky coordinates must be finite")
    cosine = math.cos(math.radians(float(reference_dec_deg)))
    x = -(ra - float(reference_ra_deg)) * 3600.0 * cosine
    y = (dec - float(reference_dec_deg)) * 3600.0
    return x, y


def dpie_axis_ratio(ellipticity: float) -> float:
    """Return b/a for the Lenstool ellipticity (a^2-b^2)/(a^2+b^2)."""
    value = float(ellipticity)
    if not 0.0 <= value < 1.0:
        raise ValueError("ellipticity must lie in [0,1)")
    return math.sqrt((1.0 - value) / (1.0 + value))


def elliptical_radius(
    xx,
    yy,
    *,
    center_x: float,
    center_y: float,
    axis_ratio: float,
    theta_deg: float,
) -> np.ndarray:
    """Elliptical radius with area-preserving major/minor rescaling."""
    q = float(axis_ratio)
    if not 0.0 < q <= 1.0:
        raise ValueError("axis ratio must lie in (0,1]")
    angle = math.radians(float(theta_deg))
    dx = np.asarray(xx, dtype=float) - float(center_x)
    dy = np.asarray(yy, dtype=float) - float(center_y)
    major = math.cos(angle) * dx + math.sin(angle) * dy
    minor = -math.sin(angle) * dx + math.cos(angle) * dy
    return np.hypot(major * math.sqrt(q), minor / math.sqrt(q))


def dpie_total_mass_msun(
    *,
    sigma_lt_km_s: float,
    r_core_arcsec: float,
    r_cut_arcsec: float,
    scale_kpc_per_arcsec: float,
) -> float:
    """Spherical dPIE total mass using sigma0=sqrt(3/2)*sigma_LT."""
    sigma_lt = float(sigma_lt_km_s)
    core = float(r_core_arcsec)
    cut = float(r_cut_arcsec)
    scale = float(scale_kpc_per_arcsec)
    if sigma_lt <= 0.0 or core < 0.0 or cut <= core or scale <= 0.0:
        raise ValueError("invalid dPIE mass parameters")
    sigma_zero_sq = 1.5 * sigma_lt * sigma_lt
    return math.pi * sigma_zero_sq * (cut - core) * scale / G_KPC_KM2_S2_MSUN


def dpie_surface_density_shape(
    xx,
    yy,
    *,
    center_x: float,
    center_y: float,
    ellipticity: float,
    theta_deg: float,
    r_core_arcsec: float,
    r_cut_arcsec: float,
) -> np.ndarray:
    """Return an unnormalized positive elliptical dPIE projected profile."""
    core = float(r_core_arcsec)
    cut = float(r_cut_arcsec)
    if core < 0.0 or cut <= core:
        raise ValueError("dPIE cut radius must exceed its core radius")
    radius = elliptical_radius(
        xx,
        yy,
        center_x=center_x,
        center_y=center_y,
        axis_ratio=dpie_axis_ratio(ellipticity),
        theta_deg=theta_deg,
    )
    surface = 1.0 / np.sqrt(radius * radius + core * core)
    surface -= 1.0 / np.sqrt(radius * radius + cut * cut)
    return np.maximum(surface, 0.0)


def sersic_surface_density_shape(
    xx,
    yy,
    *,
    center_x: float,
    center_y: float,
    effective_radius_arcsec: float,
    sersic_n: float,
    axis_ratio: float = 1.0,
    theta_deg: float = 0.0,
) -> np.ndarray:
    """Return a stable unnormalized projected Sersic profile."""
    radius_e = float(effective_radius_arcsec)
    index = float(sersic_n)
    if radius_e <= 0.0 or index <= 0.0:
        raise ValueError("Sersic radius and index must be positive")
    radius = elliptical_radius(
        xx,
        yy,
        center_x=center_x,
        center_y=center_y,
        axis_ratio=axis_ratio,
        theta_deg=theta_deg,
    )
    b_n = 2.0 * index - 1.0 / 3.0 + 4.0 / (405.0 * index)
    return np.exp(-b_n * np.power(radius / radius_e, 1.0 / index))


def gaussian_surface_density_shape(
    xx,
    yy,
    *,
    center_x: float,
    center_y: float,
    sigma_major_arcsec: float,
    axis_ratio: float,
    theta_deg: float,
) -> np.ndarray:
    """Return an unnormalized elliptical Gaussian surface profile."""
    sigma = float(sigma_major_arcsec)
    if sigma <= 0.0:
        raise ValueError("Gaussian width must be positive")
    radius = elliptical_radius(
        xx,
        yy,
        center_x=center_x,
        center_y=center_y,
        axis_ratio=axis_ratio,
        theta_deg=theta_deg,
    )
    return np.exp(-0.5 * np.square(radius / sigma))


def normalize_surface_mass(surface, total_mass_msun: float) -> np.ndarray:
    """Scale a nonnegative sampled surface shape to a requested total mass."""
    values = np.asarray(surface, dtype=float)
    total_mass = float(total_mass_msun)
    if values.ndim != 2 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("surface must be a finite nonnegative image")
    if total_mass < 0.0 or not math.isfinite(total_mass):
        raise ValueError("total mass must be finite and nonnegative")
    denominator = float(np.sum(values))
    if total_mass == 0.0:
        return np.zeros_like(values)
    if denominator <= 0.0:
        raise ValueError("positive mass requires a nonzero surface")
    return values * (total_mass / denominator)


def block_compress_surface(
    axis_arcsec,
    surface_mass,
    *,
    blocks_per_axis: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress a mass image into weighted block centroids without mass loss."""
    axis = np.asarray(axis_arcsec, dtype=float)
    mass = np.asarray(surface_mass, dtype=float)
    if axis.ndim != 1 or mass.shape != (len(axis), len(axis)):
        raise ValueError("surface shape must match a one-dimensional square axis")
    if np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("surface mass must be finite and nonnegative")
    blocks = int(blocks_per_axis)
    if blocks < 1 or blocks > len(axis):
        raise ValueError("invalid number of compression blocks")
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    edges = np.linspace(0, len(axis), blocks + 1, dtype=int)
    result_x: list[float] = []
    result_y: list[float] = []
    result_mass: list[float] = []
    for iy in range(blocks):
        for ix in range(blocks):
            ys = slice(edges[iy], edges[iy + 1])
            xs = slice(edges[ix], edges[ix + 1])
            block_mass = mass[ys, xs]
            total = float(np.sum(block_mass))
            if total <= 0.0:
                continue
            result_x.append(float(np.sum(block_mass * xx[ys, xs]) / total))
            result_y.append(float(np.sum(block_mass * yy[ys, xs]) / total))
            result_mass.append(total)
    return np.asarray(result_x), np.asarray(result_y), np.asarray(result_mass)
