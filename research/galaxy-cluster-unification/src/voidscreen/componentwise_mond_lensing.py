"""Componentwise nonlinear-before-summation MOND lensing diagnostics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19
RAD_TO_ARCSEC = 206_264.80624709636


@lru_cache(maxsize=8)
def _simple_mond_kernel_table(
    table_points: int = 4096,
    quadrature_points: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the dimensionless simple-mu point-mass excess deflection."""
    if table_points < 256 or quadrature_points < 32:
        raise ValueError("kernel table resolution is too small")
    log_x = np.linspace(np.log(1.0e-7), np.log(1.0e7), int(table_points))
    x = np.exp(log_x)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    maximum_t = 32.0
    t = 0.5 * maximum_t * (nodes + 1.0)
    quadrature_weight = 0.5 * maximum_t * weights
    inverse_cosh_squared = 1.0 / np.cosh(t) ** 2
    y = inverse_cosh_squared[None, :] / x[:, None] ** 2
    # Stable form of 0.5*(sqrt(y^2+4y)-y), in units of a0.
    excess = 2.0 * y / (np.sqrt(y * y + 4.0 * y) + y)
    kernel = 4.0 * x * np.sum(excess * quadrature_weight[None, :], axis=1)
    return log_x, kernel


def dimensionless_simple_mond_excess_deflection(impact_over_transition_radius):
    """Return the universal point-mass excess-deflection kernel.

    If ``r_M=sqrt(GM/a0)`` and ``x=b/r_M``, the physical excess deflection is
    ``sqrt(G M a0)/c^2 * F(x)``.  ``F`` tends to ``2*pi`` in the deep-MOND
    exterior and tends to zero in the high-acceleration interior.
    """
    x = np.asarray(impact_over_transition_radius, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("impact ratio must be finite and positive")
    log_x, kernel = _simple_mond_kernel_table()
    result = np.interp(np.log(x), log_x, kernel)
    result = np.where(np.log(x) > log_x[-1], 2.0 * np.pi, result)
    return result


def componentwise_simple_mond_excess_deflection(
    east_arcsec,
    north_arcsec,
    member_east_arcsec,
    member_north_arcsec,
    member_mass_msun,
    *,
    kpc_per_arcsec: float,
    distance_ratio: float,
    softening_kpc,
    a0_m_s2: float = 1.2e-10,
    gravitational_constant: float = G_SI,
    light_speed: float = C_SI,
    point_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum nonlinear point-member excesses after evaluating each component.

    This is deliberately not AQUAL or QUMOND: it tests whether applying the
    nonlinear response to resolved baryonic components *before* vector
    summation explains cluster lensing structure.
    """
    east, north = np.broadcast_arrays(
        np.asarray(east_arcsec, dtype=float), np.asarray(north_arcsec, dtype=float)
    )
    member_east = np.atleast_1d(np.asarray(member_east_arcsec, dtype=float))
    member_north = np.atleast_1d(np.asarray(member_north_arcsec, dtype=float))
    mass = np.atleast_1d(np.asarray(member_mass_msun, dtype=float))
    softening = np.broadcast_to(np.asarray(softening_kpc, dtype=float), mass.shape)
    if member_east.shape != member_north.shape or member_east.shape != mass.shape:
        raise ValueError("member coordinates and masses must have matching shapes")
    if (
        np.any(~np.isfinite(east))
        or np.any(~np.isfinite(north))
        or np.any(~np.isfinite(member_east))
        or np.any(~np.isfinite(member_north))
        or np.any(~np.isfinite(mass))
        or np.any(mass <= 0.0)
        or np.any(~np.isfinite(softening))
        or np.any(softening <= 0.0)
    ):
        raise ValueError("componentwise lens inputs must be finite and physical")
    if (
        kpc_per_arcsec <= 0.0
        or distance_ratio <= 0.0
        or a0_m_s2 <= 0.0
        or gravitational_constant <= 0.0
        or light_speed <= 0.0
        or point_chunk_size < 1
    ):
        raise ValueError("physical scales and chunk size must be positive")

    transition_kpc = (
        np.sqrt(float(gravitational_constant) * mass * M_SUN_KG / float(a0_m_s2))
        / KPC_M
    )
    amplitude_arcsec = (
        np.sqrt(
            float(gravitational_constant) * mass * M_SUN_KG * float(a0_m_s2)
        )
        / float(light_speed) ** 2
        * RAD_TO_ARCSEC
        * float(distance_ratio)
    )
    flat_east = east.ravel()
    flat_north = north.ravel()
    result_east = np.empty_like(flat_east)
    result_north = np.empty_like(flat_north)
    for start in range(0, len(flat_east), int(point_chunk_size)):
        stop = min(start + int(point_chunk_size), len(flat_east))
        delta_east_kpc = (
            flat_east[start:stop, None] - member_east[None, :]
        ) * float(kpc_per_arcsec)
        delta_north_kpc = (
            flat_north[start:stop, None] - member_north[None, :]
        ) * float(kpc_per_arcsec)
        impact_kpc = np.sqrt(
            delta_east_kpc**2
            + delta_north_kpc**2
            + softening[None, :] ** 2
        )
        magnitude = amplitude_arcsec[None, :] * (
            dimensionless_simple_mond_excess_deflection(
                impact_kpc / transition_kpc[None, :]
            )
        )
        result_east[start:stop] = np.sum(
            magnitude * delta_east_kpc / impact_kpc, axis=1
        )
        result_north[start:stop] = np.sum(
            magnitude * delta_north_kpc / impact_kpc, axis=1
        )
    return result_east.reshape(east.shape), result_north.reshape(north.shape)
