from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma, gammainc, gammaincc, gammaln

from .data import KPC_M
from .unified import C_M_S, G_SI, M_SUN_KG


def hernquist_local_density_from_total_mass(
    total_mass_msun,
    radius_kpc,
    effective_radius_kpc,
) -> np.ndarray:
    """Return the local Hernquist density in g cm^-3.

    The scale radius is ``a=0.551 R_e``, matching the BCG convention used
    by Tian et al. (2020).  ``total_mass_msun`` is the asymptotic Hernquist
    stellar mass, not the mass enclosed at ``radius_kpc``.
    """
    mass = np.asarray(total_mass_msun, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    effective_radius = np.asarray(effective_radius_kpc, dtype=float)
    mass, radius, effective_radius = np.broadcast_arrays(
        mass, radius, effective_radius
    )
    if (
        np.any(~np.isfinite(mass))
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(effective_radius))
        or np.any(mass <= 0.0)
        or np.any(radius <= 0.0)
        or np.any(effective_radius <= 0.0)
    ):
        raise ValueError("mass and radii must be finite and positive")
    scale = 0.551 * effective_radius
    density_msun_kpc3 = mass * scale / (
        2.0 * np.pi * radius * np.power(radius + scale, 3)
    )
    conversion = M_SUN_KG * 1.0e3 / np.power(KPC_M * 100.0, 3)
    return density_msun_kpc3 * conversion


def prugniel_simien_local_density_from_enclosed_mass(
    enclosed_mass_msun,
    radius_kpc,
    effective_radius_kpc,
    sersic_n,
) -> np.ndarray:
    """Infer local 3D density from an enclosed mass and a Sersic shape.

    The Prugniel-Simien approximation is normalized so its spherical mass
    inside ``radius_kpc`` equals the supplied enclosed mass.  This makes the
    local-density estimate consistent with an independently reported
    baryonic acceleration at the same radius without importing a second mass
    normalization.

    The output unit is g cm^-3.
    """
    mass = np.asarray(enclosed_mass_msun, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    effective_radius = np.asarray(effective_radius_kpc, dtype=float)
    index = np.asarray(sersic_n, dtype=float)
    mass, radius, effective_radius, index = np.broadcast_arrays(
        mass, radius, effective_radius, index
    )
    if (
        np.any(~np.isfinite(mass))
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(effective_radius))
        or np.any(~np.isfinite(index))
        or np.any(mass <= 0.0)
        or np.any(radius <= 0.0)
        or np.any(effective_radius <= 0.0)
        or np.any(index <= 0.0)
    ):
        raise ValueError("mass, radii, and Sersic index must be finite and positive")

    ratio = radius / effective_radius
    p = 1.0 - 0.6097 / index + 0.05463 / np.square(index)
    b = 2.0 * index - 1.0 / 3.0 + 0.009876 / index
    shape = index * (3.0 - p)
    transformed_radius = b * np.power(ratio, 1.0 / index)
    regularized_mass = gammainc(shape, transformed_radius)
    if np.any(regularized_mass <= 0.0):
        raise ValueError("Sersic enclosed-mass fraction underflowed")

    log_lower_gamma = gammaln(shape) + np.log(regularized_mass)
    log_density_msun_kpc3 = (
        np.log(mass)
        + shape * np.log(b)
        - np.log(4.0 * np.pi * index)
        - 3.0 * np.log(effective_radius)
        - log_lower_gamma
        - p * np.log(ratio)
        - transformed_radius
    )
    # 1 M_sun/kpc^3 in g/cm^3.
    conversion = M_SUN_KG * 1.0e3 / np.power(KPC_M * 100.0, 3)
    return np.exp(log_density_msun_kpc3) * conversion


def nfw_mass_function(radius_over_scale) -> np.ndarray:
    value = np.asarray(radius_over_scale, dtype=float)
    if np.any(value < 0.0):
        raise ValueError("NFW radius ratio must be non-negative")
    return np.log1p(value) - value / (1.0 + value)


def nfw_overdensity_conversion(
    concentration: float,
    *,
    delta_from: float = 200.0,
    delta_to: float = 500.0,
) -> tuple[float, float]:
    """Return R_to/R_from and M_to/M_from for a truncated NFW halo."""
    if concentration <= 0.0 or delta_from <= 0.0 or delta_to <= delta_from:
        raise ValueError("require concentration > 0 and delta_to > delta_from > 0")
    normalization = float(nfw_mass_function(concentration))

    def root(radius_ratio: float) -> float:
        mass_ratio = float(nfw_mass_function(concentration * radius_ratio)) / normalization
        return mass_ratio - (delta_to / delta_from) * radius_ratio**3

    radius_ratio = brentq(root, 1e-4, 1.0 - 1e-12)
    mass_ratio = float(nfw_mass_function(concentration * radius_ratio)) / normalization
    return radius_ratio, mass_ratio


def truncated_nfw_potential_factor(concentration, radius_over_outer=0.0) -> np.ndarray:
    """Potential magnitude divided by G M(<R)/(R) for an NFW profile cut at R."""
    concentration = np.asarray(concentration, dtype=float)
    radius = np.asarray(radius_over_outer, dtype=float)
    if np.any(concentration <= 0.0) or np.any(radius < 0.0) or np.any(radius > 1.0):
        raise ValueError("require concentration > 0 and 0 <= radius/R <= 1")
    mass_norm = nfw_mass_function(concentration)
    central = np.square(concentration) / ((1.0 + concentration) * mass_norm)
    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    enclosed = nfw_mass_function(concentration * safe_radius) / mass_norm
    exterior = (
        concentration**2
        * (1.0 / (1.0 + concentration * safe_radius) - 1.0 / (1.0 + concentration))
        / mass_norm
    )
    factor = enclosed / safe_radius + exterior
    return np.where(radius == 0.0, central, factor)


def sersic_deprojected_potential_factor(radius_over_re, sersic_n) -> np.ndarray:
    """Complete an enclosed-mass potential with exterior Sersic stellar shells.

    The Prugniel-Simien deprojection fixes the shape. The returned factor multiplies
    G M(<r)/r, so it does not introduce a new stellar-mass normalization.
    """
    radius = np.asarray(radius_over_re, dtype=float)
    index = np.asarray(sersic_n, dtype=float)
    if np.any(radius <= 0.0) or np.any(index <= 0.0):
        raise ValueError("Sersic radius ratio and index must be positive")
    p = 1.0 - 0.6097 / index + 0.05463 / np.square(index)
    b = 2.0 * index - 1.0 / 3.0 + 4.0 / (405.0 * index)
    b += 46.0 / (25515.0 * np.square(index))
    transformed_radius = b * np.power(radius, 1.0 / index)
    mass_shape = index * (3.0 - p)
    potential_shape = index * (2.0 - p)
    enclosed_fraction = gammainc(mass_shape, transformed_radius)
    exterior_over_total = (
        np.power(b, index)
        * gamma(potential_shape)
        * gammaincc(potential_shape, transformed_radius)
        / gamma(mass_shape)
    )
    exterior_over_interior_potential = radius * exterior_over_total / enclosed_fraction
    return 1.0 + exterior_over_interior_potential


def vikhlinin_density_shape(
    radius_over_r500: float,
    *,
    alpha: float,
    beta: float,
    core_over_r500: float,
    steepening_over_r500: float,
    epsilon: float,
) -> float:
    radius = max(float(radius_over_r500), 1e-14)
    if core_over_r500 <= 0.0 or steepening_over_r500 <= 0.0:
        raise ValueError("gas-profile radii must be positive")
    density_squared = np.power(radius / core_over_r500, -alpha)
    density_squared /= np.power(
        1.0 + np.square(radius / core_over_r500), 3.0 * beta - alpha / 2.0
    )
    density_squared /= np.power(
        1.0 + np.power(radius / steepening_over_r500, 3.0), epsilon / 3.0
    )
    return float(np.sqrt(density_squared))


def spherical_profile_potential_factor(
    density_shape: Callable[[float], float],
    radius_over_outer: float,
) -> float:
    """Return |Phi(r)|/[G M(<R)/R] for a spherical profile truncated at R."""
    radius = float(radius_over_outer)
    if not 0.0 <= radius <= 1.0:
        raise ValueError("radius_over_outer must lie in [0, 1]")
    mass_integral = quad(
        lambda value: density_shape(value) * value**2,
        0.0,
        1.0,
        epsabs=1e-10,
        limit=300,
    )[0]
    if not np.isfinite(mass_integral) or mass_integral <= 0.0:
        raise ValueError("density profile has no finite positive mass")
    exterior = quad(
        lambda value: density_shape(value) * value,
        radius,
        1.0,
        epsabs=1e-10,
        limit=300,
    )[0]
    if radius == 0.0:
        return float(exterior / mass_integral)
    enclosed = quad(
        lambda value: density_shape(value) * value**2,
        0.0,
        radius,
        epsabs=1e-10,
        limit=300,
    )[0]
    return float((enclosed / radius + exterior) / mass_integral)


def potential_chi_from_mass(mass_msun, outer_radius_kpc, profile_factor=1.0) -> np.ndarray:
    mass = np.asarray(mass_msun, dtype=float)
    radius = np.asarray(outer_radius_kpc, dtype=float)
    factor = np.asarray(profile_factor, dtype=float)
    if np.any(mass < 0.0) or np.any(radius <= 0.0) or np.any(factor <= 0.0):
        raise ValueError("mass must be non-negative and radius/profile factor positive")
    return G_SI * mass * M_SUN_KG * factor / (radius * KPC_M * C_M_S**2)
