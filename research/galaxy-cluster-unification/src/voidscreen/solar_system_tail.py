"""Solar-System diagnostics for the one-parameter isothermal acceleration tail."""

from __future__ import annotations

import math

import numpy as np


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
AU_M = 149_597_870_700.0
PARSEC_M = 3.085677581491367e16
JULIAN_YEAR_DAYS = 365.25
RAD_TO_MAS = 206_264_806.24709636


def extra_tail_acceleration_m_s2(
    radius_m,
    *,
    source_mass_kg: float = M_SUN_KG,
    parameter: float,
    reference_radius_m: float,
    a0_m_s2: float = 1.2e-10,
    screened: bool,
) -> np.ndarray:
    """Return the inward extra acceleration for the source-local point-mass law."""
    radius = np.asarray(radius_m, dtype=np.float64)
    if (
        np.any(~np.isfinite(radius))
        or np.any(radius <= 0.0)
        or not math.isfinite(source_mass_kg)
        or source_mass_kg <= 0.0
        or not math.isfinite(parameter)
        or parameter < 0.0
        or not math.isfinite(reference_radius_m)
        or reference_radius_m <= 0.0
        or not math.isfinite(a0_m_s2)
        or a0_m_s2 <= 0.0
    ):
        raise ValueError("solar-tail inputs must be finite and physical")
    mu = G_SI * source_mass_kg
    extra = parameter * mu / (reference_radius_m * radius)
    if screened:
        gbar = mu / np.square(radius)
        extra = extra * a0_m_s2 / (a0_m_s2 + gbar)
    return extra


def fractional_extra_force(
    radius_m,
    *,
    source_mass_kg: float = M_SUN_KG,
    parameter: float,
    reference_radius_m: float,
    a0_m_s2: float = 1.2e-10,
    screened: bool,
) -> np.ndarray:
    radius = np.asarray(radius_m, dtype=np.float64)
    newtonian = G_SI * source_mass_kg / np.square(radius)
    return extra_tail_acceleration_m_s2(
        radius,
        source_mass_kg=source_mass_kg,
        parameter=parameter,
        reference_radius_m=reference_radius_m,
        a0_m_s2=a0_m_s2,
        screened=screened,
    ) / newtonian


def secular_perihelion_precession_mas_per_century(
    *,
    semimajor_axis_m: float,
    eccentricity: float,
    orbital_period_days: float,
    source_mass_kg: float = M_SUN_KG,
    parameter: float,
    reference_radius_m: float,
    a0_m_s2: float = 1.2e-10,
    screened: bool,
    quadrature_points: int = 131_072,
) -> float:
    """First-order secular perihelion shift from the radial extra acceleration.

    The calculation time-averages the planar Gauss planetary equation over one
    unperturbed Kepler orbit. Inward radial perturbations use the conventional
    negative radial sign.
    """
    if (
        not math.isfinite(semimajor_axis_m)
        or semimajor_axis_m <= 0.0
        or not math.isfinite(eccentricity)
        or not 0.0 < eccentricity < 1.0
        or not math.isfinite(orbital_period_days)
        or orbital_period_days <= 0.0
        or quadrature_points < 1024
    ):
        raise ValueError("invalid osculating orbit")
    true_anomaly = np.linspace(
        0.0, 2.0 * np.pi, int(quadrature_points), endpoint=False
    )
    cosine = np.cos(true_anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    radius = (
        semimajor_axis_m
        * one_minus_e2
        / (1.0 + eccentricity * cosine)
    )
    radial_perturbation = -extra_tail_acceleration_m_s2(
        radius,
        source_mass_kg=source_mass_kg,
        parameter=parameter,
        reference_radius_m=reference_radius_m,
        a0_m_s2=a0_m_s2,
        screened=screened,
    )
    time_weight = one_minus_e2**1.5 / np.square(
        1.0 + eccentricity * cosine
    )
    mean_r_cosine = float(np.mean(radial_perturbation * cosine * time_weight))
    period_seconds = orbital_period_days * 86_400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = (
        -math.sqrt(one_minus_e2)
        / (mean_motion * semimajor_axis_m * eccentricity)
        * mean_r_cosine
    )
    radians_per_orbit = mean_rate * period_seconds
    orbits_per_century = 100.0 * JULIAN_YEAR_DAYS / orbital_period_days
    return radians_per_orbit * orbits_per_century * RAD_TO_MAS


def analytic_unscreened_precession_mas_per_century(
    *,
    semimajor_axis_m: float,
    eccentricity: float,
    orbital_period_days: float,
    source_mass_kg: float = M_SUN_KG,
    parameter: float,
    reference_radius_m: float,
) -> float:
    """Closed first-order result for an inward A/r perturbation."""
    root = math.sqrt(1.0 - eccentricity**2)
    period_seconds = orbital_period_days * 86_400.0
    mean_motion = 2.0 * math.pi / period_seconds
    source_mu = G_SI * source_mass_kg
    radians_per_orbit = (
        -2.0
        * math.pi
        * parameter
        * source_mu
        / (mean_motion**2 * semimajor_axis_m**2)
        / reference_radius_m
        * root
        * (1.0 - root)
        / eccentricity**2
    )
    orbits_per_century = 100.0 * JULIAN_YEAR_DAYS / orbital_period_days
    return radians_per_orbit * orbits_per_century * RAD_TO_MAS
