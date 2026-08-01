"""Scale-free gravity-arc apogee and residence-time phenomenology."""

from __future__ import annotations

import math

import numpy as np

from .data import KPC_M


G_SI = 6.67430e-11
M_SUN_KG = 1.988409870698051e30
AU_M = 149_597_870_700.0
RAD_TO_MAS = 206_264_806.24709636
JULIAN_YEAR_DAYS = 365.25


def extent_gate(concentration, mode: str) -> np.ndarray:
    """Return a bounded collective-return gate from R50/R80 concentration."""
    value = np.asarray(concentration, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("concentration must be finite and positive")
    if mode == "none":
        return np.ones_like(value)
    if mode == "cluster_logistic_soft":
        beta = 0.5 * 4.776372627689756
    elif mode == "cluster_logistic":
        beta = 4.776372627689756
    elif mode == "cluster_logistic_sharp":
        beta = 2.0 * 4.776372627689756
    else:
        raise ValueError(f"unknown extent gate {mode}")
    return 1.0 / (1.0 + np.exp(-beta * (value - 0.6485259912066459)))


def residence_coordinate(radius, scale_radius, *, alpha: float, apogee_ratio: float):
    """Accumulated path residence, linear for alpha=1 before saturation."""
    radius = np.asarray(radius, dtype=float)
    scale = np.asarray(scale_radius, dtype=float)
    radius, scale = np.broadcast_arrays(radius, scale)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale radius must be finite and positive")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(apogee_ratio) or apogee_ratio <= 0.0:
        raise ValueError("apogee ratio must be finite and positive")
    x = radius / scale
    numerator = np.power(x, float(alpha))
    return numerator / (1.0 + numerator / float(apogee_ratio) ** float(alpha))


def acceleration_screen(gbar_m_s2, *, a0_m_s2: float, exponent: float):
    """Suppress the nonlocal channel in high-acceleration environments."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if a0_m_s2 <= 0.0 or exponent <= 0.0:
        raise ValueError("screen constants must be positive")
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.power(gbar / float(a0_m_s2), float(exponent)))


def arc_apogee_enhancement(
    gbar_m_s2,
    radius_kpc,
    scale_radius_kpc,
    concentration,
    *,
    residence_strength: float,
    alpha: float,
    apogee_ratio: float,
    gate_mode: str,
    screen_a0_m_s2: float,
    screen_exponent: float,
) -> dict[str, np.ndarray]:
    """Return the universal multiplicative field enhancement."""
    if not math.isfinite(residence_strength) or residence_strength < 0.0:
        raise ValueError("residence strength must be finite and non-negative")
    coordinate = residence_coordinate(
        radius_kpc, scale_radius_kpc, alpha=alpha, apogee_ratio=apogee_ratio
    )
    screen = acceleration_screen(
        gbar_m_s2, a0_m_s2=screen_a0_m_s2, exponent=screen_exponent
    )
    gate = extent_gate(concentration, gate_mode)
    gate = np.broadcast_to(gate, coordinate.shape)
    fractional = float(residence_strength) * gate * coordinate * screen
    return {
        "enhancement_relative_to_local_G": 1.0 + fractional,
        "fractional_enhancement": fractional,
        "residence_coordinate": coordinate,
        "extent_gate": gate,
        "screen": screen,
    }


def mass_radius_kpc(mass_solar, *, a0_m_s2: float) -> np.ndarray:
    """Return sqrt(GM/a0), the mass-derived transition radius."""
    mass = np.asarray(mass_solar, dtype=float)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("mass must be finite and positive")
    return np.sqrt(G_SI * M_SUN_KG * mass / float(a0_m_s2)) / KPC_M


def solar_fractional_extra(
    radius_m,
    *,
    residence_strength: float,
    alpha: float,
    apogee_ratio: float,
    gate_mode: str,
    scale_mode: str,
    screen_a0_m_s2: float,
    screen_exponent: float,
    scale_mix: float = 0.0,
) -> np.ndarray:
    """Worst-case isolated-Sun fractional extra force for one arc law."""
    radius = np.asarray(radius_m, dtype=float)
    gbar = G_SI * M_SUN_KG / np.square(radius)
    if scale_mode == "baryon_r80":
        scale_kpc = 0.8 * 6.957e8 / KPC_M
    elif scale_mode == "mass_radius":
        scale_kpc = float(mass_radius_kpc(1.0, a0_m_s2=screen_a0_m_s2))
    elif scale_mode == "fixed_200kpc":
        scale_kpc = 200.0
    elif scale_mode == "hybrid_radius":
        if not 0.0 <= float(scale_mix) <= 1.0:
            raise ValueError("scale_mix must be between zero and one")
        baryon_scale = 0.8 * 6.957e8 / KPC_M
        mass_scale = float(mass_radius_kpc(1.0, a0_m_s2=screen_a0_m_s2))
        scale_kpc = baryon_scale ** (1.0 - float(scale_mix)) * mass_scale ** float(scale_mix)
    else:
        raise ValueError(scale_mode)
    # The Solar concentration is not measured on the cluster convention. Use
    # gate=1 as the conservative maximum response rather than claiming a value.
    coordinate = residence_coordinate(
        radius / KPC_M,
        scale_kpc,
        alpha=alpha,
        apogee_ratio=apogee_ratio,
    )
    screen = acceleration_screen(
        gbar, a0_m_s2=screen_a0_m_s2, exponent=screen_exponent
    )
    return float(residence_strength) * coordinate * screen


def mercury_precession_mas_per_century(
    *,
    residence_strength: float,
    alpha: float,
    apogee_ratio: float,
    gate_mode: str,
    scale_mode: str,
    screen_a0_m_s2: float,
    screen_exponent: float,
    scale_mix: float = 0.0,
    quadrature_points: int = 32768,
) -> float:
    """First-order supplementary Mercury precession for the arc law."""
    semimajor_axis_m = 0.38709893 * AU_M
    eccentricity = 0.205630
    orbital_period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, int(quadrature_points), endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    radius = semimajor_axis_m * one_minus_e2 / (1.0 + eccentricity * cosine)
    fraction = solar_fractional_extra(
        radius,
        residence_strength=residence_strength,
        alpha=alpha,
        apogee_ratio=apogee_ratio,
        gate_mode=gate_mode,
        scale_mode=scale_mode,
        screen_a0_m_s2=screen_a0_m_s2,
        screen_exponent=screen_exponent,
        scale_mix=scale_mix,
    )
    radial_perturbation = -(G_SI * M_SUN_KG / np.square(radius)) * fraction
    time_weight = one_minus_e2**1.5 / np.square(1.0 + eccentricity * cosine)
    mean_r_cosine = float(np.mean(radial_perturbation * cosine * time_weight))
    period_seconds = orbital_period_days * 86400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = (
        -math.sqrt(one_minus_e2)
        / (mean_motion * semimajor_axis_m * eccentricity)
        * mean_r_cosine
    )
    radians_per_orbit = mean_rate * period_seconds
    orbits_per_century = 100.0 * JULIAN_YEAR_DAYS / orbital_period_days
    return radians_per_orbit * orbits_per_century * RAD_TO_MAS


def solar_diagnostics(**parameters) -> dict[str, float | bool]:
    """Return Solar force-fraction and Mercury proxy diagnostics."""
    solar_radius_m = 6.957e8
    radius = np.geomspace(1.6 * solar_radius_m, 8.43 * AU_M, 1000)
    fraction = solar_fractional_extra(radius, **parameters)
    earth = float(np.interp(AU_M, radius, fraction))
    saturn = float(np.interp(8.43 * AU_M, radius, fraction))
    mercury = mercury_precession_mas_per_century(**parameters)
    return {
        "maximum_fractional_change_limb_to_Saturn": float(np.max(fraction)),
        "Earth_orbit_fractional_change": earth,
        "Saturn_orbit_fractional_change": saturn,
        "Mercury_precession_mas_per_century": mercury,
        "Cassini_proxy_pass": bool(np.max(fraction) <= 2.3e-5),
        "Earth_proxy_pass": bool(earth <= 1.0e-10),
        "Mercury_proxy_pass": bool(abs(mercury) <= 3.1),
    }
