"""Universal local invariants for the arc-apogee phenomenology."""

from __future__ import annotations

import math

import numpy as np

from .arc_apogee import (
    AU_M,
    G_SI,
    JULIAN_YEAR_DAYS,
    M_SUN_KG,
    RAD_TO_MAS,
    acceleration_screen,
    extent_gate,
    mass_radius_kpc,
    residence_coordinate,
)
from .data import KPC_M


C_M_S = 299_792_458.0
MASS_PIVOT_SOLAR = 1.0e10


def generalized_add_one(value, softness: float) -> np.ndarray:
    """Return a positive parent-preserving deformation of ``1 + value``.

    The coefficient keeps the midpoint fixed: value=1 maps to 2 for every
    softness, while softness=1 recovers exactly 1+value.
    """
    value = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("generalized-addition input must be finite and non-negative")
    if not math.isfinite(softness) or softness <= 0.0:
        raise ValueError("generalized-addition softness must be finite and positive")
    if float(softness) == 1.0:
        return 1.0 + value
    coefficient = np.power(2.0, float(softness)) - 1.0
    return np.power(
        1.0 + coefficient * np.power(value, float(softness)),
        1.0 / float(softness),
    )


def generalized_screen(gbar_m_s2, *, a0_m_s2: float, exponent: float, softness: float):
    """Return a midpoint- and asymptote-preserving deformation of the screen."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if a0_m_s2 <= 0.0 or exponent <= 0.0 or softness <= 0.0:
        raise ValueError("screen constants must be positive")
    if float(softness) == 1.0:
        return acceleration_screen(gbar, a0_m_s2=a0_m_s2, exponent=exponent)
    coefficient = np.power(2.0, float(softness)) - 1.0
    with np.errstate(over="ignore"):
        return np.power(
            1.0
            + coefficient
            * np.power(gbar / float(a0_m_s2), float(exponent) * float(softness)),
            -1.0 / float(softness),
        )


def generalized_residence_coordinate(
    radius,
    scale_radius,
    *,
    alpha: float,
    apogee_ratio: float,
    softness: float,
) -> np.ndarray:
    """Return a soft minimum of the unsaturated path and its apogee."""
    radius = np.asarray(radius, dtype=float)
    scale = np.asarray(scale_radius, dtype=float)
    radius, scale = np.broadcast_arrays(radius, scale)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale radius must be finite and positive")
    if alpha <= 0.0 or apogee_ratio <= 0.0 or softness <= 0.0:
        raise ValueError("residence constants must be positive")
    if float(softness) == 1.0:
        return residence_coordinate(
            radius, scale, alpha=alpha, apogee_ratio=apogee_ratio
        )
    path = np.power(radius / scale, float(alpha))
    apogee = np.full_like(path, float(apogee_ratio) ** float(alpha))
    k = float(softness)
    with np.errstate(over="ignore", under="ignore"):
        return np.power(np.power(path, -k) + np.power(apogee, -k), -1.0 / k)


def spherical_profile_invariants(radius_kpc, gbar_m_s2) -> dict[str, np.ndarray]:
    """Return potential, path-length, and enclosed-mass-growth invariants.

    The potential is integrated to the final measured radius and closed with a
    point-mass tail. This is a declared spherical diagnostic, not a claim that
    disks or clusters are spherical.
    """
    radius = np.asarray(radius_kpc, dtype=float)
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if radius.ndim != 1 or gbar.ndim != 1 or radius.shape != gbar.shape:
        raise ValueError("radius and gbar must be matching one-dimensional arrays")
    if len(radius) < 2 or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("at least two finite positive radii are required")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    order = np.argsort(radius, kind="stable")
    sorted_radius = radius[order]
    sorted_gbar = gbar[order]
    if np.any(np.diff(sorted_radius) <= 0.0):
        raise ValueError("radii must be unique")
    radius_m = sorted_radius * KPC_M
    # Integrate the log-linear power law on each interval exactly. This keeps a
    # sampled point-mass profile at Phi=GM/r rather than introducing a grid-
    # spacing-dependent trapezoid bias into the path-ratio invariant.
    radius_ratio = radius_m[1:] / radius_m[:-1]
    slopes = np.log(sorted_gbar[1:] / sorted_gbar[:-1]) / np.log(radius_ratio)
    near_minus_one = np.isclose(slopes, -1.0, atol=1.0e-12, rtol=0.0)
    integral_factor = np.empty_like(slopes)
    integral_factor[near_minus_one] = np.log(radius_ratio[near_minus_one])
    regular = ~near_minus_one
    integral_factor[regular] = (
        np.power(radius_ratio[regular], slopes[regular] + 1.0) - 1.0
    ) / (slopes[regular] + 1.0)
    segments = sorted_gbar[:-1] * radius_m[:-1] * integral_factor
    potential = np.zeros_like(sorted_gbar)
    potential[:-1] = np.cumsum(segments[::-1])[::-1]
    potential += sorted_gbar[-1] * radius_m[-1]
    ell_kpc = potential / sorted_gbar / KPC_M
    path_ratio = ell_kpc / sorted_radius
    edge_order = 2 if len(radius) >= 3 else 1
    enclosed_mass_proxy = sorted_gbar * np.square(radius_m)
    mass_growth = np.gradient(
        np.log(enclosed_mass_proxy), np.log(sorted_radius), edge_order=edge_order
    )
    output = {}
    for name, values in {
        "potential_m2_s2": potential,
        "potential_depth": potential / C_M_S**2,
        "potential_length_kpc": ell_kpc,
        "potential_path_ratio": path_ratio,
        "enclosed_mass_log_slope": mass_growth,
    }.items():
        restored = np.empty_like(values)
        restored[order] = values
        output[name] = restored
    return output


def invariant_multiplier(
    mode: str,
    *,
    potential_depth,
    potential_length_kpc,
    potential_path_ratio,
    enclosed_mass_log_slope,
    power: float,
    scale: float,
) -> np.ndarray:
    """Return a positive multiplier from one declared field invariant."""
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("power must be finite and non-negative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    depth, length, ratio, growth = np.broadcast_arrays(
        np.asarray(potential_depth, dtype=float),
        np.asarray(potential_length_kpc, dtype=float),
        np.asarray(potential_path_ratio, dtype=float),
        np.asarray(enclosed_mass_log_slope, dtype=float),
    )
    if any(np.any(~np.isfinite(item)) for item in (depth, length, ratio, growth)):
        raise ValueError("invariants must be finite")
    if mode == "none" or power == 0.0:
        return np.ones_like(depth)
    if mode == "path_ratio":
        return np.power(np.clip(ratio, 0.25, 100.0), power)
    if mode == "mass_growth":
        return np.power(1.0 + np.clip(growth, 0.0, 3.0), power)
    if mode == "coherence_length":
        return 1.0 + np.power(np.maximum(length, 0.0) / scale, power)
    if mode == "potential_depth":
        return 1.0 + np.power(np.maximum(depth, 0.0) / scale, power)
    raise ValueError(f"unknown invariant mode: {mode}")


def generalized_arc_response(
    gbar_m_s2,
    radius_kpc,
    total_baryonic_mass_solar,
    concentration,
    *,
    residence_strength: float,
    alpha: float,
    apogee_ratio: float,
    screen_a0_m_s2: float,
    screen_exponent: float,
    screen_scale: float = 1.0,
    mass_radius_delta: float = 0.0,
    extent_leak: float = 0.0,
    invariant_mode: str = "none",
    invariant_power: float = 0.0,
    invariant_scale: float = 1.0,
    secondary_path_ratio_power: float = 0.0,
    potential_depth=0.0,
    potential_length_kpc=0.0,
    potential_path_ratio=1.0,
    enclosed_mass_log_slope=0.0,
    photon_extra_multiplier: float = 1.0,
    residence_softness: float = 1.0,
    screen_softness: float = 1.0,
    potential_softness: float = 1.0,
    potential_path_cross: float = 1.0,
    response_addition_softness: float = 1.0,
    lensing_addition_softness: float = 1.0,
    extent_scale_coupling: float = 0.0,
    potential_scale_coupling: float = 0.0,
    mass_growth_power: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return dynamical and zero/background-plus-slip lens responses.

    ``photon_extra_multiplier`` multiplies only the new channel:
    g_lens/gbar = 1 + photon_extra_multiplier * (g_dyn/gbar - 1).
    """
    if residence_strength < 0.0 or screen_scale <= 0.0:
        raise ValueError("strength must be non-negative and screen scale positive")
    if photon_extra_multiplier <= 0.0:
        raise ValueError("photon multiplier must be positive")
    for name, value in {
        "residence_softness": residence_softness,
        "screen_softness": screen_softness,
        "potential_softness": potential_softness,
        "response_addition_softness": response_addition_softness,
        "lensing_addition_softness": lensing_addition_softness,
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in {
        "potential_path_cross": potential_path_cross,
        "extent_scale_coupling": extent_scale_coupling,
        "potential_scale_coupling": potential_scale_coupling,
        "mass_growth_power": mass_growth_power,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    mass = np.asarray(total_baryonic_mass_solar, dtype=float)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("total baryonic mass must be finite and positive")
    potential_base = invariant_multiplier(
        invariant_mode,
        potential_depth=potential_depth,
        potential_length_kpc=potential_length_kpc,
        potential_path_ratio=potential_path_ratio,
        enclosed_mass_log_slope=enclosed_mass_log_slope,
        power=invariant_power,
        scale=invariant_scale,
    )
    if invariant_mode == "potential_depth" and float(invariant_power) != 0.0:
        potential_argument = np.power(
            np.maximum(np.asarray(potential_depth, dtype=float), 0.0)
            / float(invariant_scale),
            float(invariant_power),
        )
        potential_base = generalized_add_one(
            potential_argument, float(potential_softness)
        )
    soft_extent = extent_gate(concentration, "cluster_logistic_soft")
    scale_radius = mass_radius_kpc(mass, a0_m_s2=screen_a0_m_s2)
    scale_radius *= np.power(mass / MASS_PIVOT_SOLAR, float(mass_radius_delta))
    scale_radius *= np.power(
        np.broadcast_to(soft_extent, np.broadcast(radius_kpc, scale_radius).shape),
        float(extent_scale_coupling),
    )
    scale_radius *= np.power(
        np.broadcast_to(potential_base, np.broadcast(radius_kpc, scale_radius).shape),
        float(potential_scale_coupling),
    )
    coordinate = generalized_residence_coordinate(
        radius_kpc,
        scale_radius,
        alpha=alpha,
        apogee_ratio=apogee_ratio,
        softness=residence_softness,
    )
    screen = generalized_screen(
        gbar_m_s2,
        a0_m_s2=screen_a0_m_s2 * float(screen_scale),
        exponent=screen_exponent,
        softness=screen_softness,
    )
    if float(extent_leak) == 0.0:
        gate = np.ones_like(coordinate)
    else:
        gate = np.power(np.broadcast_to(soft_extent, coordinate.shape), float(extent_leak))
    if not math.isfinite(secondary_path_ratio_power) or secondary_path_ratio_power < 0.0:
        raise ValueError("secondary path-ratio power must be finite and non-negative")
    path_multiplier = np.power(
        np.clip(np.asarray(potential_path_ratio, dtype=float), 0.25, 100.0),
        float(secondary_path_ratio_power),
    )
    if invariant_mode == "potential_depth":
        if float(potential_path_cross) == 1.0:
            invariant = potential_base * path_multiplier
        else:
            invariant = (
                potential_base
                + path_multiplier
                - 1.0
                + float(potential_path_cross)
                * (potential_base - 1.0)
                * (path_multiplier - 1.0)
            )
    else:
        invariant = potential_base * path_multiplier
    growth_multiplier = np.power(
        1.0 + np.clip(np.asarray(enclosed_mass_log_slope, dtype=float), 0.0, 3.0),
        float(mass_growth_power),
    )
    invariant *= growth_multiplier
    invariant = np.broadcast_to(invariant, coordinate.shape)
    if np.any(~np.isfinite(invariant)) or np.any(invariant <= 0.0):
        raise ValueError("structural invariant must remain finite and positive")
    unit_response = gate * coordinate * screen * invariant
    linear_fractional = float(residence_strength) * unit_response
    if float(response_addition_softness) == 1.0:
        fractional = linear_fractional
        dynamical_enhancement = 1.0 + fractional
    else:
        dynamical_enhancement = generalized_add_one(
            linear_fractional, float(response_addition_softness)
        )
        fractional = dynamical_enhancement - 1.0
    if float(lensing_addition_softness) == 1.0:
        lensing_enhancement = 1.0 + float(photon_extra_multiplier) * fractional
    else:
        lensing_enhancement = generalized_add_one(
            float(photon_extra_multiplier) * fractional,
            float(lensing_addition_softness),
        )
    return {
        "unit_fractional_response": unit_response,
        "fractional_dynamical_response": fractional,
        "dynamical_enhancement": dynamical_enhancement,
        "lensing_enhancement": lensing_enhancement,
        "scale_radius_kpc": np.broadcast_to(scale_radius, coordinate.shape),
        "residence_coordinate": coordinate,
        "screen": screen,
        "extent_factor": gate,
        "invariant_multiplier": invariant,
    }


def generalized_solar_diagnostics(**parameters) -> dict[str, float | bool]:
    """Apply the generalized law to an isolated point-mass Sun."""
    solar_radius_m = 6.957e8

    def fraction(radius_m):
        radius = np.asarray(radius_m, dtype=float)
        gbar = G_SI * M_SUN_KG / np.square(radius)
        response = generalized_arc_response(
            gbar,
            radius / KPC_M,
            np.ones_like(radius),
            np.ones_like(radius),
            potential_depth=G_SI * M_SUN_KG / radius / C_M_S**2,
            potential_length_kpc=radius / KPC_M,
            potential_path_ratio=np.ones_like(radius),
            enclosed_mass_log_slope=np.zeros_like(radius),
            **parameters,
        )
        return response["fractional_dynamical_response"]

    radius = np.geomspace(1.6 * solar_radius_m, 8.43 * AU_M, 1000)
    dynamic = fraction(radius)
    photon_multiplier = float(parameters.get("photon_extra_multiplier", 1.0))
    lens = photon_multiplier * dynamic
    earth = float(np.interp(AU_M, radius, dynamic))
    saturn = float(np.interp(8.43 * AU_M, radius, dynamic))

    semimajor = 0.38709893 * AU_M
    eccentricity = 0.205630
    period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, 32768, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    mercury_radius = semimajor * one_minus_e2 / (1.0 + eccentricity * cosine)
    radial_perturbation = -(G_SI * M_SUN_KG / np.square(mercury_radius)) * fraction(
        mercury_radius
    )
    time_weight = one_minus_e2**1.5 / np.square(1.0 + eccentricity * cosine)
    mean_r_cosine = float(np.mean(radial_perturbation * cosine * time_weight))
    period_seconds = period_days * 86400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = (
        -math.sqrt(one_minus_e2)
        / (mean_motion * semimajor * eccentricity)
        * mean_r_cosine
    )
    mercury = (
        mean_rate
        * period_seconds
        * (100.0 * JULIAN_YEAR_DAYS / period_days)
        * RAD_TO_MAS
    )
    return {
        "maximum_dynamic_fraction_limb_to_Saturn": float(np.max(dynamic)),
        "maximum_lensing_fraction_limb_to_Saturn": float(np.max(lens)),
        "Earth_orbit_fractional_change": earth,
        "Saturn_orbit_fractional_change": saturn,
        "Mercury_precession_mas_per_century": float(mercury),
        "Cassini_proxy_pass": bool(np.max(lens) <= 2.3e-5),
        "Earth_proxy_pass": bool(earth <= 1.0e-10),
        "Mercury_proxy_pass": bool(abs(mercury) <= 3.1),
    }
