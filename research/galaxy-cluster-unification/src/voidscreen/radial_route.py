"""Conservative radial remaps of an already-computed gravity response.

This module does not add an acceleration amplitude.  It treats the extra
acceleration times radius squared as an effective enclosed flux, moves a
declared fraction of that flux radially, and converts it back to acceleration.
It is a phenomenological test operator, not a covariant field equation.
"""

from __future__ import annotations

import math

import numpy as np

from .arc_apogee import (
    AU_M,
    G_SI,
    JULIAN_YEAR_DAYS,
    M_SUN_KG,
    RAD_TO_MAS,
)
from .arc_invariants import generalized_arc_response
from .data import KPC_M
from .raw_lensing import loglog_interpolate_with_tails


C_M_S = 299_792_458.0
SOLAR_RADIUS_M = 6.957e8


def remap_extra_acceleration(
    radius,
    extra_acceleration,
    *,
    route_fraction: float,
    radial_scale,
) -> np.ndarray:
    """Radially remap an extra acceleration without fitting its amplitude.

    If ``M_X(<r) = a_X(r) r^2`` is the effective enclosed response, the
    returned acceleration is

    ``[(1-f) M_X(<r) + f M_X(<r/lambda)] / r^2``.

    ``lambda < 1`` compresses the response inward and ``lambda > 1`` expands
    it outward.  Either ``f=0`` or ``lambda=1`` reproduces the input exactly.
    The omitted factor of G cancels between the two sides.
    """
    radius = np.asarray(radius, dtype=float)
    extra = np.asarray(extra_acceleration, dtype=float)
    if radius.ndim != 1 or extra.ndim != 1 or radius.shape != extra.shape:
        raise ValueError("radius and extra_acceleration must be matching vectors")
    if len(radius) < 2 or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("at least two finite positive radii are required")
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be strictly increasing")
    if np.any(~np.isfinite(extra)) or np.any(extra < 0.0):
        raise ValueError("extra_acceleration must be finite and non-negative")
    if not math.isfinite(route_fraction) or not 0.0 <= route_fraction <= 1.0:
        raise ValueError("route_fraction must lie in [0, 1]")
    scale = np.asarray(radial_scale, dtype=float)
    try:
        scale = np.broadcast_to(scale, radius.shape)
    except ValueError as error:
        raise ValueError("radial_scale must be scalar or match the radius vector") from error
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("radial_scale must be finite and positive")
    if route_fraction == 0.0 or np.all(scale == 1.0) or not np.any(extra > 0.0):
        return extra.copy()
    enclosed_flux = extra * np.square(radius)
    positive = enclosed_flux > 0.0
    target_radius = radius / scale
    if np.all(positive):
        shifted_flux = loglog_interpolate_with_tails(
            target_radius, radius, enclosed_flux
        )
    elif np.count_nonzero(positive) >= 2:
        # Exact screening can underflow to zero in strong fields.  Keep targets
        # below the first representable positive response at zero and perform
        # the logarithmic remap only on the positive part of the profile.
        shifted_flux = np.zeros_like(enclosed_flux)
        supported = target_radius >= radius[positive][0]
        shifted_flux[supported] = loglog_interpolate_with_tails(
            target_radius[supported], radius[positive], enclosed_flux[positive]
        )
    else:
        # A single isolated positive sample cannot define a logarithmic slope.
        shifted_flux = np.interp(target_radius, radius, enclosed_flux, left=0.0, right=float(enclosed_flux[-1]))
    mixed_flux = (
        (1.0 - float(route_fraction)) * enclosed_flux
        + float(route_fraction) * shifted_flux
    )
    return mixed_flux / np.square(radius)


def remap_total_acceleration(
    radius,
    baryonic_acceleration,
    parent_total_acceleration,
    *,
    route_fraction: float,
    radial_scale,
) -> np.ndarray:
    """Return baryonic acceleration plus the radially remapped extra channel."""
    gbar = np.asarray(baryonic_acceleration, dtype=float)
    parent = np.asarray(parent_total_acceleration, dtype=float)
    if gbar.shape != parent.shape:
        raise ValueError("baryonic and parent accelerations must have matching shapes")
    extra = parent - gbar
    tolerance = 1.0e-12 * np.maximum(np.abs(parent), np.abs(gbar))
    if np.any(extra < -tolerance):
        raise ValueError("parent acceleration cannot be below the baryonic acceleration")
    extra = np.maximum(extra, 0.0)
    return gbar + remap_extra_acceleration(
        radius,
        extra,
        route_fraction=route_fraction,
        radial_scale=radial_scale,
    )


def potential_transition_scale(
    potential_depth,
    *,
    log_scale_amplitude: float,
    pivot: float = 2.0e-6,
    sharpness: float = 1.0,
) -> np.ndarray:
    """Return a smooth local radial scale selected only by baryonic potential.

    ``ln(lambda) = A tanh[k ln(Phi_b/Phi_*)]``.  The parent is exactly
    recovered at ``A=0``; the scale stays between ``exp(-|A|)`` and
    ``exp(|A|)`` without naming a galaxy or cluster domain.
    """
    depth = np.asarray(potential_depth, dtype=float)
    if np.any(~np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("potential_depth must be finite and positive")
    if not math.isfinite(log_scale_amplitude):
        raise ValueError("log_scale_amplitude must be finite")
    if not math.isfinite(pivot) or pivot <= 0.0:
        raise ValueError("pivot must be finite and positive")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("sharpness must be finite and positive")
    signed_regime = np.tanh(float(sharpness) * np.log(depth / float(pivot)))
    return np.exp(float(log_scale_amplitude) * signed_regime)


def remapped_solar_diagnostics(
    *,
    response_parameters: dict,
    route_fraction: float,
    radial_scale=1.0,
    potential_log_scale_amplitude: float | None = None,
    potential_pivot: float = 2.0e-6,
    potential_sharpness: float = 1.0,
    grid_points: int = 20000,
) -> dict[str, float | bool]:
    """Evaluate the same radial remap on an isolated point-mass Sun."""
    if grid_points < 2000:
        raise ValueError("grid_points must be at least 2000")
    radius = np.geomspace(0.05 * SOLAR_RADIUS_M, 30.0 * AU_M, int(grid_points))
    gbar = G_SI * M_SUN_KG / np.square(radius)
    potential_depth = G_SI * M_SUN_KG / radius / C_M_S**2
    response = generalized_arc_response(
        gbar,
        radius / KPC_M,
        np.ones_like(radius),
        np.ones_like(radius),
        potential_depth=potential_depth,
        potential_length_kpc=radius / KPC_M,
        potential_path_ratio=np.ones_like(radius),
        enclosed_mass_log_slope=np.zeros_like(radius),
        **response_parameters,
    )
    parent_dynamic = gbar * response["dynamical_enhancement"]
    parent_lensing = gbar * response["lensing_enhancement"]
    scale = radial_scale
    if potential_log_scale_amplitude is not None:
        scale = potential_transition_scale(
            potential_depth,
            log_scale_amplitude=float(potential_log_scale_amplitude),
            pivot=float(potential_pivot),
            sharpness=float(potential_sharpness),
        )
    dynamic = remap_total_acceleration(
        radius,
        gbar,
        parent_dynamic,
        route_fraction=route_fraction,
        radial_scale=scale,
    )
    lensing = remap_total_acceleration(
        radius,
        gbar,
        parent_lensing,
        route_fraction=route_fraction,
        radial_scale=scale,
    )
    dynamic_fraction = dynamic / gbar - 1.0
    lensing_fraction = lensing / gbar - 1.0

    def lookup(values, target):
        return np.interp(np.log(np.asarray(target, dtype=float)), np.log(radius), values)

    diagnostic = (radius >= 1.6 * SOLAR_RADIUS_M) & (radius <= 8.43 * AU_M)
    earth = float(lookup(dynamic_fraction, AU_M))
    saturn = float(lookup(dynamic_fraction, 8.43 * AU_M))

    semimajor = 0.38709893 * AU_M
    eccentricity = 0.205630
    period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, 32768, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    mercury_radius = semimajor * one_minus_e2 / (1.0 + eccentricity * cosine)
    mercury_fraction = lookup(dynamic_fraction, mercury_radius)
    radial_perturbation = -(
        G_SI * M_SUN_KG / np.square(mercury_radius)
    ) * mercury_fraction
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
    maximum_dynamic = float(np.max(dynamic_fraction[diagnostic]))
    maximum_lensing = float(np.max(lensing_fraction[diagnostic]))
    return {
        "maximum_dynamic_fraction_limb_to_Saturn": maximum_dynamic,
        "maximum_lensing_fraction_limb_to_Saturn": maximum_lensing,
        "Earth_orbit_fractional_change": earth,
        "Saturn_orbit_fractional_change": saturn,
        "Mercury_precession_mas_per_century": float(mercury),
        "Cassini_proxy_pass": bool(maximum_lensing <= 2.3e-5),
        "Earth_proxy_pass": bool(earth <= 1.0e-10),
        "Mercury_proxy_pass": bool(abs(mercury) <= 3.1),
    }
