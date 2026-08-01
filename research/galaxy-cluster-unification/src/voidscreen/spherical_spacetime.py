"""Closed-space flux and excluded-sphere flow diagnostics."""

from __future__ import annotations

import math

import numpy as np

from .data import KPC_M


C_M_S = 299_792_458.0
R_SUN_M = 6.957e8


def closed_sphere_area_enhancement(x, *, maximum_x: float = 0.95 * math.pi) -> np.ndarray:
    """Return the Gauss-flux factor ``(x/sin(x))^2`` on a closed 3-space."""
    value = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("x must be finite and nonnegative")
    if np.any(value >= maximum_x):
        raise ValueError("radius reaches the closed-space antipodal domain")
    result = np.empty_like(value)
    small = value < 1.0e-4
    squared = np.square(value[small])
    result[small] = 1.0 + squared / 3.0 + np.square(squared) / 15.0
    result[~small] = np.square(value[~small] / np.sin(value[~small]))
    return result


def global_closed_acceleration(
    gbar_m_s2, radius_kpc, curvature_radius_kpc: float, *, maximum_x: float
) -> np.ndarray:
    """Apply the exact constant-positive-curvature spherical area law."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(curvature_radius_kpc) or curvature_radius_kpc <= 0.0:
        raise ValueError("curvature radius must be finite and positive")
    return gbar * closed_sphere_area_enhancement(
        radius / float(curvature_radius_kpc), maximum_x=maximum_x
    )


def local_mass_curvature_acceleration(
    gbar_m_s2,
    radius_kpc,
    curvature_multiplier: float,
    *,
    acceleration_screen_m_s2: float | None = None,
    screen_power: float = 1.0,
    maximum_x: float = 0.95 * math.pi,
) -> np.ndarray:
    """Apply extra local closed curvature relative to the GR-strength baseline."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(curvature_multiplier) or curvature_multiplier < 0.0:
        raise ValueError("curvature multiplier must be finite and nonnegative")
    weight = np.ones_like(gbar)
    if acceleration_screen_m_s2 is not None:
        if (
            not math.isfinite(acceleration_screen_m_s2)
            or acceleration_screen_m_s2 <= 0.0
            or not math.isfinite(screen_power)
            or screen_power <= 0.0
        ):
            raise ValueError("screen scale and power must be finite and positive")
        log_ratio = float(screen_power) * (
            np.log(gbar) - math.log(float(acceleration_screen_m_s2))
        )
        weight = np.exp(-np.logaddexp(0.0, log_ratio))
    potential = gbar * radius * KPC_M / C_M_S**2
    baseline_x = np.sqrt(potential)
    total_x = np.sqrt(potential * (1.0 + float(curvature_multiplier) * weight))
    baseline = closed_sphere_area_enhancement(baseline_x, maximum_x=maximum_x)
    total = closed_sphere_area_enhancement(total_x, maximum_x=maximum_x)
    return gbar * total / baseline


def hard_cavity_flow_components(theta_radian, cavity_radius_over_radius) -> tuple[np.ndarray, np.ndarray]:
    """Return exact radial and polar velocity ratios around an impermeable sphere."""
    theta = np.asarray(theta_radian, dtype=float)
    ratio = np.asarray(cavity_radius_over_radius, dtype=float)
    theta, ratio = np.broadcast_arrays(theta, ratio)
    if np.any(~np.isfinite(theta)) or np.any(~np.isfinite(ratio)):
        raise ValueError("theta and radius ratio must be finite")
    if np.any(ratio < 0.0) or np.any(ratio > 1.0):
        raise ValueError("cavity radius over field radius must be in [0,1]")
    q = np.power(ratio, 3.0)
    radial = (1.0 - q) * np.cos(theta)
    polar = -(1.0 + 0.5 * q) * np.sin(theta)
    return radial, polar


def hard_cavity_isotropic_rms_enhancement(cavity_radius_over_radius) -> np.ndarray:
    """Return the solid-angle RMS field factor; the linear dipole cancels."""
    ratio = np.asarray(cavity_radius_over_radius, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0) or np.any(ratio > 1.0):
        raise ValueError("cavity radius over field radius must be in [0,1]")
    return np.sqrt(1.0 + 0.5 * np.power(ratio, 6.0))


def hard_cavity_best_axis_enhancement(cavity_radius_over_radius) -> np.ndarray:
    """Return the most favorable tangential-axis field factor."""
    ratio = np.asarray(cavity_radius_over_radius, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0) or np.any(ratio > 1.0):
        raise ValueError("cavity radius over field radius must be in [0,1]")
    return 1.0 + 0.5 * np.power(ratio, 3.0)


def stellar_area_covering_fraction(stellar_mass_solar, disk_radius_kpc) -> np.ndarray:
    """Upper-bound the projected covering fraction if every solar mass is a Sun."""
    mass = np.asarray(stellar_mass_solar, dtype=float)
    radius = np.asarray(disk_radius_kpc, dtype=float)
    mass, radius = np.broadcast_arrays(mass, radius)
    if np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("stellar mass must be finite and nonnegative")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("disk radius must be finite and positive")
    return mass * np.square(R_SUN_M / (radius * KPC_M))
