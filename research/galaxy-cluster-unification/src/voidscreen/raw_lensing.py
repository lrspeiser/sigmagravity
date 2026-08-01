"""Raw strong-lensing helpers for fixed spherical acceleration laws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_M_S = 299_792_458.0
KPC_M = 3.085677581491367e19
RAD_TO_ARCSEC = 206_264.80624709636


def finite_ratio_or_none(numerator: float, denominator: float) -> float | None:
    """Return a finite comparison ratio, or ``None`` for an undefined score."""
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return None
    return numerator / denominator


def loglog_interpolate_with_tails(
    radius,
    anchor_radius,
    anchor_values,
    *,
    inner_slope: float | None = None,
    outer_slope: float | None = None,
) -> np.ndarray:
    """Interpolate positive radial data in log-log space with explicit tails."""
    target = np.asarray(radius, dtype=np.float64)
    anchors = np.asarray(anchor_radius, dtype=np.float64)
    values = np.asarray(anchor_values, dtype=np.float64)
    if np.any(target <= 0.0) or np.any(anchors <= 0.0) or np.any(values <= 0.0):
        raise ValueError("radii and values must be positive")
    if anchors.ndim != 1 or values.ndim != 1 or len(anchors) != len(values):
        raise ValueError("anchor_radius and anchor_values must be matching vectors")
    if len(anchors) < 2 or np.any(np.diff(anchors) <= 0.0):
        raise ValueError("at least two strictly increasing anchors are required")

    log_target = np.log10(target)
    log_radius = np.log10(anchors)
    log_values = np.log10(values)
    result = np.interp(log_target, log_radius, log_values)
    slope_inner = (
        float(inner_slope)
        if inner_slope is not None
        else float((log_values[1] - log_values[0]) / (log_radius[1] - log_radius[0]))
    )
    slope_outer = (
        float(outer_slope)
        if outer_slope is not None
        else float((log_values[-1] - log_values[-2]) / (log_radius[-1] - log_radius[-2]))
    )
    left = log_target < log_radius[0]
    right = log_target > log_radius[-1]
    result[left] = log_values[0] + slope_inner * (log_target[left] - log_radius[0])
    result[right] = log_values[-1] + slope_outer * (log_target[right] - log_radius[-1])
    return np.power(10.0, result)


def spherical_deflection_radians(
    impact_kpc,
    acceleration,
    *,
    maximum_radius_kpc: float = 1.0e6,
    integration_points: int = 800,
) -> np.ndarray:
    """Integrate a spherical zero-slip acceleration into physical deflection.

    ``acceleration`` is called with radii in kpc and must return m/s^2.  The
    hyperbolic substitution r=b*cosh(t) avoids the line-of-sight singularity.
    """
    impact = np.atleast_1d(np.asarray(impact_kpc, dtype=np.float64))
    if np.any(~np.isfinite(impact)) or np.any(impact <= 0.0):
        raise ValueError("impact_kpc must be finite and positive")
    if maximum_radius_kpc <= float(np.max(impact)):
        raise ValueError("maximum_radius_kpc must exceed every impact parameter")
    if integration_points < 64:
        raise ValueError("integration_points must be at least 64")

    output = np.empty_like(impact)
    for index, value in enumerate(impact):
        t_max = float(np.arccosh(maximum_radius_kpc / value))
        t = np.linspace(0.0, t_max, integration_points)
        radius = value * np.cosh(t)
        g = np.asarray(acceleration(radius), dtype=np.float64)
        if g.shape != radius.shape or np.any(~np.isfinite(g)) or np.any(g <= 0.0):
            raise ValueError("acceleration must return finite positive values")
        integral = float(np.trapezoid(g, t))
        output[index] = 4.0 * value * KPC_M * integral / C_M_S**2
    return output


@dataclass(frozen=True)
class RadialDeflectionField:
    """Interpolated physical deflection for one fixed radial acceleration law."""

    impact_arcsec: np.ndarray
    physical_deflection_radians: np.ndarray

    def __post_init__(self) -> None:
        impact = np.asarray(self.impact_arcsec, dtype=np.float64)
        alpha = np.asarray(self.physical_deflection_radians, dtype=np.float64)
        if impact.ndim != 1 or alpha.ndim != 1 or len(impact) != len(alpha):
            raise ValueError("impact and deflection must be matching vectors")
        if np.any(impact <= 0.0) or np.any(alpha < 0.0) or np.any(np.diff(impact) <= 0.0):
            raise ValueError("impact must increase and deflection must be nonnegative")

    def reduced_alpha_arcsec(self, impact_arcsec, distance_ratio: float) -> np.ndarray:
        radius = np.asarray(impact_arcsec, dtype=np.float64)
        if not np.isfinite(distance_ratio) or distance_ratio <= 0.0:
            raise ValueError("distance_ratio must be finite and positive")
        clipped = np.maximum(radius, self.impact_arcsec[0])
        alpha = np.interp(
            np.log(clipped),
            np.log(self.impact_arcsec),
            self.physical_deflection_radians,
        )
        return alpha * float(distance_ratio) * RAD_TO_ARCSEC


def pseudo_elliptical_deflection(
    x_arcsec,
    y_arcsec,
    radial_alpha,
    *,
    axis_ratio: float,
    phi_radian: float,
    center_x_arcsec: float,
    center_y_arcsec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a circular deflection curve into an area-preserving elliptical potential."""
    if not 0.0 < axis_ratio <= 1.0:
        raise ValueError("axis_ratio must lie in (0, 1]")
    x = np.asarray(x_arcsec, dtype=np.float64) - center_x_arcsec
    y = np.asarray(y_arcsec, dtype=np.float64) - center_y_arcsec
    cosine = np.cos(phi_radian)
    sine = np.sin(phi_radian)
    x_prime = cosine * x + sine * y
    y_prime = -sine * x + cosine * y
    radius = np.sqrt(axis_ratio * x_prime**2 + y_prime**2 / axis_ratio)
    safe_radius = np.maximum(radius, 1.0e-9)
    magnitude = np.asarray(radial_alpha(safe_radius), dtype=np.float64)
    alpha_x_prime = magnitude * axis_ratio * x_prime / safe_radius
    alpha_y_prime = magnitude * y_prime / (axis_ratio * safe_radius)
    alpha_x = cosine * alpha_x_prime - sine * alpha_y_prime
    alpha_y = sine * alpha_x_prime + cosine * alpha_y_prime
    return alpha_x, alpha_y


def shear_deflection(
    x_arcsec,
    y_arcsec,
    gamma1: float,
    gamma2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deflection from a constant external shear potential."""
    x = np.asarray(x_arcsec, dtype=np.float64)
    y = np.asarray(y_arcsec, dtype=np.float64)
    return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y
