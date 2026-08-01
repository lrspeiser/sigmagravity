"""Thin-lens deflections for resolved cluster-member geometry controls."""

from __future__ import annotations

import numpy as np

from voidscreen.raw_lensing import C_M_S, RAD_TO_ARCSEC

G_M3_KG_S2 = 6.67430e-11
M_SUN_KG = 1.988409870698051e30


def point_mass_einstein_radius_squared_arcsec2(
    mass_msun,
    *,
    lens_angular_distance_m: float,
    distance_ratio: float,
) -> np.ndarray:
    """Return the squared Einstein angle for lens-plane point masses."""
    mass = np.asarray(mass_msun, dtype=float)
    if np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("mass_msun must be finite and nonnegative")
    if not np.isfinite(lens_angular_distance_m) or lens_angular_distance_m <= 0.0:
        raise ValueError("lens_angular_distance_m must be finite and positive")
    if not np.isfinite(distance_ratio) or distance_ratio <= 0.0:
        raise ValueError("distance_ratio must be finite and positive")
    theta_squared_radians = (
        4.0
        * G_M3_KG_S2
        * mass
        * M_SUN_KG
        * float(distance_ratio)
        / (C_M_S**2 * float(lens_angular_distance_m))
    )
    return theta_squared_radians * RAD_TO_ARCSEC**2


def softened_member_deflection(
    x_arcsec,
    y_arcsec,
    member_x_arcsec,
    member_y_arcsec,
    theta_e_squared_arcsec2,
    softening_arcsec,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum circular Plummer deflections from resolved member galaxies."""
    x = np.atleast_1d(np.asarray(x_arcsec, dtype=float))
    y = np.atleast_1d(np.asarray(y_arcsec, dtype=float))
    member_x = np.atleast_1d(np.asarray(member_x_arcsec, dtype=float))
    member_y = np.atleast_1d(np.asarray(member_y_arcsec, dtype=float))
    strength = np.atleast_1d(np.asarray(theta_e_squared_arcsec2, dtype=float))
    softening = np.atleast_1d(np.asarray(softening_arcsec, dtype=float))
    if x.shape != y.shape:
        raise ValueError("x_arcsec and y_arcsec must have matching shapes")
    if not (member_x.shape == member_y.shape == strength.shape == softening.shape):
        raise ValueError("member vectors must have matching shapes")
    if np.any(softening <= 0.0) or np.any(strength < 0.0):
        raise ValueError("softening must be positive and strength nonnegative")
    dx = x[:, None] - member_x[None, :]
    dy = y[:, None] - member_y[None, :]
    denominator = dx**2 + dy**2 + softening[None, :] ** 2
    return (
        np.sum(strength[None, :] * dx / denominator, axis=1),
        np.sum(strength[None, :] * dy / denominator, axis=1),
    )


def circularized_member_deflection(
    x_arcsec,
    y_arcsec,
    member_radius_arcsec,
    theta_e_squared_arcsec2,
    softening_arcsec,
) -> tuple[np.ndarray, np.ndarray]:
    """Deflection after azimuthally spreading each softened member at fixed radius.

    This is the analytic angular average of a circular Plummer deflector.  It
    retains every member's mass, cluster-centric radius, and softening while
    deleting only its position angle.
    """
    x = np.atleast_1d(np.asarray(x_arcsec, dtype=float))
    y = np.atleast_1d(np.asarray(y_arcsec, dtype=float))
    radius = np.atleast_1d(np.asarray(member_radius_arcsec, dtype=float))
    strength = np.atleast_1d(np.asarray(theta_e_squared_arcsec2, dtype=float))
    softening = np.atleast_1d(np.asarray(softening_arcsec, dtype=float))
    if x.shape != y.shape:
        raise ValueError("x_arcsec and y_arcsec must have matching shapes")
    if not (radius.shape == strength.shape == softening.shape):
        raise ValueError("member vectors must have matching shapes")
    if np.any(radius < 0.0) or np.any(softening <= 0.0) or np.any(strength < 0.0):
        raise ValueError("member radii and strengths must be nonnegative and softening positive")

    image_radius = np.hypot(x, y)
    safe_image_radius = np.maximum(image_radius, 1.0e-12)
    image_squared = safe_image_radius[:, None] ** 2
    member_squared = radius[None, :] ** 2
    softening_squared = softening[None, :] ** 2
    discriminant = (
        (image_squared + member_squared + softening_squared) ** 2
        - 4.0 * image_squared * member_squared
    )
    root = np.sqrt(np.maximum(discriminant, 0.0))
    radial_kernel = (
        1.0 + (image_squared - member_squared - softening_squared) / root
    ) / (2.0 * safe_image_radius[:, None])
    radial_alpha = np.sum(strength[None, :] * radial_kernel, axis=1)
    alpha_x = radial_alpha * x / safe_image_radius
    alpha_y = radial_alpha * y / safe_image_radius
    at_origin = image_radius <= 1.0e-12
    alpha_x[at_origin] = 0.0
    alpha_y[at_origin] = 0.0
    return alpha_x, alpha_y


def member_geometry_delta_deflection(
    x_arcsec,
    y_arcsec,
    member_x_arcsec,
    member_y_arcsec,
    theta_e_squared_arcsec2,
    softening_arcsec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clumpy-minus-circularized deflection at fixed radial mass profile."""
    member_x = np.atleast_1d(np.asarray(member_x_arcsec, dtype=float))
    member_y = np.atleast_1d(np.asarray(member_y_arcsec, dtype=float))
    resolved_x, resolved_y = softened_member_deflection(
        x_arcsec,
        y_arcsec,
        member_x,
        member_y,
        theta_e_squared_arcsec2,
        softening_arcsec,
    )
    smooth_x, smooth_y = circularized_member_deflection(
        x_arcsec,
        y_arcsec,
        np.hypot(member_x, member_y),
        theta_e_squared_arcsec2,
        softening_arcsec,
    )
    return resolved_x - smooth_x, resolved_y - smooth_y
