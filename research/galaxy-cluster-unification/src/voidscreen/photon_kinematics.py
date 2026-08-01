"""Geometry for comparing radial-velocity and proper-motion rotation estimates."""

from __future__ import annotations

import numpy as np


def galactocentric_geometry(
    longitude_rad,
    latitude_rad,
    distance_kpc,
    *,
    solar_radius_kpc: float,
) -> dict[str, np.ndarray]:
    """Return source position and local projection vectors.

    The Galactocentric x-axis points from the Galactic center to the Sun and
    the y-axis follows Galactic rotation.  Galactic longitude zero therefore
    points in the negative x direction.
    """
    longitude = np.asarray(longitude_rad, dtype=float)
    latitude = np.asarray(latitude_rad, dtype=float)
    distance = np.asarray(distance_kpc, dtype=float)
    longitude, latitude, distance = np.broadcast_arrays(
        longitude, latitude, distance
    )
    if np.any(~np.isfinite(longitude)) or np.any(~np.isfinite(latitude)):
        raise ValueError("angles must be finite")
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("distance must be finite and positive")
    if not np.isfinite(solar_radius_kpc) or solar_radius_kpc <= 0.0:
        raise ValueError("solar radius must be finite and positive")

    cos_b = np.cos(latitude)
    sin_b = np.sin(latitude)
    cos_l = np.cos(longitude)
    sin_l = np.sin(longitude)
    x = solar_radius_kpc - distance * cos_b * cos_l
    y = distance * cos_b * sin_l
    z = distance * sin_b
    radius = np.hypot(x, y)
    if np.any(radius <= 0.0):
        raise ValueError("source cannot lie exactly at the Galactic center")

    e_r = np.stack((-cos_b * cos_l, cos_b * sin_l, sin_b), axis=-1)
    e_l = np.stack((sin_l, cos_l, np.zeros_like(sin_l)), axis=-1)
    e_b = np.stack((sin_b * cos_l, -sin_b * sin_l, cos_b), axis=-1)
    e_phi = np.stack((-y / radius, x / radius, np.zeros_like(radius)), axis=-1)
    return {
        "x_kpc": x,
        "y_kpc": y,
        "z_kpc": z,
        "radius_kpc": radius,
        "e_r": e_r,
        "e_l": e_l,
        "e_b": e_b,
        "e_phi": e_phi,
        "radial_projection": np.sum(e_phi * e_r, axis=-1),
        "longitude_projection": np.sum(e_phi * e_l, axis=-1),
    }


def solar_galactocentric_velocity(
    theta0_km_s: float,
    solar_peculiar_uvw_km_s,
) -> np.ndarray:
    """Return the Sun's Galactocentric Cartesian velocity."""
    uvw = np.asarray(solar_peculiar_uvw_km_s, dtype=float)
    if uvw.shape != (3,) or np.any(~np.isfinite(uvw)):
        raise ValueError("solar peculiar UVW must contain three finite values")
    if not np.isfinite(theta0_km_s) or theta0_km_s <= 0.0:
        raise ValueError("theta0 must be finite and positive")
    u, v, w = uvw
    return np.asarray([-u, theta0_km_s + v, w])


def lsr_to_heliocentric_velocity(
    v_lsr_km_s,
    longitude_rad,
    latitude_rad,
    standard_solar_uvw_km_s,
) -> np.ndarray:
    """Convert conventional LSR radial velocity to heliocentric velocity."""
    velocity = np.asarray(v_lsr_km_s, dtype=float)
    longitude = np.asarray(longitude_rad, dtype=float)
    latitude = np.asarray(latitude_rad, dtype=float)
    velocity, longitude, latitude = np.broadcast_arrays(
        velocity, longitude, latitude
    )
    uvw = np.asarray(standard_solar_uvw_km_s, dtype=float)
    if uvw.shape != (3,) or np.any(~np.isfinite(uvw)):
        raise ValueError("standard solar UVW must contain three finite values")
    u, v, w = uvw
    correction = (
        u * np.cos(longitude) * np.cos(latitude)
        + v * np.sin(longitude) * np.cos(latitude)
        + w * np.sin(latitude)
    )
    return velocity - correction


def circular_speed_from_channels(
    geometry: dict[str, np.ndarray],
    *,
    transverse_longitude_velocity_km_s,
    heliocentric_radial_velocity_km_s,
    solar_velocity_km_s,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer circular speed separately from transverse and radial channels."""
    transverse = np.asarray(transverse_longitude_velocity_km_s, dtype=float)
    radial = np.asarray(heliocentric_radial_velocity_km_s, dtype=float)
    transverse, radial = np.broadcast_arrays(transverse, radial)
    solar = np.asarray(solar_velocity_km_s, dtype=float)
    if solar.shape != (3,) or np.any(~np.isfinite(solar)):
        raise ValueError("solar velocity must contain three finite values")
    e_l = np.asarray(geometry["e_l"], dtype=float)
    e_r = np.asarray(geometry["e_r"], dtype=float)
    longitude_projection = np.asarray(
        geometry["longitude_projection"], dtype=float
    )
    radial_projection = np.asarray(geometry["radial_projection"], dtype=float)
    theta_pm = (
        transverse + np.sum(e_l * solar, axis=-1)
    ) / longitude_projection
    theta_rv = (
        radial + np.sum(e_r * solar, axis=-1)
    ) / radial_projection
    return theta_pm, theta_rv


def circular_channel_velocities(
    geometry: dict[str, np.ndarray],
    theta_km_s,
    *,
    solar_velocity_km_s,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-predict longitude and radial velocities for circular motion."""
    theta = np.asarray(theta_km_s, dtype=float)
    solar = np.asarray(solar_velocity_km_s, dtype=float)
    source = theta[..., np.newaxis] * np.asarray(geometry["e_phi"], dtype=float)
    relative = source - solar
    transverse = np.sum(relative * np.asarray(geometry["e_l"]), axis=-1)
    radial = np.sum(relative * np.asarray(geometry["e_r"]), axis=-1)
    return transverse, radial
