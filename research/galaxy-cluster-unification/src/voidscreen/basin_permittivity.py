"""Analytic controls for constitutive gravitational-permittivity candidates."""

from __future__ import annotations

import math

import numpy as np

from .basin_action import G_SI, KPC_M, M_SUN_KG


def spherical_permittivity_acceleration_m_s2(
    radius_m: np.ndarray | float,
    enclosed_mass_kg: np.ndarray | float,
    permittivity: np.ndarray | float,
) -> np.ndarray:
    """Spherical Gauss-law acceleration for div(epsilon grad Phi)=4 pi G rho."""
    radius = np.asarray(radius_m, dtype=np.float64)
    mass = np.asarray(enclosed_mass_kg, dtype=np.float64)
    epsilon = np.asarray(permittivity, dtype=np.float64)
    radius, mass, epsilon = np.broadcast_arrays(radius, mass, epsilon)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_m must be finite and positive")
    if np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("enclosed_mass_kg must be finite and nonnegative")
    if np.any(~np.isfinite(epsilon)) or np.any(epsilon <= 0.0):
        raise ValueError("permittivity must be finite and positive")
    return G_SI * mass / (epsilon * np.square(radius))


def confined_slab_acceleration_m_s2(
    radius_m: np.ndarray | float,
    mass_kg: float,
    half_height_m: float,
    *,
    interior_permittivity: float = 1.0,
) -> np.ndarray:
    """Far-field radial acceleration when flux is confined to a slab of half-height h."""
    radius = np.asarray(radius_m, dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_m must be finite and positive")
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    if not math.isfinite(half_height_m) or half_height_m <= 0.0:
        raise ValueError("half_height_m must be finite and positive")
    if not math.isfinite(interior_permittivity) or interior_permittivity <= 0.0:
        raise ValueError("interior_permittivity must be finite and positive")
    return G_SI * mass_kg / (interior_permittivity * half_height_m * radius)


def required_confinement_half_height_kpc(
    baryonic_mass_solar: np.ndarray | float,
    flat_velocity_km_s: np.ndarray | float,
    *,
    interior_permittivity: float = 1.0,
) -> np.ndarray:
    """Invert v_flat^2=GM/(epsilon h) into a required boundary half-height."""
    mass = np.asarray(baryonic_mass_solar, dtype=np.float64)
    velocity = np.asarray(flat_velocity_km_s, dtype=np.float64)
    mass, velocity = np.broadcast_arrays(mass, velocity)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("baryonic_mass_solar must be finite and positive")
    if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0.0):
        raise ValueError("flat_velocity_km_s must be finite and positive")
    if not math.isfinite(interior_permittivity) or interior_permittivity <= 0.0:
        raise ValueError("interior_permittivity must be finite and positive")
    velocity_m_s = velocity * 1000.0
    height_m = G_SI * mass * M_SUN_KG / (
        interior_permittivity * np.square(velocity_m_s)
    )
    return height_m / KPC_M


def confined_slab_flat_velocity_km_s(
    baryonic_mass_solar: np.ndarray | float,
    half_height_kpc: np.ndarray | float,
    *,
    interior_permittivity: float = 1.0,
) -> np.ndarray:
    """Return the ideal flux-confined flat speed."""
    mass = np.asarray(baryonic_mass_solar, dtype=np.float64)
    height = np.asarray(half_height_kpc, dtype=np.float64)
    mass, height = np.broadcast_arrays(mass, height)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("baryonic_mass_solar must be finite and positive")
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("half_height_kpc must be finite and positive")
    if not math.isfinite(interior_permittivity) or interior_permittivity <= 0.0:
        raise ValueError("interior_permittivity must be finite and positive")
    speed_squared = G_SI * mass * M_SUN_KG / (
        interior_permittivity * height * KPC_M
    )
    return np.sqrt(speed_squared) / 1000.0
