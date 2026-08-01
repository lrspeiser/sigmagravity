"""Scaling and energy-budget gates for the nonlinear NBM0 survivors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .basin_action import G_SI


@dataclass(frozen=True)
class FluxLawScaling:
    response_power: float
    acceleration_mass_exponent: float
    acceleration_radial_exponent: float
    circular_speed_radial_exponent: float
    circular_speed_fourth_power_mass_exponent: float


@dataclass(frozen=True)
class AlgebraicFieldScaling:
    base_field_radial_exponent: float
    response_power: float
    acceleration_mass_exponent: float
    acceleration_radial_exponent: float
    circular_speed_squared_radial_exponent: float


def nonlinear_flux_law_scaling(response_power: float) -> FluxLawScaling:
    """Scaling for ``div(|g|^(m-1) g) proportional rho`` in spherical symmetry.

    Gauss' law gives ``g^m r^2 proportional M``.  The unique power that produces
    both ``g proportional sqrt(M)/r`` and ``v_flat^4 proportional M`` is m=2.
    """
    if not math.isfinite(response_power) or response_power <= 0.0:
        raise ValueError("response_power must be finite and positive")
    mass_exponent = 1.0 / response_power
    acceleration_radial_exponent = -2.0 / response_power
    return FluxLawScaling(
        response_power=response_power,
        acceleration_mass_exponent=mass_exponent,
        acceleration_radial_exponent=acceleration_radial_exponent,
        circular_speed_radial_exponent=0.5
        * (1.0 + acceleration_radial_exponent),
        circular_speed_fourth_power_mass_exponent=2.0 * mass_exponent,
    )


def algebraic_inverse_field_scaling(
    response_power: float,
    *,
    base_field_radial_exponent: float = -1.0,
) -> AlgebraicFieldScaling:
    """Scaling when a linear inverse field X is mapped algebraically as Phi~X^n.

    The base field is assumed linear in source mass.  For a massless inverse
    Laplacian in three dimensions, ``X proportional M r^-1``.
    """
    if not math.isfinite(response_power):
        raise ValueError("response_power must be finite")
    if not math.isfinite(base_field_radial_exponent):
        raise ValueError("base_field_radial_exponent must be finite")
    potential_radial_exponent = response_power * base_field_radial_exponent
    return AlgebraicFieldScaling(
        base_field_radial_exponent=base_field_radial_exponent,
        response_power=response_power,
        acceleration_mass_exponent=response_power,
        acceleration_radial_exponent=potential_radial_exponent - 1.0,
        circular_speed_squared_radial_exponent=potential_radial_exponent,
    )


def canonical_scalar_exterior_energy_fraction(
    source_d: float,
    compactness: np.ndarray | float,
) -> np.ndarray:
    """Exterior scalar field energy divided by Mc^2 for a massless canonical field.

    For ``X=-2 d GM/(c^2 r)`` outside a source of radius R, integration of
    ``M_Pl^2 |grad X|^2/2`` gives ``E_X/(Mc^2)=d^2 GM/(Rc^2)``.
    """
    values = np.asarray(compactness, dtype=np.float64)
    if not math.isfinite(source_d):
        raise ValueError("source_d must be finite")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("compactness must be finite and nonnegative")
    return source_d**2 * values


def direct_force_amplitude_for_field_energy_fraction(
    target_energy_fraction: np.ndarray | float,
    compactness: np.ndarray | float,
) -> np.ndarray:
    """Return ``A_dyn=2d^2`` required for a canonical exterior energy budget."""
    target = np.asarray(target_energy_fraction, dtype=np.float64)
    compact = np.asarray(compactness, dtype=np.float64)
    target, compact = np.broadcast_arrays(target, compact)
    if np.any(~np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError("target_energy_fraction must be finite and nonnegative")
    if np.any(~np.isfinite(compact)) or np.any(compact <= 0.0):
        raise ValueError("compactness must be finite and positive")
    return 2.0 * target / compact


def compactness(mass_kg: float, radius_m: float, speed_of_light_m_s: float) -> float:
    """Return GM/(Rc^2)."""
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if not math.isfinite(speed_of_light_m_s) or speed_of_light_m_s <= 0.0:
        raise ValueError("speed_of_light_m_s must be finite and positive")
    return G_SI * mass_kg / (radius_m * speed_of_light_m_s**2)


def uniform_vacuum_radial_acceleration_m_s2(
    radius_m: np.ndarray | float,
    vacuum_mass_density_kg_m3: float,
) -> np.ndarray:
    """Outward GR acceleration from uniform vacuum energy with p=-rho c^2."""
    radius = np.asarray(radius_m, dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius_m must be finite and nonnegative")
    if not math.isfinite(vacuum_mass_density_kg_m3) or vacuum_mass_density_kg_m3 < 0.0:
        raise ValueError("vacuum density must be finite and nonnegative")
    return (8.0 * math.pi * G_SI / 3.0) * vacuum_mass_density_kg_m3 * radius
