"""Scale and amplitude audits for a spherical Vainshtein spin-2 carrier.

This module uses only the universal weak spherical limits of ghost-free
bigravity: ``r_V=(r_S L^2)^(1/3)`` and the mixing-angle coefficients of the
linear exterior solution.  It is a theory-construction audit, not an
astrophysical fit or a substitute for a nonlinear three-dimensional solver.
"""

from __future__ import annotations

import numpy as np

G_SI = 6.67430e-11
C_SI = 299_792_458.0


def vainshtein_radius_m(mass_kg: np.ndarray | float, carrier_range_m: float) -> np.ndarray:
    """Return ``(2 G M L^2 / c^2)^(1/3)`` in metres."""

    mass = np.asarray(mass_kg, dtype=float)
    range_value = float(carrier_range_m)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("mass_kg must be finite and positive")
    if not np.isfinite(range_value) or range_value <= 0.0:
        raise ValueError("carrier_range_m must be finite and positive")
    schwarzschild_radius = 2.0 * G_SI * mass / C_SI**2
    return np.cbrt(schwarzschild_radius * range_value**2)


def screening_coordinate(
    radius_m: np.ndarray | float,
    mass_kg: np.ndarray | float,
    carrier_range_m: float,
) -> np.ndarray:
    """Return ``r/r_V``; values below one are inside the spherical screen."""

    radius = np.asarray(radius_m, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius_m must be finite and positive")
    return radius / vainshtein_radius_m(mass_kg, carrier_range_m)


def transition_mean_density_kg_m3(carrier_range_m: float) -> float:
    """Return mean density at ``r=r_V`` for the spherical scaling law."""

    range_value = float(carrier_range_m)
    if not np.isfinite(range_value) or range_value <= 0.0:
        raise ValueError("carrier_range_m must be finite and positive")
    return 3.0 * C_SI**2 / (8.0 * np.pi * G_SI * range_value**2)


def carrier_range_for_transition_m(radius_m: float, mass_kg: float) -> float:
    """Return the carrier range for which a source has ``r=r_V``."""

    radius = float(radius_m)
    mass = float(mass_kg)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be finite and positive")
    schwarzschild_radius = 2.0 * G_SI * mass / C_SI**2
    return float(np.sqrt(radius**3 / schwarzschild_radius))


def bigravity_mixing_coefficients(mixing_angle_rad: np.ndarray | float) -> dict[str, np.ndarray]:
    """Return the exterior Newton and Yukawa coefficients of one-metric bigravity."""

    angle = np.asarray(mixing_angle_rad, dtype=float)
    if np.any(~np.isfinite(angle)) or np.any(angle < 0.0) or np.any(angle > np.pi / 2.0):
        raise ValueError("mixing_angle_rad must lie in [0, pi/2]")
    sine_squared = np.sin(angle) ** 2
    alpha = (1.0 - sine_squared) * (1.0 + (2.0 / 3.0) * sine_squared)
    beta = (2.0 / 3.0) * sine_squared * (1.0 + 2.0 * sine_squared)
    return {
        "newton_coefficient": alpha,
        "yukawa_coefficient": beta,
        "short_range_dynamics_factor": alpha + beta,
        "short_range_lensing_factor": alpha + 0.75 * beta,
    }


def equal_density_screening_residual(
    mass_a_kg: float,
    radius_a_m: float,
    mass_b_kg: float,
    radius_b_m: float,
    carrier_ranges_m: np.ndarray,
) -> dict[str, float]:
    """Compare screening coordinates for two equal-mean-density systems."""

    ranges = np.asarray(carrier_ranges_m, dtype=float)
    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError("carrier_ranges_m must be a nonempty vector")
    coordinate_a = np.array(
        [screening_coordinate(radius_a_m, mass_a_kg, value) for value in ranges]
    )
    coordinate_b = np.array(
        [screening_coordinate(radius_b_m, mass_b_kg, value) for value in ranges]
    )
    scale = np.maximum(np.maximum(np.abs(coordinate_a), np.abs(coordinate_b)), 1.0e-300)
    relative = np.abs(coordinate_a - coordinate_b) / scale
    density_a = float(mass_a_kg / radius_a_m**3)
    density_b = float(mass_b_kg / radius_b_m**3)
    return {
        "mass_over_radius_cubed_ratio": density_a / density_b,
        "maximum_relative_screening_coordinate_difference": float(np.max(relative)),
    }


def audit_spherical_vainshtein_carrier(
    *,
    carrier_ranges_m: np.ndarray,
    mixing_angles_rad: np.ndarray,
    protected_mass_kg: float,
    protected_radius_m: float,
    target_mass_kg: float,
    target_radius_m: float,
    required_lensing_enhancement: float,
) -> dict[str, object]:
    """Evaluate no-label separation and maximum-amplitude gates."""

    ranges = np.asarray(carrier_ranges_m, dtype=float)
    angles = np.asarray(mixing_angles_rad, dtype=float)
    if ranges.ndim != 1 or ranges.size == 0 or np.any(ranges <= 0.0):
        raise ValueError("carrier_ranges_m must be a positive nonempty vector")
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("mixing_angles_rad must be a nonempty vector")
    required = float(required_lensing_enhancement)
    if not np.isfinite(required) or required <= 1.0:
        raise ValueError("required_lensing_enhancement must exceed one")

    residual = equal_density_screening_residual(
        protected_mass_kg,
        protected_radius_m,
        target_mass_kg,
        target_radius_m,
        ranges,
    )
    protected_coordinate = np.array(
        [
            screening_coordinate(protected_radius_m, protected_mass_kg, value)
            for value in ranges
        ]
    )
    target_coordinate = np.array(
        [
            screening_coordinate(target_radius_m, target_mass_kg, value)
            for value in ranges
        ]
    )
    coefficients = bigravity_mixing_coefficients(angles)
    maximum_lensing = float(np.max(coefficients["short_range_lensing_factor"]))
    maximum_dynamics = float(np.max(coefficients["short_range_dynamics_factor"]))
    separation_exists = bool(
        np.any((protected_coordinate < 1.0) & (target_coordinate > 1.0))
    )
    gates = {
        "protected_system_screened_while_target_unscreened": separation_exists,
        "useful_lensing_amplitude": maximum_lensing >= required,
        "positive_mixing_coefficients": bool(
            np.all(coefficients["newton_coefficient"] >= 0.0)
            and np.all(coefficients["yukawa_coefficient"] >= 0.0)
        ),
    }
    return {
        "equal_density_stress_test": residual,
        "protected_screening_coordinate_range": [
            float(np.min(protected_coordinate)),
            float(np.max(protected_coordinate)),
        ],
        "target_screening_coordinate_range": [
            float(np.min(target_coordinate)),
            float(np.max(target_coordinate)),
        ],
        "maximum_unscreened_lensing_enhancement": maximum_lensing,
        "maximum_unscreened_dynamics_enhancement": maximum_dynamics,
        "gates": {name: bool(value) for name, value in gates.items()},
    }
