from __future__ import annotations

import numpy as np


def power_law_planck_function(x, x_background: float, alpha_h: float, f0: float = 1.0):
    """Return ``F(X)=F0*(-X/-X0)^(-alpha_H/2)`` on the timelike branch."""
    value = np.asarray(x, dtype=float)
    if x_background >= 0.0 or np.any(value >= 0.0) or f0 <= 0.0:
        raise ValueError("X and X_background must be timelike (negative), and F0 positive")
    return f0 * np.power(value / x_background, -0.5 * alpha_h)


def luminal_beyond_horndeski_coefficients(
    x, x_background: float, alpha_h: float, f0: float = 1.0
) -> dict[str, np.ndarray]:
    """Return the one-parameter luminal beyond-Horndeski coefficient set.

    The quadratic DHOST basis is

    ``F R + sum(A_I L_I)``.

    The choice has ``A1=A2=A5=0``, ``A3=-4 F_X/X`` and ``A4=+4 F_X/X``.
    It is the beta_1=0 subclass of the c_T=1 class-I DHOST degeneracy
    relations and has constant EFT coefficient ``alpha_H=-2 X F_X/F``.
    """
    value = np.asarray(x, dtype=float)
    f_value = power_law_planck_function(value, x_background, alpha_h, f0)
    f_x = -0.5 * alpha_h * f_value / value
    a3 = -4.0 * f_x / value
    a4 = 4.0 * f_x / value
    zeros = np.zeros_like(f_value)
    return {
        "F": f_value,
        "F_X": f_x,
        "A1": zeros,
        "A2": zeros,
        "A3": a3,
        "A4": a4,
        "A5": zeros,
    }


def dhost_degeneracy_residuals(coefficients: dict[str, np.ndarray], x) -> dict[str, np.ndarray]:
    """Evaluate the c_T=1 quadratic-DHOST A4/A5 degeneracy identities."""
    value = np.asarray(x, dtype=float)
    f_value = np.asarray(coefficients["F"], dtype=float)
    f_x = np.asarray(coefficients["F_X"], dtype=float)
    a3 = np.asarray(coefficients["A3"], dtype=float)
    predicted_a4 = (
        48.0 * np.square(f_x)
        - 8.0 * (f_value - value * f_x) * a3
        - np.square(value * a3)
    ) / (8.0 * f_value)
    predicted_a5 = (4.0 * f_x + value * a3) * a3 / (2.0 * f_value)
    return {
        "A1": np.asarray(coefficients["A1"], dtype=float),
        "A2": np.asarray(coefficients["A2"], dtype=float),
        "A4": np.asarray(coefficients["A4"], dtype=float) - predicted_a4,
        "A5": np.asarray(coefficients["A5"], dtype=float) - predicted_a5,
    }


def spherical_xi_coefficients(alpha_h: float) -> dict[str, float]:
    """Return the screened spherical coefficients for the beta_1=0 subclass."""
    if not np.isfinite(alpha_h):
        raise ValueError("alpha_H must be finite")
    return {
        "Xi1_time_mass_second_derivative": -0.5 * alpha_h,
        "Xi2_spatial_mass_first_derivative": alpha_h,
        "Xi3_spatial_mass_second_derivative": 0.0,
    }


def spherical_accelerations(
    radius,
    enclosed_mass,
    enclosed_mass_first_derivative,
    enclosed_mass_second_derivative,
    alpha_h: float,
    gravitational_constant: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return time, spatial, and Weyl radial gradients on the screened branch.

    The naming follows this project: ``Psi`` is the time potential seen by
    nonrelativistic matter and ``Phi`` is the spatial potential.  The source
    equations are the beta_1=0 specialization of the published c_T=1 DHOST
    spherical laws.
    """
    r = np.asarray(radius, dtype=float)
    mass = np.asarray(enclosed_mass, dtype=float)
    mass_prime = np.asarray(enclosed_mass_first_derivative, dtype=float)
    mass_second = np.asarray(enclosed_mass_second_derivative, dtype=float)
    if np.any(r <= 0.0) or gravitational_constant <= 0.0:
        raise ValueError("radius and the gravitational constant must be positive")
    base = gravitational_constant * mass / np.square(r)
    matter_psi = base - 0.5 * alpha_h * gravitational_constant * mass_second
    spatial_phi = base + alpha_h * gravitational_constant * mass_prime / r
    return {
        "newtonian": base,
        "matter_psi": matter_psi,
        "spatial_phi": spatial_phi,
        "photon_weyl": 0.5 * (matter_psi + spatial_phi),
    }


def weyl_edge_correction_from_density_gradient(
    radius, density_gradient, alpha_h: float, gravitational_constant: float = 1.0
) -> np.ndarray:
    """Return ``delta g_W=-pi alpha_H G r^2 rho'(r)``."""
    r = np.asarray(radius, dtype=float)
    gradient = np.asarray(density_gradient, dtype=float)
    if np.any(r <= 0.0) or gravitational_constant <= 0.0:
        raise ValueError("radius and the gravitational constant must be positive")
    return -np.pi * alpha_h * gravitational_constant * np.square(r) * gradient


def uniform_density_acceleration_ratios(alpha_h: float) -> dict[str, float]:
    """Return gravity ratios inside a constant-density sphere."""
    return {
        "matter_psi_over_newtonian": 1.0 - 3.0 * alpha_h,
        "spatial_phi_over_newtonian": 1.0 + 3.0 * alpha_h,
        "photon_weyl_over_newtonian": 1.0,
    }


def smooth_power_law_weyl_fraction(density_slope: float, alpha_h: float) -> float:
    """Fractional Weyl correction for rho proportional to r^-n, 0<n<3."""
    slope = float(density_slope)
    if not 0.0 <= slope <= 3.0:
        raise ValueError("the smooth finite-enclosed-mass screen uses 0 <= slope <= 3")
    return 0.25 * alpha_h * slope * (3.0 - slope)


def maximum_smooth_power_law_weyl_fraction(alpha_h: float) -> float:
    """Maximum of alpha_H*n*(3-n)/4 over 0<=n<=3."""
    return 9.0 * alpha_h / 16.0
