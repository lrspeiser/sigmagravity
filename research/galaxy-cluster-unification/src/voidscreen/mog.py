from __future__ import annotations

import numpy as np

from .unified import C_M_S, G_SI


def yukawa_transition(x) -> np.ndarray:
    """Return 1-(1+x)exp(-x) without cancellation at small positive x."""
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("x must be finite and nonnegative")
    output = np.empty_like(values)
    small = values < 1e-3
    z = values[small]
    output[small] = (
        np.square(z) / 2.0
        - np.power(z, 3) / 3.0
        + np.power(z, 4) / 8.0
        - np.power(z, 5) / 30.0
        + np.power(z, 6) / 144.0
    )
    z = values[~small]
    output[~small] = -np.expm1(-z) - z * np.exp(-z)
    return output


def matched_mog_enhancement(x, alpha: float) -> np.ndarray:
    """Massive-particle enhancement when 1/F=1+alpha in one environment."""
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    return 1.0 + alpha * yukawa_transition(x)


def environmental_mog_dynamic_enhancement(
    x, metric_enhancement, alpha: float
) -> np.ndarray:
    """Metric attraction minus the universal repulsive Proca force."""
    values = np.asarray(x, dtype=float)
    metric = np.asarray(metric_enhancement, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("x must be finite and nonnegative")
    if np.any(~np.isfinite(metric)) or np.any(metric <= 0.0):
        raise ValueError("metric enhancement must be finite and positive")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    return metric - alpha * (1.0 + values) * np.exp(-values)


def point_mass_dynamic_acceleration_m_s2(
    mass_kg: float,
    radius_m,
    *,
    metric_enhancement: float,
    alpha: float,
    range_m: float,
) -> np.ndarray:
    """Positive inward acceleration for the constant-scalar point solution."""
    radius = np.asarray(radius_m, dtype=float)
    if mass_kg <= 0.0 or not np.isfinite(mass_kg):
        raise ValueError("mass must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if range_m <= 0.0 or not np.isfinite(range_m):
        raise ValueError("range must be finite and positive")
    enhancement = environmental_mog_dynamic_enhancement(
        radius / range_m, metric_enhancement, alpha
    )
    return G_SI * mass_kg * enhancement / np.square(radius)


def point_mass_lensing_deflection_rad(
    mass_kg: float, impact_parameter_m, *, metric_enhancement: float
) -> np.ndarray:
    """Weak deflection from the same constant-scalar physical metric."""
    impact = np.asarray(impact_parameter_m, dtype=float)
    if mass_kg <= 0.0 or not np.isfinite(mass_kg):
        raise ValueError("mass must be finite and positive")
    if np.any(~np.isfinite(impact)) or np.any(impact <= 0.0):
        raise ValueError("impact parameter must be finite and positive")
    if metric_enhancement <= 0.0 or not np.isfinite(metric_enhancement):
        raise ValueError("metric enhancement must be finite and positive")
    return 4.0 * G_SI * mass_kg * metric_enhancement / (
        impact * C_M_S**2
    )


def chameleon_density_power(n: float) -> float:
    """Power p in s_min proportional to rho**(-p) for U=Lambda^2 s**(-n)."""
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be finite and positive")
    return 1.0 / (n + 1.0)


def chameleon_metric_enhancement(
    density_kg_m3,
    *,
    reference_density_kg_m3: float,
    z_reference: float,
    power: float,
    maximum_log_enhancement: float = 50.0,
) -> np.ndarray:
    """Adiabatic envelope exp[2 beta s_min(rho)] for the frozen action."""
    density = np.asarray(density_kg_m3, dtype=float)
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("density must be finite and positive")
    if reference_density_kg_m3 <= 0.0 or not np.isfinite(
        reference_density_kg_m3
    ):
        raise ValueError("reference density must be finite and positive")
    if z_reference <= 0.0 or not np.isfinite(z_reference):
        raise ValueError("z_reference must be finite and positive")
    if not 0.0 < power <= 1.0:
        raise ValueError("power must be in (0, 1]")
    if maximum_log_enhancement <= 0.0:
        raise ValueError("maximum log enhancement must be positive")
    log_enhancement = z_reference * np.power(
        density / reference_density_kg_m3, -power
    )
    return np.exp(np.minimum(log_enhancement, maximum_log_enhancement))


def mean_enclosed_density_kg_m3(gbar_m_s2, radius_m) -> np.ndarray:
    """Spherical mean baryonic density implied by gbar=G M(<r)/r^2."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_m, dtype=float)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    return 3.0 * gbar / (4.0 * np.pi * G_SI * radius)


def einstein_frame_scalar_kinetic(beta: float, f_value=1.0) -> np.ndarray:
    """Dimensionless scalar kinetic coefficient K=1/F+6 beta^2."""
    f_array = np.asarray(f_value, dtype=float)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    if np.any(~np.isfinite(f_array)) or np.any(f_array <= 0.0):
        raise ValueError("F must be finite and positive")
    return 1.0 / f_array + 6.0 * beta**2


def unscreened_ppn_gamma_minus_one(beta: float, f_value=1.0) -> np.ndarray:
    """Massless local scalar limit; screening or a Yukawa factor only suppresses it."""
    kinetic = einstein_frame_scalar_kinetic(beta, f_value)
    coupling_squared = beta**2 / kinetic
    return -2.0 * coupling_squared / (1.0 + coupling_squared)


def vector_light_dynamics_gamma_minus_one(
    alpha: float, metric_enhancement: float
) -> float:
    """Effective gamma shift comparing metric lensing with short-range dynamics.

    This applies when mu*r is negligible, the local scalar gradient is screened,
    and massive matter therefore measures E-alpha while light measures E.
    """
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(metric_enhancement) or metric_enhancement <= alpha:
        raise ValueError("metric enhancement must exceed alpha")
    return 2.0 * alpha / (metric_enhancement - alpha)


def mog_extra_acceleration_log_slope(x) -> np.ndarray:
    """d ln(delta g)/d ln(r) for a point mass and constant alpha, mu."""
    values = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("x must be finite and positive")
    transition = yukawa_transition(values)
    return -2.0 + np.square(values) * np.exp(-values) / transition


def spherical_vector_acceleration_m_s2(
    evaluation_radius_m,
    shell_radius_m,
    shell_mass_kg,
    *,
    alpha: float,
    range_m: float,
) -> np.ndarray:
    """Exact signed radial Proca acceleration from concentric thin mass shells.

    Positive is outward. Exterior shells give an inward contribution because the
    nearer side of a repulsive Yukawa shell is stronger. The massless limit
    recovers the ordinary shell theorem.
    """
    evaluation = np.atleast_1d(np.asarray(evaluation_radius_m, dtype=float))
    shell_radius = np.atleast_1d(np.asarray(shell_radius_m, dtype=float))
    shell_mass = np.atleast_1d(np.asarray(shell_mass_kg, dtype=float))
    if shell_radius.shape != shell_mass.shape:
        raise ValueError("shell radii and masses must have matching shapes")
    if np.any(~np.isfinite(evaluation)) or np.any(evaluation <= 0.0):
        raise ValueError("evaluation radii must be finite and positive")
    if np.any(~np.isfinite(shell_radius)) or np.any(shell_radius < 0.0):
        raise ValueError("shell radii must be finite and nonnegative")
    if np.any(~np.isfinite(shell_mass)) or np.any(shell_mass < 0.0):
        raise ValueError("shell masses must be finite and nonnegative")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(range_m) or range_m <= 0.0:
        raise ValueError("range must be finite and positive")

    inverse_range = 1.0 / range_m
    output = np.zeros_like(evaluation)
    for index, radius in enumerate(evaluation):
        x = inverse_range * radius
        outside = radius >= shell_radius

        y = inverse_range * shell_radius[outside]
        scaled = np.empty_like(y)
        small_y = y < 1e-5
        scaled[small_y] = np.exp(-x) * (
            1.0 + np.square(y[small_y]) / 6.0
        )
        yy = y[~small_y]
        scaled[~small_y] = 0.5 * (
            np.exp(-(x - yy)) - np.exp(-(x + yy))
        ) / yy
        exterior_kernel = (1.0 + x) * scaled / radius**2

        inner_shells = ~outside
        y = inverse_range * shell_radius[inner_shells]
        if x < 1e-5:
            scaled_derivative = np.exp(-y) * (
                x / 3.0 + x**3 / 30.0 + x**5 / 840.0
            )
        else:
            scaled_derivative = 0.5 * (
                (x - 1.0) * np.exp(-(y - x))
                + (x + 1.0) * np.exp(-(y + x))
            ) / x**2
        interior_kernel = np.zeros_like(y)
        if len(y):
            interior_kernel = (
                -inverse_range * scaled_derivative / shell_radius[inner_shells]
            )

        weighted = np.dot(shell_mass[outside], exterior_kernel)
        weighted += np.dot(shell_mass[inner_shells], interior_kernel)
        output[index] = alpha * G_SI * weighted
    return output
