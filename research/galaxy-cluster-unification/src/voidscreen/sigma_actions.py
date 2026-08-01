"""Action-motivated force laws gated by the screened Sigma field.

These functions hold the environmental Sigma solution fixed.  This is the
leading weak-backreaction limit of a joint action, not yet a covariant theory.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp

from .sigma_field import KPC_CM, SigmaSolve, local_sigma_equilibrium, radial_cell_centers


C_M_S = 299_792_458.0
KPC_M = KPC_CM / 100.0
G_SI = 6.67430e-11


def _fields(gbar_m_s2, sigma) -> tuple[np.ndarray, np.ndarray]:
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    field = np.asarray(sigma, dtype=np.float64)
    if gbar.shape != field.shape:
        raise ValueError("gbar and sigma must have the same shape")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError("sigma must be finite and nonnegative")
    return gbar, field


def conformal_symmetron_acceleration(
    gbar_m_s2, radius_kpc, sigma, *, alpha: float
) -> np.ndarray:
    """Matter acceleration from A(Sigma)=exp(alpha*Sigma^2/2).

    Positive radial dSigma/dr produces an additional inward acceleration.
    A negative gradient is retained as an outward contribution rather than
    silently taking its absolute value.
    """
    gbar, field = _fields(gbar_m_s2, sigma)
    radius = np.asarray(radius_kpc, dtype=np.float64)
    if radius.shape != field.shape or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must match sigma and strictly increase")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    derivative_per_m = np.gradient(field, radius) / KPC_M
    return gbar + alpha * C_M_S**2 * field * derivative_per_m


def refracted_aqual_mu(
    acceleration_m_s2,
    sigma,
    *,
    a0_m_s2: float,
    activation: float = 1.0,
    eta: float = 0.0,
) -> np.ndarray:
    """Return mu=epsilon*y/(y+activation*Sigma^2), y=g/a0."""
    acceleration = np.asarray(acceleration_m_s2, dtype=np.float64)
    field = np.asarray(sigma, dtype=np.float64)
    if acceleration.shape != field.shape or np.any(acceleration <= 0.0):
        raise ValueError("acceleration and sigma must match and acceleration be positive")
    if a0_m_s2 <= 0.0 or activation <= 0.0 or not 0.0 <= eta < 1.0:
        raise ValueError("invalid AQUAL parameters")
    y = acceleration / a0_m_s2
    epsilon = 1.0 - eta * field**2
    if np.any(epsilon <= 0.0):
        raise ValueError("eta and sigma must keep the kinetic coefficient positive")
    return epsilon * y / (y + activation * field**2)


def refracted_aqual_acceleration(
    gbar_m_s2,
    sigma,
    *,
    a0_m_s2: float,
    activation: float = 1.0,
    eta: float = 0.0,
) -> np.ndarray:
    """Solve mu(g,Sigma)*g=g_N in spherical symmetry in closed form."""
    gbar, field = _fields(gbar_m_s2, sigma)
    if a0_m_s2 <= 0.0 or activation <= 0.0 or not 0.0 <= eta < 1.0:
        raise ValueError("invalid AQUAL parameters")
    epsilon = 1.0 - eta * field**2
    if np.any(epsilon <= 0.0):
        raise ValueError("eta and sigma must keep the kinetic coefficient positive")
    transition = a0_m_s2 * activation * field**2
    return (gbar + np.sqrt(gbar**2 + 4.0 * epsilon * gbar * transition)) / (
        2.0 * epsilon
    )


def refracted_aqual_free_function(
    X,
    sigma,
    *,
    activation: float = 1.0,
    eta: float = 0.0,
) -> np.ndarray:
    """Static AQUAL free function F with dF/dX=mu(sqrt(X),Sigma).

    F is normalized to zero at X=0.  It demonstrates that the interpolation
    used here descends from a compact nonrelativistic action.
    """
    kinetic = np.asarray(X, dtype=np.float64)
    field = np.asarray(sigma, dtype=np.float64)
    if kinetic.shape != field.shape or np.any(kinetic < 0.0):
        raise ValueError("X and sigma must match and X be nonnegative")
    if activation <= 0.0 or not 0.0 <= eta < 1.0:
        raise ValueError("invalid AQUAL parameters")
    root = np.sqrt(kinetic)
    scale = activation * field**2
    epsilon = 1.0 - eta * field**2
    if np.any(epsilon <= 0.0):
        raise ValueError("eta and sigma must keep the kinetic coefficient positive")
    result = np.empty_like(root)
    zero = scale == 0.0
    result[zero] = epsilon[zero] * kinetic[zero]
    nonzero = ~zero
    s = scale[nonzero]
    y = root[nonzero]
    result[nonzero] = epsilon[nonzero] * (
        kinetic[nonzero] - 2.0 * s * y + 2.0 * s**2 * np.log1p(y / s)
    )
    return result


def refracted_aqual_free_function_sigma_derivative(
    X,
    sigma,
    *,
    activation: float = 1.0,
    eta: float = 0.0,
) -> np.ndarray:
    """Return the partial derivative dF/dSigma at fixed X."""
    kinetic = np.asarray(X, dtype=np.float64)
    field = np.asarray(sigma, dtype=np.float64)
    if kinetic.shape != field.shape or np.any(kinetic < 0.0):
        raise ValueError("X and sigma must match and X be nonnegative")
    if activation <= 0.0 or not 0.0 <= eta < 1.0:
        raise ValueError("invalid AQUAL parameters")
    root = np.sqrt(kinetic)
    scale = activation * field**2
    epsilon = 1.0 - eta * field**2
    if np.any(epsilon <= 0.0):
        raise ValueError("eta and sigma must keep the kinetic coefficient positive")
    result = np.zeros_like(root)
    nonzero = scale > 1.0e-14
    s = scale[nonzero]
    y = root[nonzero]
    logarithm = np.log1p(y / s)
    base = kinetic[nonzero] - 2.0 * s * y + 2.0 * s**2 * logarithm
    base_scale_derivative = (
        -2.0 * y + 4.0 * s * logarithm - 2.0 * y * s / (s + y)
    )
    sigma_values = field[nonzero]
    result[nonzero] = (
        -2.0 * eta * sigma_values * base
        + epsilon[nonzero]
        * base_scale_derivative
        * (2.0 * activation * sigma_values)
    )
    return result


def solve_coupled_spherical_sigma(
    radius_faces_kpc,
    density_g_cm3,
    gbar_m_s2,
    *,
    rho_s_g_cm3: float,
    length_kpc: float,
    a0_m_s2: float,
    eta: float,
    backreaction: float,
    activation: float = 1.0,
    outer_sigma: float = 1.0,
    initial_sigma=None,
) -> SigmaSolve:
    """Solve Sigma after varying the same static AQUAL action.

    ``backreaction`` is the ratio of the AQUAL energy scale to the Sigma-field
    stiffness.  Zero reproduces the one-way environmental-field approximation.
    """
    faces = np.asarray(radius_faces_kpc, dtype=np.float64)
    density = np.asarray(density_g_cm3, dtype=np.float64)
    gbar = np.asarray(gbar_m_s2, dtype=np.float64)
    if faces.ndim != 1 or len(faces) != len(density) + 1 or gbar.shape != density.shape:
        raise ValueError("faces, density, and gbar have inconsistent shapes")
    if faces[0] != 0.0 or np.any(np.diff(faces) <= 0.0):
        raise ValueError("radius faces must start at zero and strictly increase")
    if np.any(density < 0.0) or np.any(gbar <= 0.0):
        raise ValueError("density must be nonnegative and gbar positive")
    if (
        rho_s_g_cm3 <= 0.0
        or length_kpc <= 0.0
        or a0_m_s2 <= 0.0
        or activation <= 0.0
        or not 0.0 <= eta < 1.0
        or backreaction < 0.0
        or not 0.0 <= outer_sigma <= 1.0
    ):
        raise ValueError("invalid coupled-action parameters")

    centers = radial_cell_centers(faces)
    mesh = np.r_[centers, faces[-1]]
    log_centers = np.log(centers)

    def interpolate(values, radius):
        return np.interp(np.log(radius), log_centers, values)

    if initial_sigma is None:
        initial = local_sigma_equilibrium(density, rho_s_g_cm3)
    else:
        initial = np.asarray(initial_sigma, dtype=np.float64)
        if initial.shape != density.shape:
            raise ValueError("initial_sigma must match density")
    initial = np.clip(initial, 0.0, 1.0)
    initial_field = np.r_[initial, outer_sigma]
    initial_values = np.vstack(
        (initial_field, np.gradient(initial_field, mesh))
    )

    def equation(radius: np.ndarray, values: np.ndarray) -> np.ndarray:
        sigma, derivative = values
        physical_sigma = np.maximum(sigma, 0.0)
        local_gbar = interpolate(gbar, radius)
        epsilon = np.maximum(1.0e-6, 1.0 - eta * physical_sigma**2)
        transition = a0_m_s2 * activation * physical_sigma**2
        acceleration = (
            local_gbar
            + np.sqrt(local_gbar**2 + 4.0 * epsilon * local_gbar * transition)
        ) / (2.0 * epsilon)
        kinetic = (acceleration / a0_m_s2) ** 2
        feedback = refracted_aqual_free_function_sigma_derivative(
            kinetic,
            physical_sigma,
            activation=activation,
            eta=eta,
        )
        coefficient = interpolate(density, radius) / rho_s_g_cm3 - 1.0
        curvature = (
            coefficient * sigma + sigma**3 + backreaction * feedback
        ) / length_kpc**2
        return np.vstack((derivative, curvature - 2.0 * derivative / radius))

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array((left[1], right[0] - outer_sigma))

    result = solve_bvp(
        equation,
        boundary,
        mesh,
        initial_values,
        tol=1.0e-4,
        max_nodes=20_000,
        verbose=0,
    )
    field = np.maximum(result.sol(centers)[0], 0.0)
    maximum_residual = float(np.max(result.rms_residuals))
    sampled = result.sol(mesh)[0]
    physical = bool(
        np.min(sampled) >= -1.0e-4
        and np.min(1.0 - eta * np.maximum(sampled, 0.0) ** 2) > 1.0e-4
    )
    return SigmaSolve(
        field=field,
        energy=math.nan,
        converged=bool(result.success and physical and maximum_residual <= 2.0e-4),
        iterations=int(result.niter),
        maximum_scaled_gradient=maximum_residual,
    )


def scalar_field_stress_energy_profile(
    radius_kpc,
    sigma,
    *,
    length_kpc: float,
    a0_m_s2: float,
    backreaction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar mass density, enclosed mass, and Newtonian acceleration.

    The positive energy is measured relative to the broken-vacuum minimum:

        u = K * [0.5 L^2 |grad Sigma|^2 + 0.25 (1-Sigma^2)^2]

    with K=a0^2/(8*pi*G*backreaction).  This is the weak-field source implied
    by the same scalar normalization used in the coupled static action.
    """
    radius = np.asarray(radius_kpc, dtype=np.float64)
    field = np.asarray(sigma, dtype=np.float64)
    if radius.shape != field.shape or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius and sigma must match and radius strictly increase")
    if np.any(~np.isfinite(field)) or length_kpc <= 0.0 or a0_m_s2 <= 0.0:
        raise ValueError("invalid scalar energy inputs")
    if not math.isfinite(backreaction) or backreaction <= 0.0:
        raise ValueError("backreaction must be finite and positive")
    derivative = np.gradient(field, radius)
    dimensionless_energy = (
        0.5 * length_kpc**2 * derivative**2
        + 0.25 * (1.0 - field**2) ** 2
    )
    stiffness_j_m3 = a0_m_s2**2 / (8.0 * math.pi * G_SI * backreaction)
    density_kg_m3 = stiffness_j_m3 * dimensionless_energy / C_M_S**2
    radius_m = radius * KPC_M
    mass_kg = 4.0 * math.pi * cumulative_trapezoid(
        density_kg_m3 * radius_m**2, radius_m, initial=0.0
    )
    acceleration = G_SI * mass_kg / radius_m**2
    return density_kg_m3, mass_kg, acceleration
