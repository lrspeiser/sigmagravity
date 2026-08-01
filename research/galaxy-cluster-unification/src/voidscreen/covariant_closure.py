"""Weak-field diagnostics for a prospective covariant Sigma completion."""

from __future__ import annotations

import numpy as np


def equilibrium_sigma_from_density(
    density_g_cm3,
    *,
    rho_screen_g_cm3: float,
) -> np.ndarray:
    """Return the local equilibrium screening proxy used in the Sigma model.

    This is the algebraic, no-gradient limit of the field equation,
    ``Sigma^2 = max(0, 1 - rho/rho_screen)``.  It is useful for a local
    characteristic audit, but it is not a replacement for solving the full
    spatial Sigma profile.
    """
    density = np.asarray(density_g_cm3, dtype=float)
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and nonnegative")
    if not np.isfinite(rho_screen_g_cm3) or rho_screen_g_cm3 <= 0.0:
        raise ValueError("rho_screen_g_cm3 must be finite and positive")
    return np.sqrt(np.clip(1.0 - density / float(rho_screen_g_cm3), 0.0, 1.0))


def aqual_characteristics(
    acceleration_m_s2,
    sigma,
    *,
    a0_m_s2: float,
    activation: float = 1.0,
    eta: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return kinetic and characteristic coefficients of naive covariant AQUAL.

    For a Lorentz-scalar action with ``F_X=mu`` on a static spacelike
    background, the time kinetic coefficient is ``mu`` and the gradient
    coefficient parallel to the background is ``mu + 2 X F_XX``.  Their ratio
    is the squared parallel characteristic speed relative to the metric light
    cone.  Perpendicular perturbations propagate at the metric speed.
    """
    acceleration = np.asarray(acceleration_m_s2, dtype=float)
    field = np.asarray(sigma, dtype=float)
    if acceleration.shape != field.shape:
        raise ValueError("acceleration and sigma must have matching shapes")
    if np.any(~np.isfinite(acceleration)) or np.any(acceleration <= 0.0):
        raise ValueError("acceleration must be finite and positive")
    if np.any(~np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError("sigma must be finite and nonnegative")
    if a0_m_s2 <= 0.0 or activation <= 0.0 or not 0.0 <= eta < 1.0:
        raise ValueError("invalid AQUAL parameters")

    y = acceleration / float(a0_m_s2)
    transition = float(activation) * field**2
    epsilon = 1.0 - float(eta) * field**2
    if np.any(epsilon <= 0.0):
        raise ValueError("eta and sigma must keep the kinetic coefficient positive")
    mu = epsilon * y / (y + transition)
    parallel_speed_squared = 1.0 + transition / (y + transition)
    parallel_gradient = mu * parallel_speed_squared
    return {
        "mu_time_kinetic": mu,
        "parallel_gradient_coefficient": parallel_gradient,
        "parallel_speed_squared_over_c2": parallel_speed_squared,
        "perpendicular_speed_squared_over_c2": np.ones_like(mu),
    }


def causal_catchup_characteristics(
    acceleration_m_s2,
    sigma,
    *,
    a0_m_s2: float,
    delta: float,
    activation: float = 1.0,
    eta: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return characteristics for the causal ``Q`` catch-up completion.

    The weak-field equation is

    ``div(mu grad(Phi)) - Q/c^2 d_t^2 Phi = 4 pi G rho``

    with ``Q = L_parallel + delta Sigma^2`` and
    ``L_parallel = mu + 2 X mu_X``.  The added nonnegative term narrows the
    scalar characteristic cone without changing any static solution.  A
    value of zero is the fastest causal choice: the longitudinal mode then
    travels exactly on the metric light cone.
    """
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be finite and nonnegative")
    base = aqual_characteristics(
        acceleration_m_s2,
        sigma,
        a0_m_s2=a0_m_s2,
        activation=activation,
        eta=eta,
    )
    field = np.asarray(sigma, dtype=float)
    mu = base["mu_time_kinetic"]
    longitudinal = base["parallel_gradient_coefficient"]
    q_time = longitudinal + float(delta) * field**2
    parallel_speed_squared = longitudinal / q_time
    perpendicular_speed_squared = mu / q_time
    return {
        "mu_spatial_coefficient": mu,
        "parallel_spatial_coefficient": longitudinal,
        "q_time_coefficient": q_time,
        "parallel_speed_squared_over_c2": parallel_speed_squared,
        "perpendicular_speed_squared_over_c2": perpendicular_speed_squared,
        "parallel_refractive_index": np.sqrt(q_time / longitudinal),
        "perpendicular_refractive_index": np.sqrt(q_time / mu),
    }


def sigma_metric_lensing_acceleration(
    baryonic_acceleration_m_s2,
    dynamical_acceleration_m_s2,
    sigma,
    *,
    zeta: float,
) -> np.ndarray:
    """Return the weak-field lensing acceleration for a Sigma slip closure.

    Write the two metric potentials as

    ``Phi = Phi_N + phi``
    ``Psi = Phi_N + (1 + zeta*Sigma^2)*phi``.

    Nonrelativistic matter responds to ``Phi`` while light responds to the
    average of ``Phi`` and ``Psi``.  The closure therefore leaves dynamics
    unchanged and modifies only the extra, non-Newtonian acceleration.
    """
    gbar = np.asarray(baryonic_acceleration_m_s2, dtype=float)
    gdyn = np.asarray(dynamical_acceleration_m_s2, dtype=float)
    field = np.asarray(sigma, dtype=float)
    if not (gbar.shape == gdyn.shape == field.shape):
        raise ValueError("gbar, gdyn, and sigma must have matching shapes")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(gdyn)) or np.any(gdyn < gbar):
        raise ValueError("gdyn must be finite and no smaller than gbar")
    if np.any(~np.isfinite(field)) or np.any(field < 0.0):
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(zeta):
        raise ValueError("zeta must be finite")
    extra = gdyn - gbar
    return gbar + (1.0 + 0.5 * float(zeta) * field**2) * extra
