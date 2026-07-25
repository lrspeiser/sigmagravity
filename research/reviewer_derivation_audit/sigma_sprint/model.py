"""Canonical response and its nonrelativistic QUMOND potential.

With ``z = |grad(Phi_N)|^2 / g_dagger^2`` and an independently varied order
parameter ``B``, the gravitational part of the action density is

    -(8 pi G)^-1 [2 grad(Phi).grad(Phi_N) - g_dagger^2 Q(z, B)].

Variation with respect to Phi gives

    laplacian(Phi_N) = 4 pi G rho,

and variation with respect to Phi_N gives

    laplacian(Phi) = div[Q_z grad(Phi_N)].

Variation with respect to B also requires

    Euler_B(L_B) + g_dagger^2 Q_B/(8 pi G) = 0.

The Q_B term cannot be omitted in a genuinely dynamical completion.

Noether energy, momentum, and angular-momentum conservation follows only when
B is itself a dynamical field in the same translation- and rotation-invariant
action and all fields are varied.  It does *not* follow when B is inserted
from observed or model-predicted velocities.
"""

from __future__ import annotations

import numpy as np

G_SI = 6.67430e-11
C_LIGHT = 2.998e8
H0_SI = 2.27e-18
DEFAULT_G_DAGGER = C_LIGHT * H0_SI / (4.0 * np.sqrt(np.pi))
DEFAULT_A0 = float(np.exp(1.0 / (2.0 * np.pi)))


def _return_like_input(value: np.ndarray, original: object):
    return float(value) if np.ndim(original) == 0 else value


def enhancement_kernel(g_newton, g_dagger: float = DEFAULT_G_DAGGER):
    """Return h(g_N), with the g_N=0 limit represented by infinity."""
    original = g_newton
    g = np.asarray(g_newton, dtype=float)
    if np.any(g < 0) or g_dagger <= 0:
        raise ValueError("accelerations and g_dagger must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)
    return _return_like_input(h, original)


def nu(g_newton, B, g_dagger: float = DEFAULT_G_DAGGER):
    """QUMOND response Q_z = 1 + B h(g_N)."""
    g = np.asarray(g_newton, dtype=float)
    b = np.asarray(B, dtype=float)
    result = 1.0 + b * enhancement_kernel(g, g_dagger)
    return _return_like_input(np.asarray(result), g_newton)


def _u_minus_atan_u(z_array):
    u = np.power(z_array, 0.25)
    # Direct subtraction loses several digits in the deep limit.  The series
    # is alternating and rapidly convergent for this deliberately small cut.
    difference = u - np.arctan(u)
    small = u < 0.05
    if np.any(small):
        u_small = u[small] if u.ndim else np.asarray([u])
        series = np.zeros_like(u_small)
        for power in range(3, 18, 2):
            series += ((-1.0) ** ((power - 3) // 2)) * u_small**power / power
        if u.ndim:
            difference = np.asarray(difference)
            difference[small] = series
        else:
            difference = series[0]
    return difference


def q_B(z):
    """Analytic partial derivative dQ/dB, evaluated without large-z subtraction."""
    original = z
    z_array = np.asarray(z, dtype=float)
    if np.any(z_array < 0):
        raise ValueError("z must be non-negative")
    result = 4.0 * _u_minus_atan_u(z_array)
    return _return_like_input(np.asarray(result), original)


def q_potential(z, B):
    """Closed Q(z,B) whose z derivative is exactly the canonical response."""
    original = z
    z_array = np.asarray(z, dtype=float)
    if np.any(z_array < 0):
        raise ValueError("z must be non-negative")
    result = z_array + np.asarray(B, dtype=float) * q_B(z_array)
    return _return_like_input(np.asarray(result), original)


def q_z(z, B):
    """Analytic partial derivative dQ/dz."""
    original = z
    z_array = np.asarray(z, dtype=float)
    if np.any(z_array < 0):
        raise ValueError("z must be non-negative")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1.0 + np.asarray(B, dtype=float) * (
            np.power(z_array, -0.25) / (1.0 + np.sqrt(z_array))
        )
    return _return_like_input(np.asarray(result), original)


def infer_B(gbar, gtot, g_dagger: float = DEFAULT_G_DAGGER):
    """Infer the empirical order field B_obs from independent accelerations."""
    original = gbar
    gbar_array = np.asarray(gbar, dtype=float)
    gtot_array = np.asarray(gtot, dtype=float)
    if np.any(gbar_array <= 0) or np.any(gtot_array < 0):
        raise ValueError("gbar must be positive and gtot non-negative")
    result = (gtot_array / gbar_array - 1.0) / enhancement_kernel(
        gbar_array, g_dagger
    )
    return _return_like_input(np.asarray(result), original)


def predict_acceleration(gbar, B, g_dagger: float = DEFAULT_G_DAGGER):
    """Algebraic acceleration used by the submitted phenomenology."""
    gbar_array = np.asarray(gbar, dtype=float)
    result = gbar_array * nu(gbar_array, B, g_dagger)
    return _return_like_input(np.asarray(result), gbar)


def deep_btfr_velocity4(
    baryonic_mass, B, g_dagger: float = DEFAULT_G_DAGGER, G: float = G_SI
):
    """Deep-acceleration result V^4 = B^2 G M_b g_dagger."""
    mass = np.asarray(baryonic_mass, dtype=float)
    result = np.asarray(B, dtype=float) ** 2 * G * mass * g_dagger
    return _return_like_input(np.asarray(result), baryonic_mass)


def amplitude_from_path_length(
    length_kpc, *, A0: float = DEFAULT_A0, L0_kpc: float = 0.4, n: float = 0.27
):
    """Submitted empirical amplitude relation, retained only for auditing."""
    length = np.asarray(length_kpc, dtype=float)
    if np.any(length <= 0) or A0 <= 0 or L0_kpc <= 0:
        raise ValueError("lengths and amplitudes must be positive")
    result = A0 * np.power(length / L0_kpc, n)
    return _return_like_input(np.asarray(result), length_kpc)
