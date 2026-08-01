from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constitutive import standard_mu_acceleration
from .unified import A0_M_S2, C_M_S


def standard_action_function(y) -> np.ndarray:
    """Action primitive whose derivative is the standard MOND mu function."""
    value = np.asarray(y, dtype=float)
    if np.any(value < 0.0):
        raise ValueError("the kinetic invariant y must be non-negative")
    return np.sqrt(value * (1.0 + value)) - np.arcsinh(np.sqrt(value))


def standard_mu_from_y(y) -> np.ndarray:
    value = np.asarray(y, dtype=float)
    if np.any(value < 0.0):
        raise ValueError("the kinetic invariant y must be non-negative")
    return np.sqrt(value / (1.0 + value))


def aether_action_function(y) -> np.ndarray:
    """GEA term H=2(y-F_s) that yields mu_s=1-H_y/2."""
    value = np.asarray(y, dtype=float)
    return 2.0 * (value - standard_action_function(value))


def aether_feedback_shape(y) -> np.ndarray:
    """Return H-y H_y with a cancellation-safe small-y expansion."""
    value = np.asarray(y, dtype=float)
    if np.any(value < 0.0):
        raise ValueError("the kinetic invariant y must be non-negative")
    output = np.empty_like(value)
    small = value < 1e-4
    small_value = value[small]
    output[small] = (
        (2.0 / 3.0) * np.power(small_value, 1.5)
        - (3.0 / 5.0) * np.power(small_value, 2.5)
        + (15.0 / 28.0) * np.power(small_value, 3.5)
    )
    regular = value[~small]
    if regular.size:
        output[~small] = 2.0 * (
            regular * standard_mu_from_y(regular) - standard_action_function(regular)
        )
    return output


def required_exponential_coupling(frozen_amplitude: float, transition_chi: float) -> float:
    """Match a_Q(chi_t)=a0*sqrt(F) for a_Q=a0*exp(eta Q)."""
    if frozen_amplitude <= 1.0 or transition_chi <= 0.0:
        raise ValueError("require frozen_amplitude > 1 and transition_chi > 0")
    return float(0.5 * np.log(frozen_amplitude) / transition_chi)


def environment_acceleration_scale(q, eta_per_chi: float) -> np.ndarray:
    value = np.asarray(q, dtype=float)
    if np.any(value < 0.0) or eta_per_chi < 0.0:
        raise ValueError("q and eta must be non-negative")
    exponent = eta_per_chi * value
    if np.any(exponent > 700.0):
        raise OverflowError("the exponential environment response overflowed")
    return A0_M_S2 * np.exp(exponent)


def aether_feedback_energy_scale(gbar_m_s2, a_q_m_s2) -> np.ndarray:
    """Positive a_Q^2(H-YH_Y) factor before eta/(2 beta c^4)."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    a_q = np.asarray(a_q_m_s2, dtype=float)
    if np.any(gbar <= 0.0) or np.any(a_q <= 0.0):
        raise ValueError("gbar and a_q must be positive")
    total_g = standard_mu_acceleration(gbar, a_q)
    y = np.square(total_g / a_q)
    return np.square(a_q) * aether_feedback_shape(y)


def scalar_tensor_gamma_minus_one(beta: float, q: float = 0.0) -> float:
    """Jordan-frame massless-scalar PPN gamma shift for F=1+2 beta Q, Z=2 beta."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    coupling = 1.0 + 2.0 * beta * q
    if coupling <= 0.0:
        raise ValueError("the effective Planck-mass factor must be positive")
    numerator = 4.0 * beta**2
    denominator = 2.0 * beta * coupling + 8.0 * beta**2
    return float(-numerator / denominator)


def beta_from_gamma_bound(abs_gamma_minus_one_max: float) -> float:
    if not 0.0 < abs_gamma_minus_one_max < 0.5:
        raise ValueError("the gamma bound must lie between zero and one half")
    return float(abs_gamma_minus_one_max / (2.0 - 4.0 * abs_gamma_minus_one_max))


def ppn_restricted_aether_coefficients(c1: float, c14: float) -> dict[str, float]:
    """Set c13=0 and alpha2=0 in the high-field Einstein-Aether floor."""
    if not 0.0 < c14 <= c1 or c14 >= 0.5:
        raise ValueError("require 0 < c14 <= c1 and c14 < 1/2")
    c3 = -c1
    c2 = c14 / (1.0 - 2.0 * c14)
    c4 = c14 - c1
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c13": c1 + c3}


def high_field_mode_speeds_squared(c1: float, c14: float) -> dict[str, float]:
    coefficients = ppn_restricted_aether_coefficients(c1, c14)
    c2 = coefficients["c2"]
    scalar = c2 * (2.0 - c14) / (c14 * (2.0 + 3.0 * c2))
    return {"tensor": 1.0, "vector": c1 / c14, "scalar": float(scalar)}


def point_source_minimum_range_over_radius(max_fractional_error: float) -> float:
    """Minimum L/r for exp(-r/L) to differ from a 1/r potential by at most error."""
    if not 0.0 < max_fractional_error < 1.0:
        raise ValueError("max_fractional_error must lie between zero and one")
    return float(1.0 / -np.log1p(-max_fractional_error))


def exterior_feedback_ratio(
    *,
    radius_m: float,
    gbar_m_s2: float,
    target_chi: float,
    eta_per_chi: float,
    beta: float,
    range_over_radius: float,
    grid_points: int = 6000,
) -> float:
    """Conservative exterior correction from the action-required Q source.

    Only the baryonic mass already enclosed at the measurement radius is
    continued outward as a point source. Positive exterior baryons would raise
    both q and gbar. The Yukawa range is supplied in units of the measurement
    radius.
    """
    if (
        radius_m <= 0.0
        or gbar_m_s2 <= 0.0
        or target_chi <= 0.0
        or eta_per_chi < 0.0
        or beta <= 0.0
        or range_over_radius <= 0.0
        or grid_points < 100
    ):
        raise ValueError("invalid exterior-feedback input")
    maximum_ratio = max(1e3, 50.0 * range_over_radius)
    radius = radius_m * np.geomspace(1.0, maximum_ratio, grid_points)
    radial_ratio = radius_m / radius
    exterior_gbar = gbar_m_s2 * np.square(radial_ratio)
    enclosed_q = gbar_m_s2 * radius_m * radial_ratio / C_M_S**2
    a_q = environment_acceleration_scale(enclosed_q, eta_per_chi)
    source = (
        eta_per_chi
        * aether_feedback_energy_scale(exterior_gbar, a_q)
        / C_M_S**4
        / (2.0 * beta)
    )
    yukawa_kernel = (
        np.exp(-radius / (range_over_radius * radius_m))
        * np.sinh(1.0 / range_over_radius)
        * range_over_radius
    )
    delta_q = np.trapezoid(radius * source * yukawa_kernel, radius)
    return float(delta_q / target_chi)


def maximum_eta_for_feedback_gate(
    rows,
    *,
    beta: float,
    range_over_radius: float,
    maximum_fraction: float,
    grid_points: int = 3000,
) -> float:
    """Largest eta for which every supplied point clears the feedback gate."""
    prepared = list(rows)
    if not prepared:
        raise ValueError("at least one row is required")

    def maximum_ratio(log10_eta: float) -> float:
        eta = 10.0**log10_eta
        return max(
            exterior_feedback_ratio(
                radius_m=float(row[0]),
                gbar_m_s2=float(row[1]),
                target_chi=float(row[2]),
                eta_per_chi=eta,
                beta=beta,
                range_over_radius=range_over_radius,
                grid_points=grid_points,
            )
            for row in prepared
        )

    root = brentq(lambda value: maximum_ratio(value) - maximum_fraction, -10.0, 8.0)
    return float(10.0**root)
