"""Bounded algebraic response laws for the galaxy/cluster formula sweep.

These functions are phenomenological weak-field diagnostics, not relativistic
field equations.  Every model returns the spherical acceleration enhancement
``g_pred / g_bar`` and is constrained to remain positive.
"""

from __future__ import annotations

import math

import numpy as np

from .data import KPC_M
from .sigma_refracted import coherence_weight, sigma_h

C_M_S = 299_792_458.0


def _positive(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def fixed_rar_enhancement(g_bar_m_s2, acceleration_scale_m_s2: float) -> np.ndarray:
    """Return the McGaugh et al. empirical RAR enhancement."""
    g_bar = _positive(g_bar_m_s2, "g_bar_m_s2")
    if not math.isfinite(acceleration_scale_m_s2) or acceleration_scale_m_s2 <= 0.0:
        raise ValueError("acceleration_scale_m_s2 must be finite and positive")
    root = np.sqrt(g_bar / acceleration_scale_m_s2)
    denominator = -np.expm1(-root)
    return 1.0 / np.maximum(denominator, np.finfo(float).tiny)


def simple_mond_enhancement(g_bar_m_s2, acceleration_scale_m_s2: float) -> np.ndarray:
    """Return the algebraic MOND relation associated with mu(x)=x/(1+x)."""
    g_bar = _positive(g_bar_m_s2, "g_bar_m_s2")
    if not math.isfinite(acceleration_scale_m_s2) or acceleration_scale_m_s2 <= 0.0:
        raise ValueError("acceleration_scale_m_s2 must be finite and positive")
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * acceleration_scale_m_s2 / g_bar))


def dimensionless_baryonic_potential(g_bar_m_s2, radius_kpc) -> np.ndarray:
    """Return the local proxy g_bar*r/c^2 used by the potential-threshold test."""
    g_bar = _positive(g_bar_m_s2, "g_bar_m_s2")
    radius = _positive(radius_kpc, "radius_kpc")
    g_bar, radius = np.broadcast_arrays(g_bar, radius)
    return g_bar * radius * KPC_M / C_M_S**2


def _logistic(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))


def _varying_floor_permittivity(
    density,
    g_bar_m_s2,
    *,
    minimum_permittivity: float,
    critical_density: float,
    sharpness: float,
    acceleration_coupling: float,
    g_reference_m_s2: float,
) -> np.ndarray:
    rho = _positive(density, "density")
    g_bar = _positive(g_bar_m_s2, "g_bar_m_s2")
    if not 0.0 < minimum_permittivity < 1.0:
        raise ValueError("minimum_permittivity must be in (0, 1)")
    base_logit = math.log(minimum_permittivity / (1.0 - minimum_permittivity))
    floor = _logistic(
        base_logit + acceleration_coupling * np.log(g_bar / g_reference_m_s2)
    )
    log_odds = 2.0 * sharpness * np.log(rho / critical_density)
    transition = _logistic(log_odds)
    return floor + (1.0 - floor) * transition


def _permittivity_with_local_threshold(
    density: np.ndarray,
    critical_density: np.ndarray,
    *,
    minimum_permittivity: float,
    sharpness: float,
) -> np.ndarray:
    """Evaluate the RG logistic while allowing a point-dependent threshold."""
    if not 0.0 < minimum_permittivity <= 1.0:
        raise ValueError("minimum_permittivity must be in (0, 1]")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError("sharpness must be finite and positive")
    critical = _positive(critical_density, "critical_density")
    log_odds = 2.0 * sharpness * np.log(density / critical)
    transition = _logistic(log_odds)
    return minimum_permittivity + (1.0 - minimum_permittivity) * transition


def response_enhancement(
    model: str,
    g_bar_m_s2,
    density_g_cm3,
    radius_kpc,
    parameters,
    *,
    g_reference_m_s2: float = 1.0e-10,
    potential_reference: float = 1.0e-5,
    sigma_g_dagger_m_s2: float = 9.6e-11,
    rar_acceleration_m_s2: float = 1.2e-10,
    fixed_gate_log10_phi_c: float = -6.3,
    fixed_gate_sharpness: float = 4.0,
    coherence=0.0,
    coherence_gate_power: float = 2.0,
) -> np.ndarray:
    """Evaluate one preregistered response model.

    Parameter order is recorded in ``phenomenology_formula_sweep_protocol.json``.
    """
    g_bar = _positive(g_bar_m_s2, "g_bar_m_s2")
    rho = _positive(density_g_cm3, "density_g_cm3")
    radius = _positive(radius_kpc, "radius_kpc")
    g_bar, rho, radius = np.broadcast_arrays(g_bar, rho, radius)
    values = np.asarray(parameters, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("parameters must be finite")

    if model in {
        "RG",
        "RG_acceleration_threshold",
        "RG_potential_threshold",
        "RG_acceleration_floor",
        "RG_Sigma_additive",
        "RG_Sigma_quadrature",
        "RG_Sigma_product",
        "RG_density_gated_Sigma",
        "RAR_RG_additive",
        "RAR_RG_quadrature",
        "RAR_RG_product",
        "RAR_potential_gated_RG",
        "RAR_fixed_potential_gated_RG",
        "RAR_coherence_gated_RG",
        "RAR_sharp_coherence_gated_RG",
    }:
        epsilon_0, log10_rho_c, sharpness = values[:3]
    else:
        raise ValueError(f"unknown phenomenology model {model}")

    critical_density = np.full_like(rho, 10.0**float(log10_rho_c))
    if model == "RG_acceleration_threshold":
        critical_density *= np.power(g_bar / g_reference_m_s2, float(values[3]))
    elif model == "RG_potential_threshold":
        potential = dimensionless_baryonic_potential(g_bar, radius)
        critical_density *= np.power(potential / potential_reference, float(values[3]))

    if model == "RG_acceleration_floor":
        epsilon = _varying_floor_permittivity(
            rho,
            g_bar,
            minimum_permittivity=float(epsilon_0),
            critical_density=critical_density,
            sharpness=float(sharpness),
            acceleration_coupling=float(values[3]),
            g_reference_m_s2=g_reference_m_s2,
        )
    else:
        epsilon = _permittivity_with_local_threshold(
            rho,
            critical_density,
            minimum_permittivity=float(epsilon_0),
            sharpness=float(sharpness),
        )

    density_susceptibility = 1.0 / epsilon - 1.0
    if model in {
        "RG",
        "RG_acceleration_threshold",
        "RG_potential_threshold",
        "RG_acceleration_floor",
    }:
        return 1.0 + density_susceptibility

    rar_susceptibility = fixed_rar_enhancement(
        g_bar, rar_acceleration_m_s2
    ) - 1.0
    if model == "RAR_RG_additive":
        return 1.0 + rar_susceptibility + density_susceptibility
    if model == "RAR_RG_quadrature":
        return 1.0 + np.hypot(rar_susceptibility, density_susceptibility)
    if model == "RAR_RG_product":
        return (1.0 + rar_susceptibility) / epsilon
    if model == "RAR_potential_gated_RG":
        log10_phi_c, gate_sharpness = values[3:5]
        potential = dimensionless_baryonic_potential(g_bar, radius)
        gate = _logistic(
            float(gate_sharpness)
            * np.log(potential / np.power(10.0, float(log10_phi_c)))
        )
        return 1.0 + rar_susceptibility + gate * density_susceptibility
    if model == "RAR_fixed_potential_gated_RG":
        potential = dimensionless_baryonic_potential(g_bar, radius)
        gate = _logistic(
            float(fixed_gate_sharpness)
            * np.log(potential / np.power(10.0, float(fixed_gate_log10_phi_c)))
        )
        return 1.0 + rar_susceptibility + gate * density_susceptibility
    if model == "RAR_coherence_gated_RG":
        weight = np.broadcast_to(coherence_weight(coherence), g_bar.shape)
        return 1.0 + rar_susceptibility + (1.0 - weight) * density_susceptibility
    if model == "RAR_sharp_coherence_gated_RG":
        if not math.isfinite(coherence_gate_power) or coherence_gate_power <= 0.0:
            raise ValueError("coherence_gate_power must be finite and positive")
        weight = np.broadcast_to(coherence_weight(coherence), g_bar.shape)
        return (
            1.0
            + rar_susceptibility
            + np.power(1.0 - weight, coherence_gate_power) * density_susceptibility
        )

    sigma_susceptibility = float(values[3]) * sigma_h(
        g_bar, g_dagger_m_s2=sigma_g_dagger_m_s2
    )
    if model == "RG_Sigma_additive":
        return 1.0 + density_susceptibility + sigma_susceptibility
    if model == "RG_Sigma_quadrature":
        return 1.0 + np.hypot(density_susceptibility, sigma_susceptibility)
    if model == "RG_Sigma_product":
        return (1.0 + sigma_susceptibility) / epsilon
    if model == "RG_density_gated_Sigma":
        gate_power = float(values[4])
        return 1.0 + density_susceptibility + sigma_susceptibility * epsilon**gate_power
    raise AssertionError("unreachable model branch")
