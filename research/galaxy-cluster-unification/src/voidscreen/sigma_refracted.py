"""Formula-level controls for Sigma Gravity and Refracted Gravity hybrids.

The functions in this module do not constitute a relativistic theory.  They
make the algebraic and spherical limits of the two weak-field proposals
explicit so candidate combinations can be rejected before expensive field
solutions or observational fits.
"""

from __future__ import annotations

import math

import numpy as np


DEFAULT_G_DAGGER_M_S2 = 9.60e-11


def _positive_finite(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def sigma_h(
    g_newton_m_s2,
    *,
    g_dagger_m_s2: float = DEFAULT_G_DAGGER_M_S2,
) -> np.ndarray:
    """Return the locked Sigma transition function h(g_N)."""
    g_newton = _positive_finite(g_newton_m_s2, "g_newton_m_s2")
    if not math.isfinite(g_dagger_m_s2) or g_dagger_m_s2 <= 0.0:
        raise ValueError("g_dagger_m_s2 must be finite and positive")
    return np.sqrt(g_dagger_m_s2 / g_newton) * (
        g_dagger_m_s2 / (g_dagger_m_s2 + g_newton)
    )


def sigma_enhancement(
    g_newton_m_s2,
    response_amplitude: float,
    *,
    g_dagger_m_s2: float = DEFAULT_G_DAGGER_M_S2,
) -> np.ndarray:
    """Return Sigma=1+B h(g_N)."""
    if not math.isfinite(response_amplitude) or response_amplitude < 0.0:
        raise ValueError("response_amplitude must be finite and nonnegative")
    return 1.0 + response_amplitude * sigma_h(
        g_newton_m_s2, g_dagger_m_s2=g_dagger_m_s2
    )


def refracted_permittivity(
    density,
    *,
    minimum_permittivity: float,
    critical_density: float,
    rg_sharpness: float,
) -> np.ndarray:
    """Published smooth Refracted Gravity permittivity.

    This evaluates

        epsilon = epsilon_0 + (1-epsilon_0)/2
                  * {tanh[Q ln(rho/rho_c)] + 1}.

    The logaddexp implementation remains stable across very large density
    ratios and sharp transitions.
    """
    rho = _positive_finite(density, "density")
    if (
        not math.isfinite(minimum_permittivity)
        or minimum_permittivity <= 0.0
        or minimum_permittivity > 1.0
    ):
        raise ValueError("minimum_permittivity must be in (0, 1]")
    if not math.isfinite(critical_density) or critical_density <= 0.0:
        raise ValueError("critical_density must be finite and positive")
    if not math.isfinite(rg_sharpness) or rg_sharpness <= 0.0:
        raise ValueError("rg_sharpness must be finite and positive")
    log_odds = 2.0 * rg_sharpness * np.log(rho / critical_density)
    transition = np.exp(-np.logaddexp(0.0, -log_odds))
    return minimum_permittivity + (1.0 - minimum_permittivity) * transition


def nbp0_sharpness_from_rg(rg_sharpness: float) -> float:
    """Map published RG Q to the logistic exponent used by NBP0."""
    if not math.isfinite(rg_sharpness) or rg_sharpness <= 0.0:
        raise ValueError("rg_sharpness must be finite and positive")
    return 2.0 * rg_sharpness


def coherence_weight(coherence) -> np.ndarray:
    """Parameter-free smooth channel weight w(C)=3 C^2-2 C^3."""
    value = np.asarray(coherence, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("coherence must be finite and in [0, 1]")
    return np.square(value) * (3.0 - 2.0 * value)


def coherence_partitioned_coefficients(
    g_newton_m_s2,
    density,
    coherence,
    *,
    response_amplitude: float,
    minimum_permittivity: float,
    critical_density: float,
    rg_sharpness: float,
    g_dagger_m_s2: float = DEFAULT_G_DAGGER_M_S2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (epsilon_mix, nu_source) for the CPR0 diagnostic equation.

    CPR0 is the conservative weak-field interpolation

      div(epsilon_mix grad Phi) = div(nu_source grad Phi_N),

    with epsilon_mix=w+(1-w)epsilon_RG and
    nu_source=1+w B h(g_N).  C=1 recovers the fixed-B Sigma/QUMOND
    equation and C=0 recovers phenomenological Refracted Gravity.
    """
    g_newton = _positive_finite(g_newton_m_s2, "g_newton_m_s2")
    rho = _positive_finite(density, "density")
    g_newton, rho, weight = np.broadcast_arrays(
        g_newton, rho, coherence_weight(coherence)
    )
    epsilon_rg = refracted_permittivity(
        rho,
        minimum_permittivity=minimum_permittivity,
        critical_density=critical_density,
        rg_sharpness=rg_sharpness,
    )
    epsilon_mix = weight + (1.0 - weight) * epsilon_rg
    nu_source = 1.0 + weight * response_amplitude * sigma_h(
        g_newton, g_dagger_m_s2=g_dagger_m_s2
    )
    return epsilon_mix, nu_source


def coherence_partitioned_spherical_enhancement(
    g_newton_m_s2,
    density,
    coherence,
    **parameters,
) -> np.ndarray:
    """Return g/g_N=nu_source/epsilon_mix in spherical symmetry."""
    epsilon_mix, nu_source = coherence_partitioned_coefficients(
        g_newton_m_s2, density, coherence, **parameters
    )
    return nu_source / epsilon_mix


def naive_product_enhancement(
    g_newton_m_s2,
    density,
    *,
    response_amplitude: float,
    minimum_permittivity: float,
    critical_density: float,
    rg_sharpness: float,
    g_dagger_m_s2: float = DEFAULT_G_DAGGER_M_S2,
) -> np.ndarray:
    """Diagnostic only: the double-counting product Sigma/epsilon_RG."""
    return sigma_enhancement(
        g_newton_m_s2,
        response_amplitude,
        g_dagger_m_s2=g_dagger_m_s2,
    ) / refracted_permittivity(
        density,
        minimum_permittivity=minimum_permittivity,
        critical_density=critical_density,
        rg_sharpness=rg_sharpness,
    )


def additive_susceptibility_enhancement(
    g_newton_m_s2,
    density,
    *,
    response_amplitude: float,
    minimum_permittivity: float,
    critical_density: float,
    rg_sharpness: float,
    g_dagger_m_s2: float = DEFAULT_G_DAGGER_M_S2,
) -> np.ndarray:
    """Diagnostic only: add rather than multiply the two susceptibilities."""
    sigma = sigma_enhancement(
        g_newton_m_s2,
        response_amplitude,
        g_dagger_m_s2=g_dagger_m_s2,
    )
    epsilon = refracted_permittivity(
        density,
        minimum_permittivity=minimum_permittivity,
        critical_density=critical_density,
        rg_sharpness=rg_sharpness,
    )
    return sigma + 1.0 / epsilon - 1.0
