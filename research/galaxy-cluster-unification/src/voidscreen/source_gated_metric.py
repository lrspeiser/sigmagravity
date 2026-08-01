"""A one-parameter source-distribution gate for matter and lensing tests."""

from __future__ import annotations

import math

import numpy as np


def radial_source_concentration(
    radius,
    gbar_m_s2,
    *,
    maximum_mass_slope: float = 3.0,
) -> np.ndarray:
    """Estimate how centrally concentrated a radial baryonic source is.

    For a spherical equivalent profile, ``M(<r)`` is proportional to
    ``g_bar*r**2``.  The returned statistic is

    ``C = M/(M + dM/dln(r)) = 1/(1 + dln(M)/dln(r))``.

    A finite central source approaches ``C=1`` outside its mass, while a
    spatially distributed source has a lower value.  The mass slope is clipped
    to the physical spherical interval used by the frozen protocol.
    """
    r = np.asarray(radius, dtype=float)
    gbar = np.asarray(gbar_m_s2, dtype=float)
    r, gbar = np.broadcast_arrays(r, gbar)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("radius and gbar must be one-dimensional with at least two points")
    if np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("radius must be finite and positive")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError("radius must be strictly increasing")
    if not math.isfinite(maximum_mass_slope) or maximum_mass_slope <= 0.0:
        raise ValueError("maximum_mass_slope must be finite and positive")

    log_radius = np.log(r)
    log_equivalent_mass = np.log(gbar) + 2.0 * log_radius
    edge_order = 2 if len(r) >= 3 else 1
    mass_slope = np.gradient(log_equivalent_mass, log_radius, edge_order=edge_order)
    mass_slope = np.clip(mass_slope, 0.0, float(maximum_mass_slope))
    return 1.0 / (1.0 + mass_slope)


def source_gate(
    gbar_m_s2,
    concentration,
    *,
    acceleration_scale_m_s2: float,
    diffuseness_power: float = 2.0,
    concentration_power: float = 2.0,
) -> np.ndarray:
    """Return ``F=D**p*(1-C)**q`` with no fitted scale beyond fixed RAR."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    coherence = np.asarray(concentration, dtype=float)
    gbar, coherence = np.broadcast_arrays(gbar, coherence)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(coherence)) or np.any((coherence < 0.0) | (coherence > 1.0)):
        raise ValueError("concentration must lie in [0, 1]")
    if not math.isfinite(acceleration_scale_m_s2) or acceleration_scale_m_s2 <= 0.0:
        raise ValueError("acceleration scale must be finite and positive")
    for value, label in (
        (diffuseness_power, "diffuseness_power"),
        (concentration_power, "concentration_power"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    diffuseness = acceleration_scale_m_s2 / (
        acceleration_scale_m_s2 + 2.0 * gbar
    )
    return np.power(diffuseness, diffuseness_power) * np.power(
        1.0 - coherence, concentration_power
    )


def gated_extra_acceleration(
    gbar_m_s2,
    gdyn_m_s2,
    gate,
    kappa: float,
) -> np.ndarray:
    """Add ``kappa*F`` times the fixed-RAR non-Newtonian acceleration."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    gdyn = np.asarray(gdyn_m_s2, dtype=float)
    weight = np.asarray(gate, dtype=float)
    gbar, gdyn, weight = np.broadcast_arrays(gbar, gdyn, weight)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(gdyn)) or np.any(gdyn < gbar):
        raise ValueError("gdyn must be finite and at least gbar")
    if np.any(~np.isfinite(weight)) or np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("gate must lie in [0, 1]")
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    return gdyn + float(kappa) * weight * (gdyn - gbar)


def source_gated_metric_eta(
    gbar_m_s2,
    gdyn_m_s2,
    gate,
    kappa: float,
) -> np.ndarray:
    """Return ``Psi/Phi`` for ``Psi=Phi_N+(1+2*kappa*F)*phi``."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    gdyn = np.asarray(gdyn_m_s2, dtype=float)
    weight = np.asarray(gate, dtype=float)
    gbar, gdyn, weight = np.broadcast_arrays(gbar, gdyn, weight)
    gated_extra_acceleration(gbar, gdyn, weight, kappa)
    return 1.0 + 2.0 * float(kappa) * weight * (gdyn - gbar) / gdyn
