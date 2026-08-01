"""Spherical proxy mappings for the tidal-tensor response."""

from __future__ import annotations

import numpy as np


def acceleration_gate(gbar_m_s2, *, a0_m_s2: float, power: float) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if np.any(gbar <= 0.0) or a0_m_s2 <= 0.0 or power <= 0.0:
        raise ValueError("accelerations and power must be positive")
    return 1.0 / (1.0 + np.power(gbar / a0_m_s2, power))


def spherical_boost(
    gbar_m_s2,
    *,
    kappa: float,
    family: str,
    gate_power: float,
    a0_m_s2: float,
    radial_q: float = 2.0 / 3.0,
) -> np.ndarray:
    """Return g/gbar for a radial tidal eigenvalue q."""
    if kappa < 0.0 or not 0.0 < radial_q <= 1.0:
        raise ValueError("kappa must be non-negative and q in (0,1]")
    gate = acceleration_gate(
        gbar_m_s2, a0_m_s2=a0_m_s2, power=gate_power
    )
    response = kappa * radial_q * gate
    if family == "linear":
        if kappa >= 1.0:
            raise ValueError("linear family requires kappa < 1")
        return 1.0 / (1.0 - response)
    if family == "reciprocal":
        return 1.0 + response
    if family == "exponential":
        return np.exp(response)
    raise ValueError(f"unknown spherical response family: {family}")
