"""Potential-depth channel exponent for a QUMOND-style source boost."""

from __future__ import annotations

import numpy as np


def rar_qumond_boost(gbar_m_s2, a0_m_s2: float) -> np.ndarray:
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if not np.isfinite(a0_m_s2) or a0_m_s2 <= 0.0:
        raise ValueError("a0_m_s2 must be finite and positive")
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar_m_s2 must be finite and positive")
    denominator = -np.expm1(-np.sqrt(gbar / float(a0_m_s2)))
    return 1.0 / denominator


def potential_channel_exponent(
    potential_depth,
    *,
    transition_depth: float,
    transition_power: float,
    endpoint_exponent: float,
) -> np.ndarray:
    depth = np.asarray(potential_depth, dtype=float)
    if np.any(~np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError("potential_depth must be finite and non-negative")
    if not np.isfinite(transition_depth) or transition_depth <= 0.0:
        raise ValueError("transition_depth must be finite and positive")
    if not np.isfinite(transition_power) or transition_power <= 0.0:
        raise ValueError("transition_power must be finite and positive")
    if not np.isfinite(endpoint_exponent) or endpoint_exponent < 1.0:
        raise ValueError("endpoint_exponent must be finite and at least one")
    coordinate = np.power(depth / float(transition_depth), float(transition_power))
    activation = coordinate / (1.0 + coordinate)
    return 1.0 + (float(endpoint_exponent) - 1.0) * activation


def potential_channel_acceleration(
    gbar_m_s2,
    potential_depth,
    *,
    a0_m_s2: float,
    transition_depth: float,
    transition_power: float,
    endpoint_exponent: float,
) -> dict[str, np.ndarray]:
    gbar, depth = np.broadcast_arrays(
        np.asarray(gbar_m_s2, dtype=float),
        np.asarray(potential_depth, dtype=float),
    )
    boost = rar_qumond_boost(gbar, a0_m_s2)
    exponent = potential_channel_exponent(
        depth,
        transition_depth=transition_depth,
        transition_power=transition_power,
        endpoint_exponent=endpoint_exponent,
    )
    enhancement = np.power(boost, exponent)
    predicted = gbar * enhancement
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise ValueError("predicted acceleration must be finite and positive")
    return {
        "base_qumond_boost": boost,
        "channel_exponent": exponent,
        "enhancement": enhancement,
        "predicted_acceleration_m_s2": predicted,
    }

