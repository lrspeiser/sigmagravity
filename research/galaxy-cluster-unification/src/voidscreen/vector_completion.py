"""Bounded gravitational-vector completion laws."""

from __future__ import annotations

import numpy as np

from .data import KPC_M


def tidal_curvature_proxy(gbar_m_s2, radius_kpc) -> np.ndarray:
    """Return the radial weak-field tidal proxy ``g_bar/r`` in s^-2."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    gbar, radius = np.broadcast_arrays(gbar, radius)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    return gbar / (radius * KPC_M)


def bounded_completion(
    tidal_s2,
    *,
    solar_completion: float,
    tidal_transition_s2: float,
    transition_power: float,
    coherence=None,
    coherence_power: float | None = None,
) -> dict[str, np.ndarray]:
    """Convert a locally calibrated field into a bounded completed field.

    ``solar_completion`` is the fraction of a proposed maximum gravitational
    coupling represented by the locally measured value of G.  The completion
    fraction rises from that value at high curvature to at most one at low
    curvature.  Returned enhancement is relative to locally calibrated G.

    If coherence is supplied, the low-curvature activation is multiplied by
    ``(1-coherence)**coherence_power``.  This is an explicitly diagnostic
    proxy for vectors that remain unavailable to the sum; it is not assumed
    to be the final geometric definition of vector coherence.
    """
    tidal = np.asarray(tidal_s2, dtype=float)
    if np.any(~np.isfinite(tidal)) or np.any(tidal <= 0.0):
        raise ValueError("tidal curvature must be finite and positive")
    if not 0.0 < solar_completion <= 1.0:
        raise ValueError("solar_completion must lie in (0, 1]")
    if not np.isfinite(tidal_transition_s2) or tidal_transition_s2 <= 0.0:
        raise ValueError("tidal_transition_s2 must be finite and positive")
    if not np.isfinite(transition_power) or transition_power <= 0.0:
        raise ValueError("transition_power must be finite and positive")

    ratio = tidal / float(tidal_transition_s2)
    with np.errstate(over="ignore"):
        low_curvature_activation = 1.0 / (
            1.0 + np.power(ratio, float(transition_power))
        )
    availability = np.ones_like(low_curvature_activation)
    if coherence is not None:
        coherent = np.asarray(coherence, dtype=float)
        coherent = np.broadcast_to(coherent, tidal.shape)
        if np.any(~np.isfinite(coherent)) or np.any((coherent < 0.0) | (coherent > 1.0)):
            raise ValueError("coherence must be finite and lie in [0, 1]")
        if coherence_power is None or not np.isfinite(coherence_power) or coherence_power <= 0.0:
            raise ValueError("coherence_power must be finite and positive")
        availability = np.power(1.0 - coherent, float(coherence_power))

    activation = low_curvature_activation * availability
    completion = float(solar_completion) + (1.0 - float(solar_completion)) * activation
    enhancement = completion / float(solar_completion)
    return {
        "low_curvature_activation": low_curvature_activation,
        "vector_availability": availability,
        "completion_fraction": completion,
        "enhancement_relative_to_local_G": enhancement,
    }


def predict_completion_acceleration(
    gbar_m_s2,
    radius_kpc,
    parameters,
    *,
    coherence=None,
) -> dict[str, np.ndarray]:
    """Evaluate isotropic (3 parameters) or coherence-sensitive (4) law."""
    values = np.asarray(parameters, dtype=float)
    if values.shape not in {(3,), (4,)}:
        raise ValueError("parameters must contain 3 or 4 values")
    tidal = tidal_curvature_proxy(gbar_m_s2, radius_kpc)
    completion = bounded_completion(
        tidal,
        solar_completion=float(values[0]),
        tidal_transition_s2=float(10.0 ** values[1]),
        transition_power=float(values[2]),
        coherence=coherence if len(values) == 4 else None,
        coherence_power=float(values[3]) if len(values) == 4 else None,
    )
    completion["tidal_curvature_s2"] = tidal
    completion["predicted_acceleration_m_s2"] = (
        np.asarray(gbar_m_s2, dtype=float)
        * completion["enhancement_relative_to_local_G"]
    )
    return completion
