"""Weak-field tidal-tensor response candidate for Sigma Gravity."""

from __future__ import annotations

import numpy as np


def solar_gate(acceleration_m_s2, *, a0_m_s2: float) -> np.ndarray:
    """Low-acceleration gate that vanishes quadratically at high acceleration."""
    acceleration = np.asarray(acceleration_m_s2, dtype=float)
    if np.any(acceleration < 0.0) or a0_m_s2 <= 0.0:
        raise ValueError("accelerations must be non-negative and a0 positive")
    return np.square(a0_m_s2) / (
        np.square(a0_m_s2) + np.square(acceleration)
    )


def normalized_squared_tidal(tidal_tensor) -> np.ndarray:
    """Return E^2/tr(E^2), or zero where the tidal tensor vanishes."""
    tidal = np.asarray(tidal_tensor, dtype=float)
    if tidal.shape[-2:] != (3, 3):
        raise ValueError("tidal tensor must end in a 3x3 matrix")
    if np.any(~np.isfinite(tidal)):
        raise ValueError("tidal tensor must be finite")
    symmetric = 0.5 * (tidal + np.swapaxes(tidal, -1, -2))
    squared = symmetric @ symmetric
    trace = np.trace(squared, axis1=-2, axis2=-1)
    result = np.zeros_like(squared)
    np.divide(
        squared,
        trace[..., np.newaxis, np.newaxis],
        out=result,
        where=trace[..., np.newaxis, np.newaxis] > 0.0,
    )
    return result


def response_tensor(
    tidal_tensor,
    acceleration_m_s2,
    *,
    kappa: float,
    a0_m_s2: float,
    mapping: str = "linear",
) -> np.ndarray:
    """Return a positive response tensor for the anisotropic Poisson equation."""
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    if mapping == "linear" and kappa >= 1.0:
        raise ValueError("linear mapping requires 0 <= kappa < 1")
    direction = normalized_squared_tidal(tidal_tensor)
    gate = solar_gate(acceleration_m_s2, a0_m_s2=a0_m_s2)
    identity = np.eye(3)
    scaled = kappa * gate[..., np.newaxis, np.newaxis] * direction
    if mapping == "linear":
        return identity - scaled
    eigenvalues, eigenvectors = np.linalg.eigh(direction)
    scalar_gate = kappa * gate[..., np.newaxis]
    if mapping == "exponential":
        mapped = np.exp(-scalar_gate * eigenvalues)
    elif mapping == "reciprocal":
        mapped = 1.0 / (1.0 + scalar_gate * eigenvalues)
    else:
        raise ValueError(f"unknown response mapping: {mapping}")
    return (eigenvectors * mapped[..., np.newaxis, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
