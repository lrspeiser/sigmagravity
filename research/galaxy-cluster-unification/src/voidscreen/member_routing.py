"""Mass-budget-preserving weights for directional cluster-member routing."""

from __future__ import annotations

import numpy as np


def normalized_member_weights(
    base_mass,
    *,
    mass_power: float = 1.0,
    radial_dressing=None,
) -> np.ndarray:
    """Reweight members while preserving their summed effective baryonic mass.

    ``mass_power`` changes which members dominate, while ``radial_dressing``
    permits a fixed scalar parent response to modulate where routing is most
    effective.  Neither operation changes the total radial mass budget.
    """
    mass = np.asarray(base_mass, dtype=float)
    if mass.ndim != 1 or mass.size == 0:
        raise ValueError("base_mass must be a nonempty one-dimensional vector")
    if np.any(~np.isfinite(mass)) or np.any(mass < 0.0) or not np.any(mass > 0.0):
        raise ValueError("base_mass must be finite, nonnegative, and not all zero")
    if not np.isfinite(mass_power) or mass_power <= 0.0:
        raise ValueError("mass_power must be finite and positive")
    if radial_dressing is None:
        dressing = np.ones_like(mass)
    else:
        dressing = np.asarray(radial_dressing, dtype=float)
    if dressing.shape != mass.shape:
        raise ValueError("radial_dressing must match base_mass")
    if np.any(~np.isfinite(dressing)) or np.any(dressing <= 0.0):
        raise ValueError("radial_dressing must be finite and strictly positive")

    pivot = float(np.exp(np.mean(np.log(mass[mass > 0.0]))))
    raw = np.power(mass / pivot, float(mass_power)) * dressing
    return raw * (float(np.sum(mass)) / float(np.sum(raw)))
