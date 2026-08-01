"""One-dimensional spherical controls for conservative field redistribution."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erfc, expit


def low_acceleration_activation(acceleration_m_s2: float, *, a0_m_s2: float, power: float) -> float:
    """Return a bounded source-level low-acceleration activation."""
    acceleration = float(acceleration_m_s2)
    a0 = float(a0_m_s2)
    exponent = float(power)
    if acceleration < 0.0 or a0 <= 0.0 or exponent <= 0.0:
        raise ValueError("activation inputs must be physical")
    return 1.0 / (1.0 + (acceleration / a0) ** exponent)


def radial_shape_activation(concentration: float, *, midpoint: float, width: float) -> float:
    """Bounded logistic gate for the dimensionless radial ratio R50/R80."""
    value = float(concentration)
    center = float(midpoint)
    scale = float(width)
    if not (0.0 <= value <= 1.0) or not (0.0 < center < 1.0) or scale <= 0.0:
        raise ValueError("radial-shape gate inputs must be physical")
    return float(expit((value - center) / scale))


def redistributed_cumulative_mass(
    radius,
    cumulative_mass,
    *,
    r80: float,
    position_scale: float,
    width_over_r80: float,
    bins: int = 1024,
) -> tuple[np.ndarray, float]:
    """Move radial shells, smooth them, and conserve their total mass."""
    r = np.asarray(radius, dtype=float)
    mass = np.asarray(cumulative_mass, dtype=float)
    if (
        r.ndim != 1
        or mass.shape != r.shape
        or len(r) < 2
        or np.any(~np.isfinite(r))
        or np.any(~np.isfinite(mass))
        or np.any(r <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or np.any(np.diff(mass) < -1e-12)
        or mass[-1] <= 0.0
        or float(r80) <= 0.0
        or float(position_scale) < 0.0
        or float(width_over_r80) < 0.0
        or int(bins) < 64
    ):
        raise ValueError("invalid conservative redistribution profile")
    shells = np.diff(np.r_[0.0, mass])
    positions = float(position_scale) * r
    width = float(width_over_r80) * float(r80)
    maximum = max(float(r[-1]), float(np.max(positions)) + 6.0 * width, float(r80))
    edges = np.linspace(0.0, maximum, int(bins) + 1)
    histogram, _ = np.histogram(positions, bins=edges, weights=shells)
    spacing = float(edges[1] - edges[0])
    redistributed = histogram.astype(float)
    if width > 0.0:
        redistributed = gaussian_filter1d(redistributed, width / spacing, mode="constant")
    raw_total = float(np.sum(redistributed))
    redistributed *= float(mass[-1]) / raw_total
    cumulative = np.cumsum(redistributed)
    sampled = np.interp(r, np.r_[0.0, edges[1:]], np.r_[0.0, cumulative])
    error = abs(float(np.sum(redistributed)) / float(mass[-1]) - 1.0)
    return sampled, error


def gaussian_tail_upper_bound(*, evaluation_radius: float, source_radius: float, sigma: float) -> float:
    """One-sided 1-D upper bound on redistributed mass beyond an exterior radius."""
    evaluation = float(evaluation_radius)
    source = float(source_radius)
    width = float(sigma)
    if evaluation <= source or source < 0.0 or width <= 0.0:
        raise ValueError("tail bound requires an exterior evaluation radius and positive width")
    z = (evaluation - source) / width
    return float(0.5 * erfc(z / math.sqrt(2.0)))
