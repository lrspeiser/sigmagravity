"""Conservative projected return kernels for a hidden-path gravity test.

The model is deliberately phenomenological.  A fraction of baryon-sourced
field strength leaves the observed plane, follows an unobserved curved path,
and returns after a projected displacement.  Convolution by a normalized ring
kernel conserves the integrated source weight.  It therefore creates an
*apparent* projected source pattern without adding material mass.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19


def transition_radius_arcsec(
    total_baryonic_mass_msun: float,
    *,
    a0_m_s2: float,
    scale_kpc_per_arcsec: float,
) -> float:
    """Return sqrt(G M/a0), expressed as a projected angular distance."""
    mass = float(total_baryonic_mass_msun)
    a0 = float(a0_m_s2)
    scale = float(scale_kpc_per_arcsec)
    if mass <= 0.0 or a0 <= 0.0 or scale <= 0.0:
        raise ValueError("mass, a0, and angular scale must be positive")
    radius_kpc = math.sqrt(G_SI * mass * M_SUN_KG / a0) / KPC_M
    return radius_kpc / scale


def normalized_ring_kernel(
    axis_arcsec,
    *,
    return_radius_arcsec: float,
    width_arcsec: float,
) -> np.ndarray:
    """Return a unit-normalized isotropic projected return annulus."""
    axis = np.asarray(axis_arcsec, dtype=float)
    radius = float(return_radius_arcsec)
    width = float(width_arcsec)
    if axis.ndim != 1 or len(axis) < 5 or radius < 0.0 or width <= 0.0:
        raise ValueError("invalid ring-kernel geometry")
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    values = np.exp(-0.5 * np.square((np.hypot(xx, yy) - radius) / width))
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("ring kernel has zero support")
    return values / total


def normalized_directional_ring_kernel(
    axis_arcsec,
    *,
    return_radius_arcsec: float,
    width_arcsec: float,
    major_axis_deg: float,
    directional_concentration: float,
) -> np.ndarray:
    """Return a normalized ring modulated by a baryon-defined spin-2 axis."""
    axis = np.asarray(axis_arcsec, dtype=float)
    radius = float(return_radius_arcsec)
    width = float(width_arcsec)
    concentration = float(directional_concentration)
    if axis.ndim != 1 or len(axis) < 5 or radius < 0.0 or width <= 0.0:
        raise ValueError("invalid directional ring geometry")
    if not math.isfinite(concentration):
        raise ValueError("directional concentration must be finite")
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    angle = np.arctan2(yy, xx) - math.radians(float(major_axis_deg))
    radial = np.exp(-0.5 * np.square((np.hypot(xx, yy) - radius) / width))
    angular = np.exp(concentration * np.cos(2.0 * angle))
    values = radial * angular
    return values / np.sum(values)


def routed_arrival_map(
    baryon_surface,
    ring_kernel,
    *,
    routed_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized total prediction and its normalized return layer."""
    baryon = np.asarray(baryon_surface, dtype=float)
    kernel = np.asarray(ring_kernel, dtype=float)
    fraction = float(routed_fraction)
    if baryon.ndim != 2 or kernel.shape != baryon.shape:
        raise ValueError("baryon map and kernel must be matching square images")
    if np.any(~np.isfinite(baryon)) or np.any(baryon < 0.0) or np.sum(baryon) <= 0.0:
        raise ValueError("baryon map must be finite, nonnegative, and nonzero")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("routed fraction must lie in [0,1]")
    local = baryon / np.sum(baryon)
    arrival = fftconvolve(local, kernel, mode="same")
    arrival = np.maximum(arrival, 0.0)
    arrival /= np.sum(arrival)
    prediction = (1.0 - fraction) * local + fraction * arrival
    prediction /= np.sum(prediction)
    return prediction, arrival


def jensen_shannon_divergence(first, second, mask=None) -> float:
    """Jensen-Shannon divergence between two nonnegative sampled maps."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("maps must have matching shapes")
    use = np.ones(a.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if use.shape != a.shape:
        raise ValueError("mask must match map shape")
    p = np.maximum(a[use], 0.0)
    q = np.maximum(b[use], 0.0)
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        return float("nan")
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    return float(
        0.5 * np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p]))
        + 0.5 * np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q]))
    )


def source_origin_probabilities(
    source_x_arcsec,
    source_y_arcsec,
    source_weights,
    *,
    destination_x_arcsec: float,
    destination_y_arcsec: float,
    return_radius_arcsec: float,
    width_arcsec: float,
    major_axis_deg: float | None = None,
    directional_concentration: float = 0.0,
) -> np.ndarray:
    """Bayesian attribution of one return location to discrete baryon sources."""
    x = np.asarray(source_x_arcsec, dtype=float)
    y = np.asarray(source_y_arcsec, dtype=float)
    weight = np.asarray(source_weights, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or weight.shape != x.shape:
        raise ValueError("source arrays must be matching vectors")
    if np.any(weight < 0.0) or np.sum(weight) <= 0.0 or float(width_arcsec) <= 0.0:
        raise ValueError("source weights and return width must be positive")
    distance = np.hypot(x - float(destination_x_arcsec), y - float(destination_y_arcsec))
    likelihood = np.exp(
        -0.5 * np.square((distance - float(return_radius_arcsec)) / float(width_arcsec))
    )
    if major_axis_deg is not None and directional_concentration != 0.0:
        angle = np.arctan2(
            float(destination_y_arcsec) - y,
            float(destination_x_arcsec) - x,
        ) - math.radians(float(major_axis_deg))
        likelihood *= np.exp(float(directional_concentration) * np.cos(2.0 * angle))
    posterior = weight * likelihood
    total = float(np.sum(posterior))
    if total <= np.finfo(float).tiny:
        return np.full_like(weight, 1.0 / len(weight))
    return posterior / total


def semicircle_arc_geometry(projected_distance_arcsec: float, scale_kpc_per_arcsec: float) -> dict[str, float]:
    """Geometry of the simplest hidden semicircular route between two points."""
    distance = float(projected_distance_arcsec)
    scale = float(scale_kpc_per_arcsec)
    if distance < 0.0 or scale <= 0.0:
        raise ValueError("arc distance must be nonnegative and scale positive")
    radius_kpc = 0.5 * distance * scale
    return {
        "projected_distance_kpc": distance * scale,
        "maximum_hidden_height_kpc": radius_kpc,
        "semicircle_path_length_kpc": math.pi * radius_kpc,
    }
