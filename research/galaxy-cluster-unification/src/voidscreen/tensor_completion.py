"""Bounded directional completion from the local Newtonian tidal tensor."""

from __future__ import annotations

import math

import numpy as np

from .data import KPC_M


G_SI = 6.67430e-11


TENSOR_MODELS = {
    "tensor_isotropic",
    "tensor_alignment",
    "tensor_competition",
    "tensor_dominance",
}


def spherical_tidal_eigenvalues(gbar_m_s2, radius_kpc, density_g_cm3) -> np.ndarray:
    """Return signed Hessian eigenvalues ``(radial, tangential, tangential)``.

    Spherical Poisson closure gives ``Phi_rr = 4*pi*G*rho - 2*g/r`` and
    ``Phi_tt = g/r``.  Density is the local baryonic density.
    """
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    density = np.asarray(density_g_cm3, dtype=float)
    gbar, radius, density = np.broadcast_arrays(gbar, radius, density)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("radius must be finite and positive")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and nonnegative")
    tangential = gbar / (radius * KPC_M)
    poisson_trace = 4.0 * math.pi * G_SI * density * 1000.0
    radial = poisson_trace - 2.0 * tangential
    return np.stack([radial, tangential, tangential], axis=-1)


def spherical_profile_tidal_eigenvalues(gbar_m_s2, radius_kpc) -> np.ndarray:
    """Reconstruct spherical Hessian eigenvalues from a resolved radial field."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    if gbar.ndim != 1 or radius.ndim != 1 or gbar.shape != radius.shape:
        raise ValueError("gbar and radius must be matching one-dimensional arrays")
    if len(radius) < 3 or np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("profile needs at least three finite positive accelerations")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("profile radii must be finite and positive")
    order = np.argsort(radius, kind="stable")
    sorted_radius_m = radius[order] * KPC_M
    if np.any(np.diff(sorted_radius_m) <= 0.0):
        raise ValueError("profile radii must be unique")
    sorted_gbar = gbar[order]
    edge_order = 2 if len(radius) >= 3 else 1
    radial = np.gradient(sorted_gbar, sorted_radius_m, edge_order=edge_order)
    tangential = sorted_gbar / sorted_radius_m
    sorted_values = np.stack([radial, tangential, tangential], axis=-1)
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def axisymmetric_tidal_eigenvalues(
    gbar_m_s2, radius_kpc, density_g_cm3
) -> np.ndarray:
    """Reconstruct ``(R, phi, z)`` Hessian eigenvalues in a disk midplane."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    radius = np.asarray(radius_kpc, dtype=float)
    density = np.asarray(density_g_cm3, dtype=float)
    if gbar.ndim != 1 or radius.ndim != 1 or density.ndim != 1:
        raise ValueError("axisymmetric inputs must be one-dimensional")
    if not (gbar.shape == radius.shape == density.shape):
        raise ValueError("axisymmetric inputs must have matching shapes")
    if len(radius) < 2 or np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("profile needs at least two finite positive accelerations")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("profile radii must be finite and positive")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and nonnegative")

    order = np.argsort(radius, kind="stable")
    sorted_radius_m = radius[order] * KPC_M
    if np.any(np.diff(sorted_radius_m) <= 0.0):
        raise ValueError("profile radii must be unique")
    sorted_gbar = gbar[order]
    radial = np.gradient(
        sorted_gbar,
        sorted_radius_m,
        edge_order=2 if len(radius) >= 3 else 1,
    )
    azimuthal = sorted_gbar / sorted_radius_m
    poisson_trace = 4.0 * math.pi * G_SI * density[order] * 1000.0
    vertical = poisson_trace - radial - azimuthal
    sorted_values = np.stack([radial, azimuthal, vertical], axis=-1)
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def tensor_completion(
    tidal_eigenvalues_s2,
    direction_components,
    model: str,
    parameters,
) -> dict[str, np.ndarray]:
    """Project a bounded completion tensor onto a physical direction.

    ``C_ij = C_solar I + (1-C_solar) A_T P_ij``.  The eigenvalues of ``P``
    are constrained to [0, 1], so every eigenvalue of ``C`` remains between
    the locally calibrated completion and the proposed universal maximum.
    """
    if model not in TENSOR_MODELS:
        raise ValueError(f"unknown tensor model {model}")
    values = np.asarray(parameters, dtype=float)
    expected = 3 if model == "tensor_isotropic" else 4
    if values.shape != (expected,):
        raise ValueError(f"{model} requires {expected} parameters")
    c_solar, log10_transition, transition_power = map(float, values[:3])
    if not 0.0 < c_solar <= 1.0:
        raise ValueError("solar completion must lie in (0, 1]")
    if not np.isfinite(log10_transition):
        raise ValueError("transition must be finite")
    if not np.isfinite(transition_power) or transition_power <= 0.0:
        raise ValueError("transition power must be finite and positive")

    tidal = np.asarray(tidal_eigenvalues_s2, dtype=float)
    direction = np.asarray(direction_components, dtype=float)
    if tidal.ndim < 1 or tidal.shape[-1] != 3 or np.any(~np.isfinite(tidal)):
        raise ValueError("tidal eigenvalues must be finite with final dimension three")
    direction = np.broadcast_to(direction, tidal.shape).astype(float, copy=False)
    if np.any(~np.isfinite(direction)):
        raise ValueError("direction must be finite")
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(direction_norm <= 0.0):
        raise ValueError("direction must be nonzero")
    direction = direction / direction_norm

    magnitude = np.abs(tidal)
    tidal_norm = np.linalg.norm(magnitude, axis=-1)
    if np.any(tidal_norm <= 0.0):
        raise ValueError("tidal tensor must have nonzero norm")
    transition = 10.0**log10_transition
    with np.errstate(over="ignore"):
        low_curvature = 1.0 / (1.0 + np.power(tidal_norm / transition, transition_power))

    if model == "tensor_isotropic":
        availability = np.ones_like(magnitude)
    else:
        q = float(values[3])
        if not np.isfinite(q) or q <= 0.0:
            raise ValueError("directional power must be finite and positive")
        if model == "tensor_alignment":
            availability = np.power(magnitude / tidal_norm[..., None], q)
        elif model == "tensor_competition":
            powered = np.power(magnitude, q)
            availability = powered / np.sum(powered, axis=-1, keepdims=True)
        else:
            maximum = np.max(magnitude, axis=-1, keepdims=True)
            availability = np.power(magnitude / maximum, q)

    tensor_eigenvalues = c_solar + (1.0 - c_solar) * low_curvature[..., None] * availability
    weights = np.square(direction)
    projected = np.sum(weights * tensor_eigenvalues, axis=-1)
    projected_availability = np.sum(weights * availability, axis=-1)
    return {
        "tidal_norm_s2": tidal_norm,
        "low_curvature_activation": low_curvature,
        "directional_availability": availability,
        "projected_availability": projected_availability,
        "completion_tensor_eigenvalues": tensor_eigenvalues,
        "projected_completion_fraction": projected,
        "enhancement_relative_to_local_G": projected / c_solar,
    }


def predict_tensor_acceleration(
    gbar_m_s2,
    tidal_eigenvalues_s2,
    model: str,
    parameters,
    *,
    direction_components=(1.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Return the acceleration obtained from ``n_i C_ij n_j``."""
    gbar = np.asarray(gbar_m_s2, dtype=float)
    if np.any(~np.isfinite(gbar)) or np.any(gbar <= 0.0):
        raise ValueError("gbar must be finite and positive")
    completion = tensor_completion(
        tidal_eigenvalues_s2,
        direction_components,
        model,
        parameters,
    )
    completion["predicted_acceleration_m_s2"] = (
        gbar * completion["enhancement_relative_to_local_G"]
    )
    return completion
