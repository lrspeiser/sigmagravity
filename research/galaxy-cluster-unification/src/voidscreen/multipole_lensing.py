"""Matched curl-free angular multipoles for lens-model controls."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from voidscreen.stellar_morphology_lensing import StellarMorphologyDeflectionField


def _sample(axis: np.ndarray, values: np.ndarray, x, y) -> np.ndarray:
    spacing = float(axis[1] - axis[0])
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    coordinates = np.vstack(
        [((y - axis[0]) / spacing).ravel(), ((x - axis[0]) / spacing).ravel()]
    )
    return map_coordinates(
        values,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(x.shape)


def build_matched_multipole_deflection_field(
    axis_arcsec,
    *,
    order: int,
    phase_rad: float,
    radial_scale_arcsec: float,
    taper_inner_arcsec: float,
    support_radius_arcsec: float,
    target_deflection_rms_arcsec: float,
) -> StellarMorphologyDeflectionField:
    """Build a one-amplitude, zero-monopole multipole from a scalar potential."""

    axis = np.asarray(axis_arcsec, dtype=float)
    if axis.ndim != 1 or len(axis) < 64 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("axis must be strictly increasing with at least 64 cells")
    spacing = float(axis[1] - axis[0])
    if not np.allclose(np.diff(axis), spacing):
        raise ValueError("axis spacing must be uniform")
    if int(order) != order or int(order) < 2:
        raise ValueError("multipole order must be an integer of at least two")
    if not 0.0 < radial_scale_arcsec < support_radius_arcsec:
        raise ValueError("radial scale must lie inside support")
    if not 0.0 < taper_inner_arcsec < support_radius_arcsec < max(abs(axis[0]), abs(axis[-1])):
        raise ValueError("taper and support must fit inside the grid")
    if not np.isfinite(target_deflection_rms_arcsec) or target_deflection_rms_arcsec <= 0.0:
        raise ValueError("target deflection RMS must be positive")

    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)
    scaled = radius / float(radial_scale_arcsec)
    envelope = np.power(scaled, int(order)) * np.exp(-0.5 * scaled * scaled)
    taper = np.ones_like(radius)
    transition = (radius > taper_inner_arcsec) & (radius < support_radius_arcsec)
    taper[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (radius[transition] - taper_inner_arcsec)
            / (support_radius_arcsec - taper_inner_arcsec)
        )
    )
    taper[radius >= support_radius_arcsec] = 0.0
    potential = envelope * taper * np.cos(int(order) * (angle - float(phase_rad)))
    alpha_y, alpha_x = np.gradient(potential, spacing, edge_order=2)
    raw_rms = float(np.sqrt(np.mean(alpha_x * alpha_x + alpha_y * alpha_y)))
    if not np.isfinite(raw_rms) or raw_rms <= np.finfo(float).tiny:
        raise RuntimeError("multipole normalization is degenerate")
    normalization = float(target_deflection_rms_arcsec) / raw_rms
    potential *= normalization
    alpha_x *= normalization
    alpha_y *= normalization

    dax_dx = np.gradient(alpha_x, spacing, axis=1, edge_order=2)
    day_dy = np.gradient(alpha_y, spacing, axis=0, edge_order=2)
    day_dx = np.gradient(alpha_y, spacing, axis=1, edge_order=2)
    dax_dy = np.gradient(alpha_x, spacing, axis=0, edge_order=2)
    divergence = dax_dx + day_dy
    curl = day_dx - dax_dy
    convergence = 0.5 * divergence

    circular_radius = np.linspace(spacing, support_radius_arcsec + 20.0, 128)
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    polar_x = circular_radius[:, None] * np.cos(phi)[None, :]
    polar_y = circular_radius[:, None] * np.sin(phi)[None, :]
    polar_ax = _sample(axis, alpha_x, polar_x, polar_y)
    polar_ay = _sample(axis, alpha_y, polar_x, polar_y)
    circular_mean = np.mean(
        polar_ax * np.cos(phi)[None, :] + polar_ay * np.sin(phi)[None, :],
        axis=1,
    )

    source_integral_fraction = float(
        abs(np.sum(divergence))
        / max(np.sum(np.abs(divergence)), np.finfo(float).tiny)
    )
    normalized_curl = float(
        np.sqrt(np.mean(curl * curl))
        / max(np.sqrt(np.mean(divergence * divergence)), np.finfo(float).tiny)
    )
    edge = np.r_[
        alpha_x[0],
        alpha_x[-1],
        alpha_x[:, 0],
        alpha_x[:, -1],
        alpha_y[0],
        alpha_y[-1],
        alpha_y[:, 0],
        alpha_y[:, -1],
    ]
    audit = {
        "operator": "matched_scalar_potential_multipole",
        "order": int(order),
        "phase_rad": float(phase_rad),
        "radial_scale_arcsec": float(radial_scale_arcsec),
        "taper_inner_arcsec": float(taper_inner_arcsec),
        "support_radius_arcsec": float(support_radius_arcsec),
        "normalization": normalization,
        "unit_deflection_RMS_arcsec": float(
            np.sqrt(np.mean(alpha_x * alpha_x + alpha_y * alpha_y))
        ),
        "unit_deflection_maximum_arcsec": float(np.max(np.hypot(alpha_x, alpha_y))),
        "maximum_circular_mean_deflection_arcsec": float(np.max(np.abs(circular_mean))),
        "source_integral_fraction": source_integral_fraction,
        "normalized_curl_RMS": normalized_curl,
        "maximum_edge_correction_arcsec": float(np.max(np.abs(edge))),
    }
    return StellarMorphologyDeflectionField(
        axis,
        alpha_x,
        alpha_y,
        circular_radius,
        circular_mean,
        np.ones_like(potential),
        convergence,
        audit,
    )
