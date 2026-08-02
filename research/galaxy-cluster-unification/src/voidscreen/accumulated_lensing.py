"""Projected lens operator for finite path-accumulated component transport."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage

from voidscreen.geometric_transport import (
    component_cancellation,
    high_acceleration_screen,
    streamline_incoherence,
    thin_sheet_newtonian_field,
)
from voidscreen.stellar_morphology_lensing import StellarMorphologyDeflectionField


def _normalize_proxy(proxy: np.ndarray, fraction: float, total_mass_msun: float, cell_kpc: float):
    values = np.maximum(np.asarray(proxy, dtype=np.float64), 0.0)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or float(np.sum(values)) <= 0.0:
        raise ValueError("component proxy must be a positive square map")
    return values * float(fraction) * float(total_mass_msun) / (
        float(np.sum(values)) * float(cell_kpc) ** 2
    )


def _spectral_lens_solve(source: np.ndarray, spacing_arcsec: float):
    cells = source.shape[0]
    frequency = 2.0 * np.pi * np.fft.fftfreq(cells, d=float(spacing_arcsec))
    kx, ky = np.meshgrid(frequency, frequency, indexing="xy")
    wave2 = kx * kx + ky * ky
    transformed = np.fft.fft2(source - float(np.mean(source)))
    potential_hat = np.zeros_like(transformed)
    active = wave2 > 0.0
    potential_hat[active] = -transformed[active] / wave2[active]
    alpha_x_hat = 1j * kx * potential_hat
    alpha_y_hat = 1j * ky * potential_hat
    alpha_x = np.real(np.fft.ifft2(alpha_x_hat))
    alpha_y = np.real(np.fft.ifft2(alpha_y_hat))
    curl_hat = 1j * (kx * alpha_y_hat - ky * alpha_x_hat)
    divergence_hat = 1j * (kx * alpha_x_hat + ky * alpha_y_hat)
    curl = np.real(np.fft.ifft2(curl_hat))
    divergence = np.real(np.fft.ifft2(divergence_hat))
    normalized_curl = float(
        np.sqrt(np.mean(curl**2))
        / max(np.sqrt(np.mean(divergence**2)), np.finfo(float).tiny)
    )
    return alpha_x, alpha_y, normalized_curl


def build_accumulated_transport_deflection_field(
    axis_arcsec,
    stellar_proxy,
    gas_proxy,
    *,
    angular_scale_kpc_per_arcsec: float,
    carrier_alpha_arcsec: Callable[[np.ndarray], np.ndarray],
    radial_gbar_m_s2: Callable[[np.ndarray], np.ndarray],
    stellar_mass_fraction: float = 0.1,
    gas_mass_fraction: float = 0.9,
    proxy_total_mass_msun: float = 1.0e14,
    coherence_length_kpc: float = 10.0,
    accumulation_power: float = 1.0,
    a0_m_s2: float = 1.2e-10,
    common_smoothing_kpc: float = 0.0,
    closure: str = "path_tensor",
    taper_inner_arcsec: float = 180.0,
    support_radius_arcsec: float = 220.0,
) -> StellarMorphologyDeflectionField:
    """Apply the accumulated tensor to a fixed radial carrier lens potential.

    The common proxy mass cancels from component angles and the tidal length;
    it is retained only so the Newtonian map solver has physical units.  The
    absolute deflection scale comes entirely from ``carrier_alpha_arcsec``.
    """

    axis = np.asarray(axis_arcsec, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 64 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("axis must be a strictly increasing vector")
    spacing = float(np.median(np.diff(axis)))
    if not np.allclose(np.diff(axis), spacing):
        raise ValueError("axis must be uniformly spaced")
    scale = float(angular_scale_kpc_per_arcsec)
    if scale <= 0.0 or coherence_length_kpc <= 0.0 or accumulation_power <= 0.0:
        raise ValueError("physical scales must be positive")
    if not np.isclose(stellar_mass_fraction + gas_mass_fraction, 1.0):
        raise ValueError("component fractions must sum to one")
    cell_kpc = spacing * scale
    star_proxy = np.asarray(stellar_proxy, dtype=np.float64)
    gas_proxy_values = np.asarray(gas_proxy, dtype=np.float64)
    if common_smoothing_kpc < 0.0:
        raise ValueError("common_smoothing_kpc must be non-negative")
    smoothing_pixels = float(common_smoothing_kpc) / cell_kpc
    if smoothing_pixels > 0.0:
        star_proxy = ndimage.gaussian_filter(star_proxy, smoothing_pixels, mode="constant")
        gas_proxy_values = ndimage.gaussian_filter(
            gas_proxy_values, smoothing_pixels, mode="constant"
        )
    stars = _normalize_proxy(
        star_proxy, stellar_mass_fraction, proxy_total_mass_msun, cell_kpc
    )
    gas = _normalize_proxy(
        gas_proxy_values, gas_mass_fraction, proxy_total_mass_msun, cell_kpc
    )
    star_field = thin_sheet_newtonian_field(stars, cell_kpc)
    gas_field = thin_sheet_newtonian_field(gas, cell_kpc)
    total_field = thin_sheet_newtonian_field(stars + gas, cell_kpc)
    path = streamline_incoherence(total_field, cell_kpc)
    cancellation = component_cancellation(star_field, gas_field)
    survival = 1.0 - np.exp(
        -np.power(path.trace_length_kpc / float(coherence_length_kpc), accumulation_power)
    )
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius_arcsec = np.hypot(xx, yy)
    radius_kpc = np.maximum(radius_arcsec * scale, 1.0e-6)
    gbar = np.asarray(radial_gbar_m_s2(radius_kpc), dtype=np.float64)
    if gbar.shape != radius_kpc.shape or np.any(gbar <= 0.0) or not np.all(np.isfinite(gbar)):
        raise ValueError("radial_gbar_m_s2 returned invalid values")
    activation = cancellation * survival * high_acceleration_screen(gbar, a0_m_s2)
    taper = np.ones_like(radius_arcsec)
    transition = (radius_arcsec > taper_inner_arcsec) & (radius_arcsec < support_radius_arcsec)
    taper[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (radius_arcsec[transition] - taper_inner_arcsec)
            / (support_radius_arcsec - taper_inner_arcsec)
        )
    )
    taper[radius_arcsec >= support_radius_arcsec] = 0.0
    activation *= taper
    safe_radius = np.maximum(radius_arcsec, spacing / 20.0)
    carrier = np.asarray(carrier_alpha_arcsec(safe_radius), dtype=np.float64)
    radial_x = xx / safe_radius
    radial_y = yy / safe_radius
    grad_x = carrier * radial_x
    grad_y = carrier * radial_y
    star_norm = np.maximum(star_field.magnitude_m_s2, np.finfo(float).tiny)
    gas_norm = np.maximum(gas_field.magnitude_m_s2, np.finfo(float).tiny)
    difference_x = gas_field.acceleration_x_m_s2 / gas_norm - star_field.acceleration_x_m_s2 / star_norm
    difference_y = gas_field.acceleration_y_m_s2 / gas_norm - star_field.acceleration_y_m_s2 / star_norm
    difference_norm = np.hypot(difference_x, difference_y)
    valid_difference = difference_norm > 1e-12
    difference_x = np.where(
        valid_difference, difference_x / np.maximum(difference_norm, 1e-12), path.mean_direction_x
    )
    difference_y = np.where(
        valid_difference, difference_y / np.maximum(difference_norm, 1e-12), path.mean_direction_y
    )
    perpendicular_x = -difference_y
    perpendicular_y = difference_x

    def rank_one(direction_x, direction_y):
        projection = direction_x * grad_x + direction_y * grad_y
        return activation * direction_x * projection, activation * direction_y * projection

    closure_id = str(closure)
    if closure_id == "path_tensor":
        flux_x, flux_y = rank_one(path.mean_direction_x, path.mean_direction_y)
    elif closure_id == "difference_tensor":
        flux_x, flux_y = rank_one(difference_x, difference_y)
    elif closure_id == "perpendicular_difference_tensor":
        flux_x, flux_y = rank_one(perpendicular_x, perpendicular_y)
    elif closure_id == "isotropic_control":
        flux_x, flux_y = activation * grad_x, activation * grad_y
    elif closure_id == "gas_minus_star_flux":
        flux_x, flux_y = activation * carrier * difference_x, activation * carrier * difference_y
    elif closure_id == "star_minus_gas_flux":
        flux_x, flux_y = -activation * carrier * difference_x, -activation * carrier * difference_y
    elif closure_id == "perpendicular_cw_flux":
        flux_x, flux_y = activation * carrier * perpendicular_x, activation * carrier * perpendicular_y
    elif closure_id == "perpendicular_ccw_flux":
        flux_x, flux_y = -activation * carrier * perpendicular_x, -activation * carrier * perpendicular_y
    else:
        raise ValueError(f"unknown accumulated lens closure: {closure_id}")
    source = np.gradient(flux_x, spacing, axis=1, edge_order=2) + np.gradient(
        flux_y, spacing, axis=0, edge_order=2
    )
    alpha_x, alpha_y, normalized_curl = _spectral_lens_solve(source, spacing)
    circular_radius = np.linspace(spacing, support_radius_arcsec + 20.0, 64)
    circular_zero = np.zeros_like(circular_radius)
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
    active = radius_arcsec <= support_radius_arcsec
    audit = {
        "operator": "finite_path_accumulated_component_tensor",
        "closure": closure_id,
        "coherence_length_kpc": float(coherence_length_kpc),
        "accumulation_power": float(accumulation_power),
        "stellar_mass_fraction": float(stellar_mass_fraction),
        "gas_mass_fraction": float(gas_mass_fraction),
        "proxy_total_mass_msun": float(proxy_total_mass_msun),
        "common_smoothing_kpc": float(common_smoothing_kpc),
        "proxy_mass_changes_deflection_normalization": False,
        "activation_weighted_mean": float(np.mean(activation[active])),
        "activation_maximum": float(np.max(activation[active])),
        "cancellation_mean": float(np.mean(cancellation[active])),
        "survival_mean": float(np.mean(survival[active])),
        "trace_length_median_kpc": float(np.median(path.trace_length_kpc[active])),
        "source_integral_fraction": float(
            abs(np.sum(source)) / max(np.sum(np.abs(source)), np.finfo(float).tiny)
        ),
        "normalized_curl_RMS": normalized_curl,
        "unit_deflection_RMS_arcsec": float(np.sqrt(np.mean(alpha_x**2 + alpha_y**2))),
        "unit_deflection_maximum_arcsec": float(np.max(np.hypot(alpha_x, alpha_y))),
        "maximum_edge_correction_arcsec": float(np.max(np.abs(edge))),
        "carrier_deflection_median_arcsec": float(np.median(carrier[active])),
    }
    return StellarMorphologyDeflectionField(
        axis,
        alpha_x,
        alpha_y,
        circular_radius,
        circular_zero,
        activation,
        0.5 * source,
        audit,
    )
