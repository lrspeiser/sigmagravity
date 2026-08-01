"""Monopole-conserving lens perturbations from continuous stellar light."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import map_coordinates


def _sample_grid(axis: np.ndarray, field: np.ndarray, x, y) -> np.ndarray:
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    spacing = float(axis[1] - axis[0])
    coordinates = np.vstack(
        [
            ((y - axis[0]) / spacing).ravel(),
            ((x - axis[0]) / spacing).ravel(),
        ]
    )
    return map_coordinates(
        field,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(x.shape)


def radial_convergence_from_deflection(
    radius_arcsec,
    alpha_arcsec,
) -> np.ndarray:
    """Recover axisymmetric convergence from ``alpha(R)`` in angular units."""

    radius = np.asarray(radius_arcsec, dtype=float)
    alpha = np.asarray(alpha_arcsec, dtype=float)
    if radius.ndim != 1 or alpha.shape != radius.shape or len(radius) < 8:
        raise ValueError("radius and alpha must be matching one-dimensional vectors")
    if np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be positive and strictly increasing")
    if np.any(~np.isfinite(alpha)):
        raise ValueError("deflection must be finite")
    derivative = np.gradient(alpha, radius, edge_order=2)
    return 0.5 * (alpha / radius + derivative)


def normalized_light_weights(
    positive_light,
    carrier_convergence,
    radius_arcsec,
    *,
    contrast_cap: float,
    contrast_mode: str = "hard",
    annulus_width_arcsec: float,
    support_radius_arcsec: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return bounded positive weights with exact carrier-weighted annular mean one."""

    light, carrier, radius = np.broadcast_arrays(
        np.asarray(positive_light, dtype=float),
        np.asarray(carrier_convergence, dtype=float),
        np.asarray(radius_arcsec, dtype=float),
    )
    if np.any(~np.isfinite(light)) or np.any(light < 0.0):
        raise ValueError("positive_light must be finite and nonnegative")
    if np.any(~np.isfinite(carrier)):
        raise ValueError("carrier convergence must be finite")
    if contrast_cap < 1.0 or annulus_width_arcsec <= 0.0 or support_radius_arcsec <= 0.0:
        raise ValueError("invalid light-weight settings")
    mode = str(contrast_mode)
    if mode not in {"hard", "tanh", "exponential", "rational"}:
        raise ValueError("contrast_mode must be hard, tanh, exponential, or rational")
    weights = np.ones_like(light)
    bins = np.floor(radius / float(annulus_width_arcsec)).astype(int)
    maximum_bin = int(np.floor(support_radius_arcsec / annulus_width_arcsec))
    maximum_error = 0.0
    minimum_weight = 1.0
    maximum_weight = 1.0
    empty_bins = 0
    for index in range(maximum_bin + 1):
        selected = (bins == index) & (radius <= support_radius_arcsec)
        if not np.any(selected):
            continue
        mean_light = float(np.mean(light[selected]))
        if mean_light <= np.finfo(float).tiny:
            empty_bins += 1
            continue
        ratio = np.maximum(light[selected] / mean_light, 0.0)
        cap = float(contrast_cap)
        if mode == "hard":
            raw = np.minimum(ratio, cap)
        elif mode == "tanh":
            raw = cap * np.tanh(ratio / cap)
        elif mode == "exponential":
            raw = cap * (-np.expm1(-ratio / cap))
        else:
            raw = cap * ratio / (cap + ratio)
        radial_carrier = carrier[selected]
        carrier_sum = float(np.sum(radial_carrier))
        if abs(carrier_sum) <= np.finfo(float).tiny:
            normalization = float(np.mean(raw))
        else:
            normalization = float(np.sum(radial_carrier * raw) / carrier_sum)
        if not np.isfinite(normalization) or normalization <= np.finfo(float).tiny:
            empty_bins += 1
            continue
        local = raw / normalization
        weights[selected] = local
        residual = float(np.mean(radial_carrier * (local - 1.0)))
        scale = max(float(np.mean(np.abs(radial_carrier))), np.finfo(float).tiny)
        maximum_error = max(maximum_error, abs(residual) / scale)
        minimum_weight = min(minimum_weight, float(np.min(local)))
        maximum_weight = max(maximum_weight, float(np.max(local)))
    weights[radius > support_radius_arcsec] = 1.0
    return weights, {
        "contrast_mode": mode,
        "maximum_carrier_weighted_annular_mean_error": maximum_error,
        "minimum_light_weight": minimum_weight,
        "maximum_light_weight": maximum_weight,
        "empty_annuli": float(empty_bins),
    }


@dataclass(frozen=True)
class StellarMorphologyDeflectionField:
    """Unit-strength angular correction with an explicitly removed monopole."""

    axis_arcsec: np.ndarray
    raw_alpha_x_arcsec: np.ndarray
    raw_alpha_y_arcsec: np.ndarray
    circular_radius_arcsec: np.ndarray
    circular_radial_alpha_arcsec: np.ndarray
    light_weight: np.ndarray
    delta_convergence: np.ndarray
    audit: dict[str, float]

    def alpha_arcsec(
        self,
        x_arcsec,
        y_arcsec,
        *,
        distance_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(distance_ratio) or distance_ratio <= 0.0:
            raise ValueError("distance_ratio must be finite and positive")
        x, y = np.broadcast_arrays(
            np.asarray(x_arcsec, dtype=float), np.asarray(y_arcsec, dtype=float)
        )
        raw_x = _sample_grid(self.axis_arcsec, self.raw_alpha_x_arcsec, x, y)
        raw_y = _sample_grid(self.axis_arcsec, self.raw_alpha_y_arcsec, x, y)
        radius = np.hypot(x, y)
        safe = np.maximum(radius, 1.0e-12)
        circular = np.interp(
            radius,
            self.circular_radius_arcsec,
            self.circular_radial_alpha_arcsec,
            left=0.0,
            right=0.0,
        )
        scale = float(distance_ratio)
        return (
            scale * (raw_x - circular * x / safe),
            scale * (raw_y - circular * y / safe),
        )


@dataclass(frozen=True)
class BlendedMorphologyDeflectionField:
    """Linear blend of two already-conservative morphology fields."""

    left: StellarMorphologyDeflectionField
    right: StellarMorphologyDeflectionField
    right_fraction: float
    audit: dict[str, float]

    def __post_init__(self):
        if not np.isfinite(self.right_fraction) or not 0.0 <= self.right_fraction <= 1.0:
            raise ValueError("right_fraction must lie in [0, 1]")
        if not np.array_equal(self.left.axis_arcsec, self.right.axis_arcsec):
            raise ValueError("blended morphology fields must use the same grid")

    def alpha_arcsec(self, x_arcsec, y_arcsec, *, distance_ratio: float):
        left_x, left_y = self.left.alpha_arcsec(
            x_arcsec, y_arcsec, distance_ratio=distance_ratio
        )
        right_x, right_y = self.right.alpha_arcsec(
            x_arcsec, y_arcsec, distance_ratio=distance_ratio
        )
        fraction = float(self.right_fraction)
        return (
            (1.0 - fraction) * left_x + fraction * right_x,
            (1.0 - fraction) * left_y + fraction * right_y,
        )


def blend_morphology_deflection_fields(left, right, right_fraction: float):
    """Blend zero-monopole fields without introducing a new radial response."""
    fraction = float(right_fraction)
    audit = {
        "blend_right_fraction": fraction,
        "maximum_annular_convergence_mean_fraction": max(
            float(left.audit.get("maximum_annular_convergence_mean_fraction", 0.0)),
            float(right.audit.get("maximum_annular_convergence_mean_fraction", 0.0)),
        ),
        "normalized_curl_RMS": max(
            float(left.audit.get("normalized_curl_RMS", 0.0)),
            float(right.audit.get("normalized_curl_RMS", 0.0)),
        ),
    }
    return BlendedMorphologyDeflectionField(left, right, fraction, audit)


def build_stellar_morphology_deflection_field(
    axis_arcsec,
    positive_light,
    carrier_alpha_arcsec: Callable[[np.ndarray], np.ndarray],
    *,
    contrast_cap: float,
    contrast_mode: str = "hard",
    contrast_strength: float = 1.0,
    annulus_width_arcsec: float,
    taper_inner_arcsec: float,
    support_radius_arcsec: float,
    radial_samples: int = 2048,
    circular_radii: int = 512,
    circular_azimuths: int = 720,
) -> StellarMorphologyDeflectionField:
    """Build a curl-free redistribution field from a positive light template."""

    axis = np.asarray(axis_arcsec, dtype=float)
    light = np.asarray(positive_light, dtype=float)
    if axis.ndim != 1 or len(axis) < 64 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("axis must be a strictly increasing vector with at least 64 cells")
    if light.shape != (len(axis), len(axis)):
        raise ValueError("positive_light must be square on axis_arcsec")
    spacing = float(axis[1] - axis[0])
    if not np.allclose(np.diff(axis), spacing):
        raise ValueError("axis spacing must be uniform")
    if not 0.0 < taper_inner_arcsec < support_radius_arcsec < max(abs(axis[0]), abs(axis[-1])):
        raise ValueError("taper and support radii do not fit inside the grid")

    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    radius = np.hypot(xx, yy)
    sample_radius = np.geomspace(max(spacing / 20.0, 1.0e-3), np.max(radius) * 1.05, int(radial_samples))
    sample_alpha = np.asarray(carrier_alpha_arcsec(sample_radius), dtype=float)
    sample_kappa = radial_convergence_from_deflection(sample_radius, sample_alpha)
    carrier = np.interp(radius, sample_radius, sample_kappa)
    weights, weight_audit = normalized_light_weights(
        light,
        carrier,
        radius,
        contrast_cap=float(contrast_cap),
        contrast_mode=str(contrast_mode),
        annulus_width_arcsec=float(annulus_width_arcsec),
        support_radius_arcsec=float(support_radius_arcsec),
    )
    if not np.isfinite(contrast_strength) or not 0.0 <= contrast_strength <= 1.0:
        raise ValueError("contrast_strength must lie in [0, 1]")
    weights = 1.0 + float(contrast_strength) * (weights - 1.0)
    supported = radius <= support_radius_arcsec
    weight_audit = {
        **weight_audit,
        "contrast_strength": float(contrast_strength),
        "minimum_light_weight_after_strength": float(np.min(weights[supported])),
        "maximum_light_weight_after_strength": float(np.max(weights[supported])),
    }
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
    delta_kappa = taper * carrier * (weights - 1.0)
    # Remove the remaining discrete within-bin mean (normally roundoff plus
    # taper variation) so every projected annulus is exactly conservative.
    bins = np.floor(radius / float(annulus_width_arcsec)).astype(int)
    maximum_bin = int(np.floor(support_radius_arcsec / annulus_width_arcsec))
    maximum_convergence_error = 0.0
    for index in range(maximum_bin + 1):
        selected = (bins == index) & (radius <= support_radius_arcsec)
        if not np.any(selected):
            continue
        mean_delta = float(np.mean(delta_kappa[selected]))
        delta_kappa[selected] -= mean_delta
        scale = max(float(np.mean(np.abs(carrier[selected]))), np.finfo(float).tiny)
        maximum_convergence_error = max(
            maximum_convergence_error,
            abs(float(np.mean(delta_kappa[selected]))) / scale,
        )

    frequency = 2.0 * np.pi * np.fft.fftfreq(len(axis), d=spacing)
    kx, ky = np.meshgrid(frequency, frequency, indexing="xy")
    k2 = kx * kx + ky * ky
    safe_k2 = np.where(k2 > 0.0, k2, 1.0)
    kappa_hat = np.fft.fft2(delta_kappa)
    alpha_x_hat = -2.0j * kx * kappa_hat / safe_k2
    alpha_y_hat = -2.0j * ky * kappa_hat / safe_k2
    alpha_x_hat[0, 0] = 0.0
    alpha_y_hat[0, 0] = 0.0
    raw_x = np.fft.ifft2(alpha_x_hat).real
    raw_y = np.fft.ifft2(alpha_y_hat).real

    circular_radius = np.linspace(spacing, support_radius_arcsec + 20.0, int(circular_radii))
    phi = np.linspace(0.0, 2.0 * np.pi, int(circular_azimuths), endpoint=False)
    polar_x = circular_radius[:, None] * np.cos(phi)[None, :]
    polar_y = circular_radius[:, None] * np.sin(phi)[None, :]
    polar_ax = _sample_grid(axis, raw_x, polar_x, polar_y)
    polar_ay = _sample_grid(axis, raw_y, polar_x, polar_y)
    circular_mean = np.mean(
        polar_ax * np.cos(phi)[None, :] + polar_ay * np.sin(phi)[None, :],
        axis=1,
    )

    independent_phi = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    check_radius = np.geomspace(max(spacing, 0.5), support_radius_arcsec, 64)
    check_x = check_radius[:, None] * np.cos(independent_phi)[None, :]
    check_y = check_radius[:, None] * np.sin(independent_phi)[None, :]
    check_raw_x = _sample_grid(axis, raw_x, check_x, check_y)
    check_raw_y = _sample_grid(axis, raw_y, check_x, check_y)
    check_circular = np.interp(check_radius, circular_radius, circular_mean)
    check_radial = np.mean(
        (check_raw_x - check_circular[:, None] * np.cos(independent_phi)[None, :])
        * np.cos(independent_phi)[None, :]
        + (check_raw_y - check_circular[:, None] * np.sin(independent_phi)[None, :])
        * np.sin(independent_phi)[None, :],
        axis=1,
    )
    curl_hat = 1.0j * (kx * alpha_y_hat - ky * alpha_x_hat)
    divergence_hat = 1.0j * (kx * alpha_x_hat + ky * alpha_y_hat)
    curl = np.fft.ifft2(curl_hat).real
    divergence = np.fft.ifft2(divergence_hat).real
    normalized_curl = float(
        np.sqrt(np.mean(np.square(curl)))
        / max(np.sqrt(np.mean(np.square(divergence))), np.finfo(float).tiny)
    )
    edge = np.r_[raw_x[0], raw_x[-1], raw_x[:, 0], raw_x[:, -1], raw_y[0], raw_y[-1], raw_y[:, 0], raw_y[:, -1]]
    edge_kappa = np.r_[
        delta_kappa[0],
        delta_kappa[-1],
        delta_kappa[:, 0],
        delta_kappa[:, -1],
    ]
    audit = {
        **weight_audit,
        "grid_spacing_arcsec": spacing,
        "carrier_convergence_minimum": float(np.min(carrier[radius <= support_radius_arcsec])),
        "carrier_convergence_maximum": float(np.max(carrier[radius <= support_radius_arcsec])),
        "maximum_annular_convergence_mean_fraction": maximum_convergence_error,
        "delta_convergence_minimum": float(np.min(delta_kappa)),
        "delta_convergence_maximum": float(np.max(delta_kappa)),
        "raw_correction_RMS_arcsec": float(np.sqrt(np.mean(raw_x * raw_x + raw_y * raw_y))),
        "raw_correction_maximum_arcsec": float(np.max(np.hypot(raw_x, raw_y))),
        "maximum_independent_circular_mean_deflection_arcsec": float(np.max(np.abs(check_radial))),
        "normalized_curl_RMS": normalized_curl,
        "maximum_edge_correction_arcsec": float(np.max(np.abs(edge))),
        "maximum_edge_delta_convergence": float(np.max(np.abs(edge_kappa))),
    }
    return StellarMorphologyDeflectionField(
        axis,
        raw_x,
        raw_y,
        circular_radius,
        circular_mean,
        weights,
        delta_kappa,
        audit,
    )
