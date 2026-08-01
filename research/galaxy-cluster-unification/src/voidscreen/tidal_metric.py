"""Curl-free first-order lens corrections from an observed member tidal tensor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import map_coordinates


def _validate_members(x, y, weights) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    member_x = np.asarray(x, dtype=float)
    member_y = np.asarray(y, dtype=float)
    light = np.asarray(weights, dtype=float)
    if member_x.ndim != 1 or member_y.shape != member_x.shape or light.shape != member_x.shape:
        raise ValueError("member coordinates and weights must be matching vectors")
    if len(member_x) < 3 or np.any(~np.isfinite(member_x)) or np.any(~np.isfinite(member_y)):
        raise ValueError("at least three finite member positions are required")
    if np.any(~np.isfinite(light)) or np.any(light < 0.0) or light.sum() <= 0.0:
        raise ValueError("member weights must be finite, nonnegative, and nonzero")
    return member_x, member_y, light / light.sum()


def normalized_spin2_tensor(
    x,
    y,
    member_x,
    member_y,
    weights,
    *,
    softening_arcsec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded Qxx and Qxy for the trace-free 2-D member tensor.

    The complex number Qxx+i Qxy has magnitude no greater than one.  Looping
    over members rather than materializing a (pixel, member) cube keeps map
    construction memory-bounded.
    """

    if not np.isfinite(softening_arcsec) or softening_arcsec <= 0.0:
        raise ValueError("softening_arcsec must be finite and positive")
    member_x, member_y, light = _validate_members(member_x, member_y, weights)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    numerator_xx = np.zeros_like(x)
    numerator_xy = np.zeros_like(x)
    denominator = np.zeros_like(x)
    h2 = float(softening_arcsec) ** 2
    for mx, my, weight in zip(member_x, member_y, light, strict=True):
        dx = x - mx
        dy = y - my
        r2 = dx * dx + dy * dy
        kernel = weight / np.square(r2 + h2)
        numerator_xx += kernel * (dx * dx - dy * dy)
        numerator_xy += kernel * (2.0 * dx * dy)
        denominator += kernel * r2
    safe = np.maximum(denominator, np.finfo(float).tiny)
    qxx = np.where(denominator > 0.0, numerator_xx / safe, 0.0)
    qxy = np.where(denominator > 0.0, numerator_xy / safe, 0.0)
    magnitude = np.hypot(qxx, qxy)
    rescale = np.maximum(1.0, magnitude)
    return qxx / rescale, qxy / rescale


@dataclass(frozen=True)
class TidalCorrectionField:
    """One unit-coupling deflection correction sampled on a square grid."""

    axis_arcsec: np.ndarray
    alpha_x_arcsec: np.ndarray
    alpha_y_arcsec: np.ndarray
    qxx: np.ndarray
    qxy: np.ndarray
    audit: dict[str, float]

    def alpha_arcsec(self, x_arcsec, y_arcsec) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.broadcast_arrays(
            np.asarray(x_arcsec, dtype=float), np.asarray(y_arcsec, dtype=float)
        )
        axis = self.axis_arcsec
        spacing = float(axis[1] - axis[0])
        coordinates = np.vstack(
            [
                ((y - axis[0]) / spacing).ravel(),
                ((x - axis[0]) / spacing).ravel(),
            ]
        )
        alpha_x = map_coordinates(
            self.alpha_x_arcsec,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(x.shape)
        alpha_y = map_coordinates(
            self.alpha_y_arcsec,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(y.shape)
        return alpha_x, alpha_y


def build_tidal_correction_field(
    member_x,
    member_y,
    weights,
    *,
    softening_arcsec: float,
    extra_alpha_arcsec: Callable[[np.ndarray], np.ndarray],
    half_width_arcsec: float = 256.0,
    pixels_per_axis: int = 512,
    polar_mean_radii: int = 384,
    polar_mean_azimuths: int = 720,
    subtract_circular_mean: bool = True,
) -> TidalCorrectionField:
    """Build Q_env and solve the linearized anisotropic Poisson equation.

    For baseline extra-potential deflection ``a=grad(phi)``, the first-order
    equation is ``laplacian(delta_phi)=-div(Q a)``.  Its Fourier solution is
    projected onto the longitudinal (gradient) component, guaranteeing that
    the returned lens correction is curl-free up to numerical roundoff.
    """

    member_x, member_y, light = _validate_members(member_x, member_y, weights)
    if half_width_arcsec <= 0.0 or pixels_per_axis < 64:
        raise ValueError("the square field must have positive width and at least 64 pixels")
    if polar_mean_radii < 32 or polar_mean_azimuths < 64:
        raise ValueError("the circular mean grid is too small")

    size = int(pixels_per_axis)
    spacing = 2.0 * float(half_width_arcsec) / size
    axis = -float(half_width_arcsec) + spacing * np.arange(size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    raw_qxx, raw_qxy = normalized_spin2_tensor(
        xx,
        yy,
        member_x,
        member_y,
        light,
        softening_arcsec=softening_arcsec,
    )

    maximum_radius = float(np.hypot(half_width_arcsec, half_width_arcsec))
    polar_radius = np.linspace(0.0, maximum_radius, int(polar_mean_radii))
    polar_phi = np.linspace(0.0, 2.0 * np.pi, int(polar_mean_azimuths), endpoint=False)
    polar_x = polar_radius[:, None] * np.cos(polar_phi)[None, :]
    polar_y = polar_radius[:, None] * np.sin(polar_phi)[None, :]
    polar_qxx, polar_qxy = normalized_spin2_tensor(
        polar_x,
        polar_y,
        member_x,
        member_y,
        light,
        softening_arcsec=softening_arcsec,
    )
    cos2 = np.cos(2.0 * polar_phi)[None, :]
    sin2 = np.sin(2.0 * polar_phi)[None, :]
    mean_radial = np.mean(polar_qxx * cos2 + polar_qxy * sin2, axis=1)
    mean_cross = np.mean(polar_qxy * cos2 - polar_qxx * sin2, axis=1)

    radius = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    local_radial = np.interp(radius.ravel(), polar_radius, mean_radial).reshape(radius.shape)
    local_cross = np.interp(radius.ravel(), polar_radius, mean_cross).reshape(radius.shape)
    local_cos2 = np.cos(2.0 * phi)
    local_sin2 = np.sin(2.0 * phi)
    circular_qxx = local_radial * local_cos2 - local_cross * local_sin2
    circular_qxy = local_radial * local_sin2 + local_cross * local_cos2
    if subtract_circular_mean:
        # Both the raw and circular tensors are bounded by one; dividing their
        # difference by two keeps every environmental eigenvalue in [-1, 1].
        qxx = 0.5 * (raw_qxx - circular_qxx)
        qxy = 0.5 * (raw_qxy - circular_qxy)
        solver_qxx = qxx
        solver_qxy = qxy
    else:
        # The full tensor retains the radial spin-2 stress.  Its circular part
        # is solved analytically below; only the decaying non-circular contrast
        # enters the periodic FFT, avoiding a false boundary discontinuity.
        qxx = raw_qxx.copy()
        qxy = raw_qxy.copy()
        solver_qxx = raw_qxx - circular_qxx
        solver_qxy = raw_qxy - circular_qxy
    q_magnitude = np.hypot(qxx, qxy)
    q_rescale = np.maximum(1.0, q_magnitude)
    qxx /= q_rescale
    qxy /= q_rescale

    safe_radius = np.maximum(radius, spacing / 20.0)
    radial_alpha = np.asarray(extra_alpha_arcsec(safe_radius), dtype=float)
    if radial_alpha.shape != radius.shape or np.any(~np.isfinite(radial_alpha)):
        raise ValueError("extra_alpha_arcsec must return one finite value per radius")
    base_x = radial_alpha * xx / safe_radius
    base_y = radial_alpha * yy / safe_radius
    base_x[radius == 0.0] = 0.0
    base_y[radius == 0.0] = 0.0
    flux_x = solver_qxx * base_x + solver_qxy * base_y
    flux_y = solver_qxy * base_x - solver_qxx * base_y

    frequency = 2.0 * np.pi * np.fft.fftfreq(size, d=spacing)
    kx, ky = np.meshgrid(frequency, frequency, indexing="xy")
    k2 = kx * kx + ky * ky
    flux_x_hat = np.fft.fft2(flux_x)
    flux_y_hat = np.fft.fft2(flux_y)
    dot_hat = kx * flux_x_hat + ky * flux_y_hat
    safe_k2 = np.where(k2 > 0.0, k2, 1.0)
    correction_x_hat = -kx * dot_hat / safe_k2
    correction_y_hat = -ky * dot_hat / safe_k2
    correction_x_hat[0, 0] = 0.0
    correction_y_hat[0, 0] = 0.0
    correction_x = np.fft.ifft2(correction_x_hat).real
    correction_y = np.fft.ifft2(correction_y_hat).real
    if not subtract_circular_mean:
        # For Q_circ = q_r (e_r e_r-e_phi e_phi), the longitudinal solution
        # of div(delta_alpha)=-div(Q_circ alpha_r e_r) is exactly
        # delta_alpha_r=-q_r alpha_r.  The circular cross component is purely
        # transverse and contributes no scalar lens-potential correction.
        correction_x -= local_radial * base_x
        correction_y -= local_radial * base_y

    curl_hat = 1j * (kx * correction_y_hat - ky * correction_x_hat)
    divergence_hat = 1j * (kx * correction_x_hat + ky * correction_y_hat)
    curl = np.fft.ifft2(curl_hat).real
    divergence = np.fft.ifft2(divergence_hat).real
    normalized_curl = float(
        np.sqrt(np.mean(np.square(curl)))
        / max(np.sqrt(np.mean(np.square(divergence))), np.finfo(float).tiny)
    )
    edge_values = np.r_[q_magnitude[0], q_magnitude[-1], q_magnitude[:, 0], q_magnitude[:, -1]]
    solver_magnitude = np.hypot(solver_qxx, solver_qxy)
    solver_edge_values = np.r_[
        solver_magnitude[0],
        solver_magnitude[-1],
        solver_magnitude[:, 0],
        solver_magnitude[:, -1],
    ]
    audit = {
        "grid_spacing_arcsec": float(spacing),
        "maximum_Q_eigenvalue": float(np.max(q_magnitude)),
        "RMS_Q_eigenvalue": float(np.sqrt(np.mean(np.square(q_magnitude)))),
        "maximum_edge_Q_eigenvalue": float(np.max(edge_values)),
        "maximum_solver_edge_Q_eigenvalue": float(np.max(solver_edge_values)),
        "maximum_abs_circular_cross_mean": float(np.max(np.abs(mean_cross))),
        "circular_mean_subtracted": bool(subtract_circular_mean),
        "correction_RMS_arcsec_at_distance_ratio_one": float(
            np.sqrt(np.mean(correction_x * correction_x + correction_y * correction_y))
        ),
        "correction_maximum_arcsec_at_distance_ratio_one": float(
            np.max(np.hypot(correction_x, correction_y))
        ),
        "normalized_curl_RMS": normalized_curl,
    }
    return TidalCorrectionField(axis, correction_x, correction_y, qxx, qxy, audit)
