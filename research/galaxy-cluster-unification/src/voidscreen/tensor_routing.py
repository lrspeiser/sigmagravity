"""Conservative anisotropic kernels driven by a baryonic point field."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def weighted_radii(positions, weights) -> tuple[np.ndarray, float, float, float]:
    """Return the barycentric center, R50, R80, and R50/R80."""
    xy = np.asarray(positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or w.shape != (len(xy),):
        raise ValueError("positions and weights have incompatible shapes")
    if np.any(~np.isfinite(xy)) or np.any(~np.isfinite(w)) or np.any(w < 0.0) or np.sum(w) <= 0.0:
        raise ValueError("positions and weights must be finite and non-negative")
    w = w / np.sum(w)
    center = np.sum(xy * w[:, None], axis=0)
    radius = np.linalg.norm(xy - center, axis=1)
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(w[order])
    radii = []
    for fraction in (0.5, 0.8):
        index = min(int(np.searchsorted(cumulative, fraction)), len(order) - 1)
        radii.append(float(radius[order[index]]))
    if radii[1] <= 0.0:
        raise ValueError("R80 must be positive")
    return center, radii[0], radii[1], radii[0] / radii[1]


def baryonic_field_frames(positions, weights, *, softening: float) -> dict[str, np.ndarray]:
    """Return inward, external-field, and softened tidal principal directions."""
    xy = np.asarray(positions, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    center, _, _, _ = weighted_radii(xy, w)
    inward = center[None, :] - xy
    radius = np.linalg.norm(inward, axis=1)
    inward /= np.maximum(radius[:, None], np.finfo(float).tiny)

    delta = xy[None, :, :] - xy[:, None, :]
    distance2 = np.sum(delta**2, axis=2)
    np.fill_diagonal(distance2, np.inf)
    softened = distance2 + float(softening) ** 2
    external = np.sum(w[None, :, None] * delta / softened[:, :, None] ** 1.5, axis=1)
    external_norm = np.linalg.norm(external, axis=1)
    external /= np.maximum(external_norm[:, None], np.finfo(float).tiny)
    external[external_norm == 0.0] = inward[external_norm == 0.0]

    inv3 = np.where(np.isfinite(softened), softened**-1.5, 0.0)
    inv5 = np.where(np.isfinite(softened), softened**-2.5, 0.0)
    tensor = np.zeros((len(xy), 2, 2), dtype=float)
    for a in range(2):
        for b in range(2):
            identity = 1.0 if a == b else 0.0
            tensor[:, a, b] = np.sum(
                w[None, :] * (3.0 * delta[:, :, a] * delta[:, :, b] * inv5 - identity * inv3),
                axis=1,
            )
    eigenvalue, eigenvector = np.linalg.eigh(tensor)
    principal_index = np.argmax(np.abs(eigenvalue), axis=1)
    principal = np.asarray(
        [eigenvector[index, :, principal_index[index]] for index in range(len(xy))]
    )
    sign = np.sign(np.sum(principal * inward, axis=1))
    sign[sign == 0.0] = 1.0
    principal *= sign[:, None]
    return {
        "center": center,
        "radius": radius,
        "inward": inward,
        "external": external,
        "tidal": principal,
    }


def anisotropic_gaussian_deposit(
    axis,
    centers,
    weights,
    principal_axes,
    *,
    geometric_sigma: float,
    axis_ratio: float,
    axis_samples: int = 7,
) -> np.ndarray:
    """Deposit a positive, normalized approximation to oriented Gaussians."""
    grid_axis = np.asarray(axis, dtype=float)
    xy = np.asarray(centers, dtype=float)
    w = np.asarray(weights, dtype=float)
    vectors = np.asarray(principal_axes, dtype=float)
    if len(grid_axis) < 3 or np.any(np.diff(grid_axis) <= 0.0):
        raise ValueError("axis must be increasing")
    if xy.shape != vectors.shape or xy.ndim != 2 or xy.shape[1] != 2 or w.shape != (len(xy),):
        raise ValueError("kernel arrays have incompatible shapes")
    if geometric_sigma <= 0.0 or axis_ratio < 1.0 or axis_samples < 3:
        raise ValueError("kernel widths and sample count are invalid")
    w = w / np.sum(w)
    spacing = float(grid_axis[1] - grid_axis[0])
    edges = np.r_[grid_axis - 0.5 * spacing, grid_axis[-1] + 0.5 * spacing]
    minor = float(geometric_sigma) / np.sqrt(float(axis_ratio))
    major = float(geometric_sigma) * np.sqrt(float(axis_ratio))
    extra = np.sqrt(max(0.0, major**2 - minor**2))
    if extra == 0.0:
        points = xy
        point_weights = w
    else:
        z = np.linspace(-2.5, 2.5, int(axis_samples))
        z_weights = np.exp(-0.5 * z**2)
        z_weights /= np.sum(z_weights)
        variance = float(np.sum(z_weights * z**2))
        z /= np.sqrt(variance)
        points = (xy[:, None, :] + extra * z[None, :, None] * vectors[:, None, :]).reshape(-1, 2)
        point_weights = (w[:, None] * z_weights[None, :]).ravel()
    histogram, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=[edges, edges], weights=point_weights
    )
    smoothed = gaussian_filter(histogram, sigma=minor / spacing, mode="constant", cval=0.0)
    total = float(np.sum(smoothed))
    if total <= 0.0:
        raise ValueError("anisotropic deposition lost all weight")
    return smoothed / total


def redistributed_cumulative_mass_tensor(
    radius,
    cumulative_mass,
    *,
    r80: float,
    length_over_r80: float,
    radius_exponent: float,
    width_over_r80: float,
    axis_ratio: float,
    bins: int = 1024,
) -> tuple[np.ndarray, float]:
    """Axisymmetric radial projection of an inward anisotropic route kernel."""
    r = np.asarray(radius, dtype=float)
    mass = np.asarray(cumulative_mass, dtype=float)
    if (
        r.ndim != 1
        or mass.shape != r.shape
        or len(r) < 2
        or np.any(np.diff(r) <= 0.0)
        or np.any(np.diff(mass) < -1.0e-12)
        or mass[-1] <= 0.0
        or r80 <= 0.0
        or length_over_r80 < 0.0
        or width_over_r80 < 0.0
        or axis_ratio < 1.0
    ):
        raise ValueError("invalid radial tensor redistribution")
    shells = np.diff(np.r_[0.0, mass])
    ratio = np.maximum(r / float(r80), 1.0e-6)
    travel = float(length_over_r80) * float(r80) * np.clip(
        ratio ** float(radius_exponent), 0.2, 3.0
    )
    positions = np.maximum(r - travel, 0.0)
    sigma = float(width_over_r80) * float(r80) * np.sqrt(float(axis_ratio))
    maximum = max(float(r[-1]), float(np.max(positions)) + 6.0 * sigma, float(r80))
    edges = np.linspace(0.0, maximum, int(bins) + 1)
    histogram, _ = np.histogram(positions, bins=edges, weights=shells)
    routed = histogram.astype(float)
    spacing = float(edges[1] - edges[0])
    if sigma > 0.0:
        routed = gaussian_filter1d(routed, sigma / spacing, mode="constant")
    routed *= float(mass[-1]) / float(np.sum(routed))
    cumulative = np.cumsum(routed)
    sampled = np.interp(r, np.r_[0.0, edges[1:]], np.r_[0.0, cumulative])
    conservation_error = abs(float(np.sum(routed)) / float(mass[-1]) - 1.0)
    return sampled, conservation_error


def curl_free_deflection_diagnostic(kappa, spacing: float) -> dict[str, float]:
    """Solve the periodic 2-D Poisson closure and report spectral residuals."""
    field = np.asarray(kappa, dtype=float)
    if field.ndim != 2 or np.any(~np.isfinite(field)) or spacing <= 0.0:
        raise ValueError("kappa and spacing must be finite")
    centered = field - np.mean(field)
    ky = 2.0 * np.pi * np.fft.fftfreq(field.shape[0], d=float(spacing))
    kx = 2.0 * np.pi * np.fft.fftfreq(field.shape[1], d=float(spacing))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx_grid**2 + ky_grid**2
    source_hat = np.fft.fft2(centered)
    psi_hat = np.zeros_like(source_hat, dtype=complex)
    nonzero = k2 > 0.0
    psi_hat[nonzero] = -2.0 * source_hat[nonzero] / k2[nonzero]
    alpha_x_hat = 1j * kx_grid * psi_hat
    alpha_y_hat = 1j * ky_grid * psi_hat
    curl_hat = 1j * kx_grid * alpha_y_hat - 1j * ky_grid * alpha_x_hat
    divergence_hat = 1j * kx_grid * alpha_x_hat + 1j * ky_grid * alpha_y_hat
    target_hat = 2.0 * source_hat
    norm = max(float(np.linalg.norm(target_hat)), np.finfo(float).tiny)
    return {
        "relative_curl_norm": float(np.linalg.norm(curl_hat) / norm),
        "relative_poisson_residual": float(np.linalg.norm(divergence_hat - target_hat) / norm),
    }
