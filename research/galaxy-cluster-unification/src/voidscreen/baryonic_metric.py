"""First-order, curl-free lens response from a baryon-built field metric.

The module treats the ordinary baryonic deflection as the zeroth-order field
and constructs a positive constitutive tensor from two baryonic observables:
the projected low-acceleration gate and the trace-free tidal direction.  The
linear response is projected onto a scalar potential, so the returned lens
correction cannot acquire an independent vector/curl mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .tidal_metric import TidalCorrectionField


G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_KG = 1.98847e30
KPC_M = 3.085677581491367e19
ARCSEC_PER_RADIAN = 206_264.80624709636


@dataclass(frozen=True)
class BaryonicMetricWorkspace:
    """Parameter-independent projected baryonic fields for one lens."""

    axis_arcsec: np.ndarray
    x_grid_arcsec: np.ndarray
    y_grid_arcsec: np.ndarray
    lens_alpha_x_arcsec: np.ndarray
    lens_alpha_y_arcsec: np.ndarray
    physical_acceleration_x_m_s2: np.ndarray
    physical_acceleration_y_m_s2: np.ndarray
    morphology: dict[str, float]
    scale_kpc_per_arcsec: float
    half_width_arcsec: float


@dataclass(frozen=True)
class BaryonicMetricState:
    """Smoothing-dependent acceleration magnitude and tidal direction."""

    smoothing_r80_fraction: float
    smoothing_width_arcsec: float
    acceleration_m_s2: np.ndarray
    qxx: np.ndarray
    qxy: np.ndarray


def low_acceleration_gate(acceleration_m_s2, a0_m_s2: float, power: float) -> np.ndarray:
    """Return ``1/[1+(g/a0)^power]`` with validated physical inputs."""
    acceleration = np.asarray(acceleration_m_s2, dtype=float)
    if np.any(~np.isfinite(acceleration)) or np.any(acceleration < 0.0):
        raise ValueError("acceleration must be finite and nonnegative")
    if not math.isfinite(a0_m_s2) or a0_m_s2 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("gate power must be finite and positive")
    ratio = acceleration / float(a0_m_s2)
    return 1.0 / (1.0 + np.power(ratio, float(power)))


def spherical_metric_acceleration(
    baryonic_acceleration_m_s2,
    *,
    minimum_permittivity: float,
    a0_m_s2: float,
    gate_power: float,
) -> np.ndarray:
    """Spherical exterior-field limit of the scalar part of the metric."""
    if not 0.0 < minimum_permittivity <= 1.0:
        raise ValueError("minimum permittivity must lie in (0,1]")
    gbar = np.asarray(baryonic_acceleration_m_s2, dtype=float)
    gate = low_acceleration_gate(gbar, a0_m_s2, gate_power)
    epsilon = 1.0 - (1.0 - float(minimum_permittivity)) * gate
    return gbar / epsilon


def weighted_morphology(x, y, weights) -> dict[str, float]:
    """Measure centroid, R80, and a decomposition-stable spin-2 asymmetry."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or weight.shape != x.shape or x.size == 0:
        raise ValueError("source positions and weights must be matching vectors")
    if (
        np.any(~np.isfinite(x))
        or np.any(~np.isfinite(y))
        or np.any(~np.isfinite(weight))
        or np.any(weight < 0.0)
        or np.sum(weight) <= 0.0
    ):
        raise ValueError("source positions and weights must be finite and nonnegative")
    weight = weight / np.sum(weight)
    center_x = float(np.sum(weight * x))
    center_y = float(np.sum(weight * y))
    dx = x - center_x
    dy = y - center_y
    radius = np.hypot(dx, dy)
    order = np.argsort(radius)
    cumulative = np.cumsum(weight[order])
    r80 = float(radius[order][min(np.searchsorted(cumulative, 0.8), len(order) - 1)])
    denominator = float(np.sum(weight * (dx * dx + dy * dy)))
    if denominator <= np.finfo(float).tiny:
        quadrupole = 0.0
    else:
        qxx = float(np.sum(weight * (dx * dx - dy * dy))) / denominator
        qxy = float(np.sum(weight * (2.0 * dx * dy))) / denominator
        quadrupole = float(np.clip(np.hypot(qxx, qxy), 0.0, 1.0))
    return {
        "center_x_arcsec": center_x,
        "center_y_arcsec": center_y,
        "r80_arcsec": r80,
        "quadrupole_asymmetry": quadrupole,
    }


def asymmetry_gate(quadrupole: float, threshold: float = 0.05, power: float = 4.0) -> float:
    """Smoothly suppress directional response for circular source layouts."""
    if not 0.0 <= quadrupole <= 1.0:
        raise ValueError("quadrupole must lie in [0,1]")
    if threshold <= 0.0 or power <= 0.0:
        raise ValueError("asymmetry threshold and power must be positive")
    numerator = float(quadrupole) ** float(power)
    return numerator / (numerator + float(threshold) ** float(power))


def _circular_potential_residual(
    potential: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    spacing: float,
) -> np.ndarray:
    radius = np.hypot(xx - center_x, yy - center_y)
    bins = np.floor(radius / float(spacing)).astype(int)
    count = np.bincount(bins.ravel())
    total = np.bincount(bins.ravel(), weights=potential.ravel())
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    radial_axis = (np.arange(len(mean), dtype=float) + 0.5) * float(spacing)
    circular = np.interp(radius.ravel(), radial_axis, mean, left=mean[0], right=mean[-1])
    return potential - circular.reshape(potential.shape)


def _validated_sources(x, y, weights, total_mass_msun: float):
    morphology = weighted_morphology(x, y, weights)
    source_x = np.asarray(x, dtype=float)
    source_y = np.asarray(y, dtype=float)
    # Make an owned copy: in-place normalization must never mutate a caller's
    # pandas Series or NumPy source-mass array.
    light = np.array(weights, dtype=float, copy=True)
    light /= np.sum(light)
    if not math.isfinite(total_mass_msun) or total_mass_msun <= 0.0:
        raise ValueError("total mass must be finite and positive")
    return source_x, source_y, light, light * float(total_mass_msun), morphology


def prepare_baryonic_metric_workspace(
    source_x_arcsec,
    source_y_arcsec,
    weights,
    *,
    total_mass_msun: float,
    scale_kpc_per_arcsec: float,
    half_width_arcsec: float = 256.0,
    pixels_per_axis: int = 256,
    point_softening_arcsec: float = 2.0,
) -> BaryonicMetricWorkspace:
    """Calculate expensive point-source fields once for a parameter sweep."""
    if not math.isfinite(scale_kpc_per_arcsec) or scale_kpc_per_arcsec <= 0.0:
        raise ValueError("angular scale must be finite and positive")
    if half_width_arcsec <= 0.0 or pixels_per_axis < 64 or point_softening_arcsec <= 0.0:
        raise ValueError("invalid field geometry")
    sx, sy, _, masses, morphology = _validated_sources(
        source_x_arcsec, source_y_arcsec, weights, total_mass_msun
    )
    size = int(pixels_per_axis)
    spacing = 2.0 * float(half_width_arcsec) / size
    axis = -float(half_width_arcsec) + spacing * np.arange(size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    soft2 = float(point_softening_arcsec) ** 2
    coefficient_per_msun = (
        4.0
        * G_SI
        * M_SUN_KG
        / (C_SI**2 * float(scale_kpc_per_arcsec) * KPC_M)
        * ARCSEC_PER_RADIAN
    )
    lens_x = np.zeros_like(xx)
    lens_y = np.zeros_like(yy)
    physical_x = np.zeros_like(xx)
    physical_y = np.zeros_like(yy)
    meter_per_arcsec = float(scale_kpc_per_arcsec) * KPC_M
    soft_m2 = (float(point_softening_arcsec) * meter_per_arcsec) ** 2
    for source_x, source_y, mass in zip(sx, sy, masses, strict=True):
        dx = xx - source_x
        dy = yy - source_y
        distance2 = dx * dx + dy * dy + soft2
        coefficient = coefficient_per_msun * mass
        lens_x += coefficient * dx / distance2
        lens_y += coefficient * dy / distance2
        dx_m = dx * meter_per_arcsec
        dy_m = dy * meter_per_arcsec
        distance_m2 = dx_m * dx_m + dy_m * dy_m + soft_m2
        inverse_cube = np.power(distance_m2, -1.5)
        physical_x += G_SI * M_SUN_KG * mass * dx_m * inverse_cube
        physical_y += G_SI * M_SUN_KG * mass * dy_m * inverse_cube
    return BaryonicMetricWorkspace(
        axis,
        xx,
        yy,
        lens_x,
        lens_y,
        physical_x,
        physical_y,
        morphology,
        float(scale_kpc_per_arcsec),
        float(half_width_arcsec),
    )


def prepare_baryonic_metric_state(
    workspace: BaryonicMetricWorkspace,
    smoothing_r80_fraction: float,
) -> BaryonicMetricState:
    """Calculate the smoothed acceleration and tidal tensor once per width."""
    if not math.isfinite(smoothing_r80_fraction) or smoothing_r80_fraction <= 0.0:
        raise ValueError("smoothing fraction must be finite and positive")
    spacing = float(workspace.axis_arcsec[1] - workspace.axis_arcsec[0])
    width_arcsec = max(
        float(smoothing_r80_fraction)
        * max(workspace.morphology["r80_arcsec"], spacing),
        spacing,
    )
    sigma_pixels = width_arcsec / spacing
    smooth_lens_x = gaussian_filter(
        workspace.lens_alpha_x_arcsec, sigma_pixels, mode="nearest"
    )
    smooth_lens_y = gaussian_filter(
        workspace.lens_alpha_y_arcsec, sigma_pixels, mode="nearest"
    )
    smooth_g_x = gaussian_filter(
        workspace.physical_acceleration_x_m_s2, sigma_pixels, mode="nearest"
    )
    smooth_g_y = gaussian_filter(
        workspace.physical_acceleration_y_m_s2, sigma_pixels, mode="nearest"
    )
    acceleration = np.hypot(smooth_g_x, smooth_g_y)
    d_ax_dx = np.gradient(smooth_lens_x, spacing, axis=1, edge_order=2)
    d_ax_dy = np.gradient(smooth_lens_x, spacing, axis=0, edge_order=2)
    d_ay_dx = np.gradient(smooth_lens_y, spacing, axis=1, edge_order=2)
    d_ay_dy = np.gradient(smooth_lens_y, spacing, axis=0, edge_order=2)
    tidal_xx = 0.5 * (d_ax_dx - d_ay_dy)
    tidal_xy = 0.5 * (d_ax_dy + d_ay_dx)
    tidal_norm = np.hypot(tidal_xx, tidal_xy)
    positive = tidal_norm[tidal_norm > 0.0]
    floor = max(
        float(np.percentile(positive, 1.0)) if positive.size else 0.0,
        np.finfo(float).tiny,
    )
    qxx = np.divide(
        tidal_xx,
        tidal_norm,
        out=np.zeros_like(tidal_xx),
        where=tidal_norm > floor,
    )
    qxy = np.divide(
        tidal_xy,
        tidal_norm,
        out=np.zeros_like(tidal_xy),
        where=tidal_norm > floor,
    )
    return BaryonicMetricState(
        float(smoothing_r80_fraction), float(width_arcsec), acceleration, qxx, qxy
    )


def build_baryonic_metric_correction_field(
    source_x_arcsec,
    source_y_arcsec,
    weights,
    *,
    total_mass_msun: float,
    scale_kpc_per_arcsec: float,
    minimum_permittivity: float,
    a0_m_s2: float,
    gate_power: float,
    anisotropy: float,
    smoothing_r80_fraction: float,
    half_width_arcsec: float = 256.0,
    pixels_per_axis: int = 256,
    point_softening_arcsec: float = 2.0,
    asymmetry_threshold: float = 0.05,
    asymmetry_power: float = 4.0,
    subtract_circular_mean: bool = True,
    workspace: BaryonicMetricWorkspace | None = None,
    state: BaryonicMetricState | None = None,
) -> TidalCorrectionField:
    """Build the first-order lens correction for one baryonic metric.

    The constitutive tensor is

    ``K = epsilon exp[anisotropy * S * H * Q]``

    with ``epsilon=1-(1-epsilon0)S``.  ``S`` is calculated from the projected
    baryonic acceleration, ``Q`` is the normalized tidal spin-2 tensor, and
    ``H`` is a global baryonic asymmetry gate.  The correction solves the
    linearized equation ``laplacian(delta psi)=-div[(K-I) grad(psi_b)]``.
    """
    if not 0.0 < minimum_permittivity <= 1.0:
        raise ValueError("minimum permittivity must lie in (0,1]")
    if not math.isfinite(anisotropy):
        raise ValueError("anisotropy must be finite")
    if workspace is None:
        workspace = prepare_baryonic_metric_workspace(
            source_x_arcsec,
            source_y_arcsec,
            weights,
            total_mass_msun=total_mass_msun,
            scale_kpc_per_arcsec=scale_kpc_per_arcsec,
            half_width_arcsec=half_width_arcsec,
            pixels_per_axis=pixels_per_axis,
            point_softening_arcsec=point_softening_arcsec,
        )
    if state is None:
        state = prepare_baryonic_metric_state(workspace, smoothing_r80_fraction)
    elif not math.isclose(
        state.smoothing_r80_fraction,
        float(smoothing_r80_fraction),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("state smoothing fraction does not match the requested value")
    axis = workspace.axis_arcsec
    xx = workspace.x_grid_arcsec
    yy = workspace.y_grid_arcsec
    lens_x = workspace.lens_alpha_x_arcsec
    lens_y = workspace.lens_alpha_y_arcsec
    morphology = workspace.morphology
    size = len(axis)
    spacing = float(axis[1] - axis[0])
    half_width_arcsec = workspace.half_width_arcsec
    scale_kpc_per_arcsec = workspace.scale_kpc_per_arcsec
    width_arcsec = state.smoothing_width_arcsec
    acceleration = state.acceleration_m_s2
    qxx = state.qxx
    qxy = state.qxy
    gate = low_acceleration_gate(acceleration, a0_m_s2, gate_power)

    directional_gate = asymmetry_gate(
        morphology["quadrupole_asymmetry"], asymmetry_threshold, asymmetry_power
    )
    epsilon = 1.0 - (1.0 - float(minimum_permittivity)) * gate
    chi = float(anisotropy) * gate * directional_gate
    cosine = np.cosh(chi)
    sine = np.sinh(chi)
    kxx = epsilon * (cosine + sine * qxx)
    kxy = epsilon * sine * qxy
    kyy = epsilon * (cosine - sine * qxx)
    mxx = kxx - 1.0
    mxy = kxy
    myy = kyy - 1.0

    radius_from_grid_center = np.hypot(xx, yy)
    taper_start = 0.72 * float(half_width_arcsec)
    taper_end = 0.96 * float(half_width_arcsec)
    taper = np.ones_like(xx)
    transition = (radius_from_grid_center > taper_start) & (radius_from_grid_center < taper_end)
    taper[radius_from_grid_center >= taper_end] = 0.0
    taper[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            * (radius_from_grid_center[transition] - taper_start)
            / (taper_end - taper_start)
        )
    )
    flux_x = taper * (mxx * lens_x + mxy * lens_y)
    flux_y = taper * (mxy * lens_x + myy * lens_y)

    frequency = 2.0 * np.pi * np.fft.fftfreq(size, d=spacing)
    kx, ky = np.meshgrid(frequency, frequency, indexing="xy")
    k2 = kx * kx + ky * ky
    dot_hat = kx * np.fft.fft2(flux_x) + ky * np.fft.fft2(flux_y)
    safe_k2 = np.where(k2 > 0.0, k2, 1.0)
    potential_hat = 1j * dot_hat / safe_k2
    potential_hat[0, 0] = 0.0
    potential = np.fft.ifft2(potential_hat).real
    full_y, full_x = np.gradient(potential, spacing, spacing, edge_order=2)
    if subtract_circular_mean:
        residual_potential = _circular_potential_residual(
            potential,
            xx,
            yy,
            morphology["center_x_arcsec"],
            morphology["center_y_arcsec"],
            spacing,
        )
        correction_y, correction_x = np.gradient(
            residual_potential, spacing, spacing, edge_order=2
        )
    else:
        correction_x, correction_y = full_x, full_y

    d_y_dx = np.gradient(correction_y, spacing, axis=1, edge_order=2)
    d_x_dy = np.gradient(correction_x, spacing, axis=0, edge_order=2)
    d_x_dx = np.gradient(correction_x, spacing, axis=1, edge_order=2)
    d_y_dy = np.gradient(correction_y, spacing, axis=0, edge_order=2)
    interior = radius_from_grid_center <= 0.65 * float(half_width_arcsec)
    curl = d_y_dx - d_x_dy
    divergence = d_x_dx + d_y_dy
    normalized_curl = float(
        np.sqrt(np.mean(np.square(curl[interior])))
        / max(np.sqrt(np.mean(np.square(divergence[interior]))), np.finfo(float).tiny)
    )
    full_rms = float(np.sqrt(np.mean(full_x[interior] ** 2 + full_y[interior] ** 2)))
    residual_rms = float(
        np.sqrt(np.mean(correction_x[interior] ** 2 + correction_y[interior] ** 2))
    )
    q_radius = np.hypot(qxx, qxy)
    eigen_low = epsilon * np.exp(-np.abs(chi) * q_radius)
    eigen_high = epsilon * np.exp(np.abs(chi) * q_radius)
    audit = {
        **morphology,
        "grid_spacing_arcsec": float(spacing),
        "smoothing_width_arcsec": float(width_arcsec),
        "smoothing_width_kpc": float(width_arcsec * scale_kpc_per_arcsec),
        "asymmetry_gate": float(directional_gate),
        "gate_minimum": float(np.min(gate[interior])),
        "gate_median": float(np.median(gate[interior])),
        "gate_maximum": float(np.max(gate[interior])),
        "metric_minimum_eigenvalue": float(np.min(eigen_low[interior])),
        "metric_maximum_eigenvalue": float(np.max(eigen_high[interior])),
        "maximum_Q_eigenvalue": float(np.max(q_radius[interior])),
        "RMS_Q_eigenvalue": float(np.sqrt(np.mean(q_radius[interior] ** 2))),
        "maximum_edge_Q_eigenvalue": float(np.max(q_radius[~interior])),
        "maximum_solver_edge_Q_eigenvalue": float(np.max((taper * q_radius)[~interior])),
        "maximum_abs_circular_cross_mean": 0.0,
        "circular_mean_subtracted": bool(subtract_circular_mean),
        "full_metric_correction_RMS_arcsec": full_rms,
        "circular_residual_fraction": residual_rms / max(full_rms, np.finfo(float).tiny),
        "correction_RMS_arcsec_at_distance_ratio_one": residual_rms,
        "correction_maximum_arcsec_at_distance_ratio_one": float(
            np.max(np.hypot(correction_x[interior], correction_y[interior]))
        ),
        "normalized_curl_RMS": normalized_curl,
    }
    return TidalCorrectionField(axis, correction_x, correction_y, qxx, qxy, audit)


def _affine_fit(alpha_x, alpha_y, dx, dy, mask, mode: str):
    x = np.asarray(dx, dtype=float)[mask]
    y = np.asarray(dy, dtype=float)[mask]
    ax = np.asarray(alpha_x, dtype=float)[mask]
    ay = np.asarray(alpha_y, dtype=float)[mask]
    if mode == "trace":
        design_x = np.column_stack([x, np.ones_like(x), np.zeros_like(x)])
        design_y = np.column_stack([y, np.zeros_like(y), np.ones_like(y)])
    elif mode == "symmetric":
        design_x = np.column_stack(
            [x, y, np.zeros_like(x), np.ones_like(x), np.zeros_like(x)]
        )
        design_y = np.column_stack(
            [np.zeros_like(y), x, y, np.zeros_like(y), np.ones_like(y)]
        )
    else:
        raise ValueError("affine mode must be 'trace' or 'symmetric'")
    design = np.vstack([design_x, design_y])
    target = np.r_[ax, ay]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    predicted = design @ coefficients
    centered = target - np.mean(target)
    denominator = float(np.sum(np.square(centered)))
    r2 = (
        float(1.0 - np.sum(np.square(target - predicted)) / denominator)
        if denominator > np.finfo(float).tiny
        else 0.0
    )
    return coefficients, r2


def remove_baryonic_affine_modes(
    field: TidalCorrectionField,
    *,
    aperture_r80_fraction: float,
    removal_fraction: float = 1.0,
    mode: str = "symmetric",
    taper_outer_factor: float = 2.0,
) -> TidalCorrectionField:
    """Remove long-wavelength potential modes in a baryon-defined aperture.

    The affine coefficients are fitted uniformly on a circular grid aperture
    centered on the field's baryonic centroid.  Neither image positions nor a
    lensing target enter the fit.  A quadratic scalar potential generates the
    symmetric affine deflection; a cosine window keeps its subtraction local
    while preserving a single curl-free potential.
    """
    if not math.isfinite(aperture_r80_fraction) or aperture_r80_fraction <= 0.0:
        raise ValueError("affine aperture fraction must be finite and positive")
    if not math.isfinite(removal_fraction) or not 0.0 <= removal_fraction <= 1.0:
        raise ValueError("affine removal fraction must lie in [0,1]")
    if not math.isfinite(taper_outer_factor) or taper_outer_factor <= 1.0:
        raise ValueError("taper outer factor must exceed one")
    audit = dict(field.audit)
    required = {"center_x_arcsec", "center_y_arcsec", "r80_arcsec"}
    if not required.issubset(audit):
        raise ValueError("field audit lacks baryonic centroid or R80")
    axis = np.asarray(field.axis_arcsec, dtype=float)
    spacing = float(axis[1] - axis[0])
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    dx = xx - float(audit["center_x_arcsec"])
    dy = yy - float(audit["center_y_arcsec"])
    radius = np.hypot(dx, dy)
    aperture = float(aperture_r80_fraction) * float(audit["r80_arcsec"])
    mask = radius <= aperture
    if np.count_nonzero(mask) < 32:
        raise ValueError("affine aperture contains too few grid cells")
    coefficients, pre_r2 = _affine_fit(
        field.alpha_x_arcsec,
        field.alpha_y_arcsec,
        dx,
        dy,
        mask,
        mode,
    )
    if mode == "trace":
        trace, constant_x, constant_y = coefficients
        polynomial = (
            0.5 * trace * (dx * dx + dy * dy)
            + constant_x * dx
            + constant_y * dy
        )
    else:
        axx, axy, ayy, constant_x, constant_y = coefficients
        polynomial = (
            0.5 * axx * dx * dx
            + axy * dx * dy
            + 0.5 * ayy * dy * dy
            + constant_x * dx
            + constant_y * dy
        )
    outer = float(taper_outer_factor) * aperture
    window = np.ones_like(radius)
    transition = (radius > aperture) & (radius < outer)
    window[radius >= outer] = 0.0
    window[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (radius[transition] - aperture) / (outer - aperture))
    )
    removed_potential = float(removal_fraction) * window * polynomial
    removed_y, removed_x = np.gradient(
        removed_potential, spacing, spacing, edge_order=2
    )
    result_x = np.asarray(field.alpha_x_arcsec, dtype=float) - removed_x
    result_y = np.asarray(field.alpha_y_arcsec, dtype=float) - removed_y
    _, post_r2 = _affine_fit(result_x, result_y, dx, dy, mask, mode)
    d_y_dx = np.gradient(result_y, spacing, axis=1, edge_order=2)
    d_x_dy = np.gradient(result_x, spacing, axis=0, edge_order=2)
    d_x_dx = np.gradient(result_x, spacing, axis=1, edge_order=2)
    d_y_dy = np.gradient(result_y, spacing, axis=0, edge_order=2)
    curl = d_y_dx - d_x_dy
    divergence = d_x_dx + d_y_dy
    interior = radius <= min(1.5 * outer, 0.75 * (axis[-1] - axis[0]))
    normalized_curl = float(
        np.sqrt(np.mean(np.square(curl[interior])))
        / max(np.sqrt(np.mean(np.square(divergence[interior]))), np.finfo(float).tiny)
    )
    audit.update(
        {
            "affine_removal_mode": mode,
            "affine_aperture_r80_fraction": float(aperture_r80_fraction),
            "affine_aperture_arcsec": aperture,
            "affine_removal_fraction": float(removal_fraction),
            "affine_taper_outer_factor": float(taper_outer_factor),
            "baryon_grid_affine_R2_before": pre_r2,
            "baryon_grid_affine_R2_after": post_r2,
            "removed_affine_RMS_arcsec": float(
                np.sqrt(np.mean(removed_x[mask] ** 2 + removed_y[mask] ** 2))
            ),
            "correction_RMS_arcsec_at_distance_ratio_one": float(
                np.sqrt(np.mean(result_x[mask] ** 2 + result_y[mask] ** 2))
            ),
            "correction_maximum_arcsec_at_distance_ratio_one": float(
                np.max(np.hypot(result_x[mask], result_y[mask]))
            ),
            "normalized_curl_RMS": normalized_curl,
        }
    )
    return TidalCorrectionField(axis, result_x, result_y, field.qxx, field.qxy, audit)
