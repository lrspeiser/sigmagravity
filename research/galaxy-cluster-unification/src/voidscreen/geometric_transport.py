"""Baryon-only geometric activations for a conservative transport field law.

This module does not fit observations.  It constructs projected diagnostics of
the dimension-independent equation

    div[(nu I + lambda S C h h) grad(Phi_N)] = laplacian(Phi),

where every spatial field is calculated from baryons.  Solving the final
Poisson equation makes the resulting acceleration conservative even though the
intermediate tensor is anisotropic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage

G_SI = 6.67430e-11
KPC_M = 3.085677581491367e19
M_SUN_KG = 1.98847e30


@dataclass(frozen=True)
class ThinSheetField:
    potential_m2_s2: np.ndarray
    acceleration_x_m_s2: np.ndarray
    acceleration_y_m_s2: np.ndarray
    magnitude_m_s2: np.ndarray


@dataclass(frozen=True)
class PathGeometry:
    incoherence: np.ndarray
    mean_direction_x: np.ndarray
    mean_direction_y: np.ndarray
    trace_length_kpc: np.ndarray


def _square_map(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 17:
        raise ValueError(f"{name} must be a square 2D map with at least 17 cells")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def resample_surface_density(surface: np.ndarray, target_cells: int) -> np.ndarray:
    """Resample a fixed-extent surface-density map while preserving its integral."""

    values = _square_map(surface, "surface")
    target = int(target_cells)
    if target < 17 or target % 2 == 0:
        raise ValueError("target_cells must be an odd integer of at least 17")
    if target == values.shape[0]:
        return values.copy()
    coordinates = np.linspace(0.0, values.shape[0] - 1.0, target)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    sampled = ndimage.map_coordinates(values, [yy, xx], order=1, mode="constant", cval=0.0)
    old_sum = float(np.sum(values))
    if old_sum > 0.0 and float(np.sum(sampled)) > 0.0:
        spacing_ratio = (values.shape[0] - 1.0) / (target - 1.0)
        sampled *= old_sum / (float(np.sum(sampled)) * spacing_ratio**2)
    return sampled


def thin_sheet_newtonian_field(
    surface_density_msun_kpc2: np.ndarray,
    cell_kpc: float,
    *,
    gravitational_constant: float = G_SI,
    padding_factor: float = 2.0,
) -> ThinSheetField:
    """Return the in-plane Newtonian field of a zero-padded thin mass sheet.

    The Fourier Green function is ``Phi(k)=-2*pi*G*Sigma(k)/|k|``.  Padding
    suppresses periodic copies; the zero mode fixes an irrelevant potential
    gauge and never enters the acceleration.
    """

    surface = _square_map(surface_density_msun_kpc2, "surface_density")
    if np.any(surface < 0.0) or cell_kpc <= 0.0 or padding_factor < 1.5:
        raise ValueError("surface density, cell size, or padding is invalid")
    cells = surface.shape[0]
    padded_cells = fft.next_fast_len(int(np.ceil(float(padding_factor) * cells)))
    before = (padded_cells - cells) // 2
    after = padded_cells - cells - before
    padded = np.pad(surface, ((before, after), (before, after)), mode="constant")
    sigma_si = padded * M_SUN_KG / KPC_M**2
    step_m = float(cell_kpc) * KPC_M
    ky = 2.0 * np.pi * fft.fftfreq(padded_cells, d=step_m)
    kx = 2.0 * np.pi * fft.fftfreq(padded_cells, d=step_m)
    wave = np.hypot(ky[:, None], kx[None, :])
    sigma_hat = fft.fft2(sigma_si)
    phi_hat = np.zeros_like(sigma_hat, dtype=np.complex128)
    active = wave > 0.0
    phi_hat[active] = -2.0 * np.pi * float(gravitational_constant) * sigma_hat[active] / wave[active]
    ax_hat = -1j * kx[None, :] * phi_hat
    ay_hat = -1j * ky[:, None] * phi_hat
    region = (slice(before, before + cells), slice(before, before + cells))
    potential = np.real(fft.ifft2(phi_hat))[region]
    acceleration_x = np.real(fft.ifft2(ax_hat))[region]
    acceleration_y = np.real(fft.ifft2(ay_hat))[region]
    magnitude = np.hypot(acceleration_x, acceleration_y)
    return ThinSheetField(potential, acceleration_x, acceleration_y, magnitude)


def component_cancellation(
    first: ThinSheetField,
    second: ThinSheetField,
    *,
    relative_floor: float = 1e-12,
) -> np.ndarray:
    """Measure vector disagreement between two baryonic component fields."""

    if first.magnitude_m_s2.shape != second.magnitude_m_s2.shape:
        raise ValueError("component fields must have the same shape")
    denominator = first.magnitude_m_s2 + second.magnitude_m_s2
    total = np.hypot(
        first.acceleration_x_m_s2 + second.acceleration_x_m_s2,
        first.acceleration_y_m_s2 + second.acceleration_y_m_s2,
    )
    floor = float(np.max(denominator)) * float(relative_floor)
    result = np.zeros_like(denominator)
    active = denominator > floor
    result[active] = 1.0 - total[active] / denominator[active]
    return np.clip(result, 0.0, 1.0)


def component_angle_mismatch(
    first: ThinSheetField,
    second: ThinSheetField,
    *,
    mode: str = "quadratic_cancellation",
    relative_floor: float = 1e-12,
) -> np.ndarray:
    """Return a bounded disagreement measure for two component vectors.

    ``linear_chord_mix`` is first order in the angle near alignment and is
    weighted to vanish when either component disappears. ``oriented_cross_mix``
    uses the normalized cross-product magnitude. The legacy cancellation is
    retained as the quadratic control.
    """

    mode_id = str(mode)
    if mode_id == "quadratic_cancellation":
        return component_cancellation(first, second, relative_floor=relative_floor)
    first_magnitude = np.asarray(first.magnitude_m_s2, dtype=np.float64)
    second_magnitude = np.asarray(second.magnitude_m_s2, dtype=np.float64)
    if first_magnitude.shape != second_magnitude.shape:
        raise ValueError("component fields must have the same shape")
    denominator = first_magnitude + second_magnitude
    floor = float(np.max(denominator)) * float(relative_floor)
    active = (denominator > floor) & (first_magnitude > floor) & (second_magnitude > floor)
    dot = np.zeros_like(denominator)
    dot[active] = (
        first.acceleration_x_m_s2[active] * second.acceleration_x_m_s2[active]
        + first.acceleration_y_m_s2[active] * second.acceleration_y_m_s2[active]
    ) / (first_magnitude[active] * second_magnitude[active])
    dot = np.clip(dot, -1.0, 1.0)
    result = np.zeros_like(denominator)
    if mode_id == "linear_chord_mix":
        mixing = np.zeros_like(denominator)
        mixing[active] = (
            2.0
            * np.sqrt(first_magnitude[active] * second_magnitude[active])
            / denominator[active]
        )
        result[active] = mixing[active] * np.sqrt(0.5 * (1.0 - dot[active]))
    elif mode_id == "oriented_cross_mix":
        mixing = np.zeros_like(denominator)
        mixing[active] = (
            2.0
            * first_magnitude[active]
            * second_magnitude[active]
            / np.square(denominator[active])
        )
        result[active] = mixing[active] * np.sqrt(np.maximum(1.0 - dot[active] ** 2, 0.0))
    elif mode_id == "transverse_tensor_mix":
        mixing = np.zeros_like(denominator)
        mixing[active] = (
            2.0
            * np.sqrt(first_magnitude[active] * second_magnitude[active])
            / denominator[active]
        )
        result[active] = mixing[active] * np.maximum(1.0 - dot[active] ** 2, 0.0)
    else:
        raise ValueError(f"unknown component angle mismatch mode: {mode_id}")
    return np.clip(result, 0.0, 1.0)


def _tidal_trace_length_pixels(
    field: ThinSheetField,
    cell_kpc: float,
    beta: float,
    floor_cells: float,
    cap_cells: float,
) -> np.ndarray:
    step_m = float(cell_kpc) * KPC_M
    dax_dx = np.gradient(field.acceleration_x_m_s2, step_m, axis=1, edge_order=2)
    dax_dy = np.gradient(field.acceleration_x_m_s2, step_m, axis=0, edge_order=2)
    day_dx = np.gradient(field.acceleration_y_m_s2, step_m, axis=1, edge_order=2)
    day_dy = np.gradient(field.acceleration_y_m_s2, step_m, axis=0, edge_order=2)
    hessian_norm = np.sqrt(dax_dx**2 + dax_dy**2 + day_dx**2 + day_dy**2)
    positive = hessian_norm[hessian_norm > 0.0]
    numerical_floor = float(np.quantile(positive, 1e-3)) if positive.size else 1.0
    length_m = float(beta) * field.magnitude_m_s2 / np.maximum(hessian_norm, numerical_floor)
    length_pixels = length_m / step_m
    return np.clip(length_pixels, float(floor_cells), float(cap_cells))


def streamline_incoherence(
    field: ThinSheetField,
    cell_kpc: float,
    *,
    beta: float = 1.0,
    trace_steps: int = 24,
    trace_length_floor_cells: float = 1.0,
    trace_length_cap_cells: float = 48.0,
) -> PathGeometry:
    """Trace the Newtonian direction and return its path-averaged disagreement.

    Each cell advances along its local gravitational field for the adaptive,
    gauge-independent length ``ell=beta |g|/||grad g||_F``.  A radial field
    retains one direction along each ray and therefore has a near-zero result.
    """

    magnitude = np.asarray(field.magnitude_m_s2, dtype=np.float64)
    if trace_steps < 4 or beta <= 0.0:
        raise ValueError("trace_steps and beta must be positive")
    peak = float(np.max(magnitude))
    active = magnitude > max(peak * 1e-12, np.finfo(float).tiny)
    direction_x = np.zeros_like(magnitude)
    direction_y = np.zeros_like(magnitude)
    direction_x[active] = field.acceleration_x_m_s2[active] / magnitude[active]
    direction_y[active] = field.acceleration_y_m_s2[active] / magnitude[active]
    length_pixels = _tidal_trace_length_pixels(
        field,
        cell_kpc,
        beta,
        trace_length_floor_cells,
        trace_length_cap_cells,
    )
    rows, columns = np.indices(magnitude.shape, dtype=np.float64)
    current_rows = rows.copy()
    current_columns = columns.copy()
    summed_x = np.zeros_like(magnitude)
    summed_y = np.zeros_like(magnitude)
    summed_weight = np.zeros_like(magnitude)
    step_pixels = length_pixels / float(trace_steps)
    for index in range(int(trace_steps) + 1):
        sampled_x = ndimage.map_coordinates(
            direction_x, [current_rows, current_columns], order=1, mode="constant", cval=0.0
        )
        sampled_y = ndimage.map_coordinates(
            direction_y, [current_rows, current_columns], order=1, mode="constant", cval=0.0
        )
        norm = np.hypot(sampled_x, sampled_y)
        valid = norm > 1e-9
        sampled_x[valid] /= norm[valid]
        sampled_y[valid] /= norm[valid]
        weight = np.exp(-float(index) / max(float(trace_steps), 1.0)) * valid
        summed_x += weight * sampled_x
        summed_y += weight * sampled_y
        summed_weight += weight
        current_columns += step_pixels * sampled_x
        current_rows += step_pixels * sampled_y
    safe_weight = np.maximum(summed_weight, 1.0)
    mean_x = summed_x / safe_weight
    mean_y = summed_y / safe_weight
    coherence = np.clip(np.hypot(mean_x, mean_y), 0.0, 1.0)
    mean_norm = np.maximum(coherence, 1e-15)
    unit_x = mean_x / mean_norm
    unit_y = mean_y / mean_norm
    incoherence = np.where(active, 1.0 - coherence**2, 0.0)
    return PathGeometry(
        np.clip(incoherence, 0.0, 1.0),
        unit_x,
        unit_y,
        length_pixels * float(cell_kpc),
    )


def high_acceleration_screen(magnitude_m_s2: np.ndarray, a0_m_s2: float) -> np.ndarray:
    values = np.maximum(np.asarray(magnitude_m_s2, dtype=np.float64), 0.0)
    if a0_m_s2 <= 0.0:
        raise ValueError("a0 must be positive")
    return float(a0_m_s2) / (float(a0_m_s2) + values)


def hybrid_geometry(path_incoherence: np.ndarray, cancellation: np.ndarray) -> np.ndarray:
    first = np.asarray(path_incoherence, dtype=np.float64)
    second = np.asarray(cancellation, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("geometry fields must have the same shape")
    return np.clip(1.0 - (1.0 - first) * (1.0 - second), 0.0, 1.0)


def mass_centroid_and_r80(surface: np.ndarray, cell_kpc: float) -> tuple[float, float, float]:
    values = _square_map(surface, "surface")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("surface must have positive mass")
    rows, columns = np.indices(values.shape, dtype=float)
    center_row = float(np.sum(rows * values) / total)
    center_column = float(np.sum(columns * values) / total)
    radius_pixels = np.hypot(rows - center_row, columns - center_column)
    order = np.argsort(radius_pixels.ravel())
    cumulative = np.cumsum(values.ravel()[order])
    index = int(np.searchsorted(cumulative, 0.8 * total, side="left"))
    r80 = float(radius_pixels.ravel()[order[min(index, len(order) - 1)]] * cell_kpc)
    return center_column, center_row, r80


def aperture_weighted_statistics(
    values: np.ndarray,
    total_surface: np.ndarray,
    field_magnitude: np.ndarray,
    cell_kpc: float,
    *,
    inner_r80: float = 0.05,
    outer_r80: float = 1.2,
) -> dict[str, float]:
    geometry = _square_map(values, "values")
    surface = _square_map(total_surface, "total_surface")
    magnitude = _square_map(field_magnitude, "field_magnitude")
    center_x, center_y, r80 = mass_centroid_and_r80(surface, cell_kpc)
    rows, columns = np.indices(surface.shape, dtype=float)
    radius = np.hypot(rows - center_y, columns - center_x) * float(cell_kpc)
    mask = (radius >= inner_r80 * r80) & (radius <= outer_r80 * r80)
    floor = float(np.max(magnitude)) * 1e-8
    weights = np.where(mask & (magnitude > floor), magnitude, 0.0)
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("weighted aperture is empty")
    normalized = weights / float(np.sum(weights))
    flat_values = geometry[mask]
    return {
        "weighted_mean": float(np.sum(normalized * geometry)),
        "median": float(np.median(flat_values)),
        "p90": float(np.quantile(flat_values, 0.9)),
        "maximum": float(np.max(flat_values)),
        "r80_kpc": float(r80),
        "active_pixels": int(np.sum(mask)),
    }


def tensor_source_2d(
    field: ThinSheetField,
    path: PathGeometry,
    geometry: np.ndarray,
    cell_kpc: float,
    *,
    a0_m_s2: float,
    geometric_strength: float,
) -> np.ndarray:
    """Return the extra anisotropic source ``div(lambda S C h h grad Phi_N)``."""

    if geometric_strength < 0.0:
        raise ValueError("geometric_strength must be non-negative")
    activation = high_acceleration_screen(field.magnitude_m_s2, a0_m_s2) * np.asarray(
        geometry, dtype=np.float64
    )
    grad_x = -field.acceleration_x_m_s2
    grad_y = -field.acceleration_y_m_s2
    projection = path.mean_direction_x * grad_x + path.mean_direction_y * grad_y
    flux_x = float(geometric_strength) * activation * path.mean_direction_x * projection
    flux_y = float(geometric_strength) * activation * path.mean_direction_y * projection
    step_m = float(cell_kpc) * KPC_M
    return np.gradient(flux_x, step_m, axis=1, edge_order=2) + np.gradient(
        flux_y, step_m, axis=0, edge_order=2
    )


def spectral_poisson_acceleration_2d(
    source_s2: np.ndarray, cell_kpc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a zero-mean projected Poisson source and return a curl-free field."""

    source = _square_map(source_s2, "source")
    step_m = float(cell_kpc) * KPC_M
    ky = 2.0 * np.pi * fft.fftfreq(source.shape[0], d=step_m)
    kx = 2.0 * np.pi * fft.fftfreq(source.shape[1], d=step_m)
    wave2 = ky[:, None] ** 2 + kx[None, :] ** 2
    transformed = fft.fft2(source - float(np.mean(source)))
    potential_hat = np.zeros_like(transformed)
    active = wave2 > 0.0
    potential_hat[active] = -transformed[active] / wave2[active]
    ax = np.real(fft.ifft2(-1j * kx[None, :] * potential_hat))
    ay = np.real(fft.ifft2(-1j * ky[:, None] * potential_hat))
    potential = np.real(fft.ifft2(potential_hat))
    return potential, ax, ay


def normalized_discrete_curl(ax: np.ndarray, ay: np.ndarray, cell_kpc: float) -> float:
    step_m = float(cell_kpc) * KPC_M
    curl = np.gradient(ay, step_m, axis=1, edge_order=2) - np.gradient(
        ax, step_m, axis=0, edge_order=2
    )
    gradient_scale = np.sqrt(
        np.mean(np.gradient(ax, step_m, axis=1, edge_order=2) ** 2)
        + np.mean(np.gradient(ay, step_m, axis=0, edge_order=2) ** 2)
    )
    return float(np.sqrt(np.mean(curl**2)) / max(gradient_scale, np.finfo(float).tiny))
