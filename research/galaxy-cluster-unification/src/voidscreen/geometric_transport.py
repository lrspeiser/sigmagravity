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
from scipy import fft, ndimage, sparse
from scipy.sparse import linalg as sparse_linalg

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


def symmetric_streamline_average(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    trace_length_pixels: np.ndarray,
    *,
    steps: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Average a vector flux forward and backward along curved streamlines.

    The integration length is supplied pointwise by field geometry. Forward
    and backward traces receive identical weight, so reversing the direction
    field leaves the result unchanged. Samples outside the map are omitted and
    the remaining weights are normalized locally.
    """

    fx = _square_map(flux_x, "flux_x")
    fy = _square_map(flux_y, "flux_y")
    dx = _square_map(direction_x, "direction_x")
    dy = _square_map(direction_y, "direction_y")
    length = _square_map(trace_length_pixels, "trace_length_pixels")
    if not (fx.shape == fy.shape == dx.shape == dy.shape == length.shape):
        raise ValueError("streamline fields must have matching shapes")
    if int(steps) != steps or int(steps) < 2:
        raise ValueError("streamline steps must be an integer of at least two")
    if np.any(length < 0.0):
        raise ValueError("trace lengths must be nonnegative")
    norm = np.hypot(dx, dy)
    unit_x = np.zeros_like(dx)
    unit_y = np.zeros_like(dy)
    active = norm > 1e-12
    unit_x[active] = dx[active] / norm[active]
    unit_y[active] = dy[active] / norm[active]
    rows, columns = np.indices(fx.shape, dtype=np.float64)
    summed_x = fx.copy()
    summed_y = fy.copy()
    weights = np.ones_like(fx)
    step_length = length / float(steps)
    maximum_row = float(fx.shape[0] - 1)
    maximum_column = float(fx.shape[1] - 1)
    for sign in (-1.0, 1.0):
        current_rows = rows.copy()
        current_columns = columns.copy()
        for _ in range(int(steps)):
            local_x = ndimage.map_coordinates(
                unit_x,
                [current_rows, current_columns],
                order=1,
                mode="constant",
                cval=0.0,
            )
            local_y = ndimage.map_coordinates(
                unit_y,
                [current_rows, current_columns],
                order=1,
                mode="constant",
                cval=0.0,
            )
            local_norm = np.hypot(local_x, local_y)
            valid_direction = local_norm > 1e-12
            local_x[valid_direction] /= local_norm[valid_direction]
            local_y[valid_direction] /= local_norm[valid_direction]
            current_columns += sign * step_length * local_x
            current_rows += sign * step_length * local_y
            valid = (
                valid_direction
                & (current_rows >= 0.0)
                & (current_rows <= maximum_row)
                & (current_columns >= 0.0)
                & (current_columns <= maximum_column)
            )
            sampled_x = ndimage.map_coordinates(
                fx,
                [current_rows, current_columns],
                order=1,
                mode="constant",
                cval=0.0,
            )
            sampled_y = ndimage.map_coordinates(
                fy,
                [current_rows, current_columns],
                order=1,
                mode="constant",
                cval=0.0,
            )
            summed_x += valid * sampled_x
            summed_y += valid * sampled_y
            weights += valid
    averaged_x = summed_x / weights
    averaged_y = summed_y / weights
    original_rms = float(np.sqrt(np.mean(fx * fx + fy * fy)))
    difference_rms = float(
        np.sqrt(np.mean(np.square(averaged_x - fx) + np.square(averaged_y - fy)))
    )
    return averaged_x, averaged_y, {
        "streamline_steps": int(steps),
        "mean_samples_per_cell": float(np.mean(weights)),
        "minimum_samples_per_cell": float(np.min(weights)),
        "maximum_samples_per_cell": float(np.max(weights)),
        "transport_relative_change_RMS": difference_rms
        / max(original_rms, np.finfo(float).tiny),
        "transport_flux_RMS_ratio": float(
            np.sqrt(np.mean(averaged_x * averaged_x + averaged_y * averaged_y))
            / max(original_rms, np.finfo(float).tiny)
        ),
    }


def _bilinear_deposit(
    output: np.ndarray,
    values: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    valid: np.ndarray,
) -> None:
    maximum = output.shape[0] - 1
    safe_rows = np.clip(rows, 0.0, float(maximum))
    safe_columns = np.clip(columns, 0.0, float(maximum))
    row0 = np.floor(safe_rows).astype(np.intp)
    column0 = np.floor(safe_columns).astype(np.intp)
    row1 = np.minimum(row0 + 1, maximum)
    column1 = np.minimum(column0 + 1, maximum)
    row_fraction = safe_rows - row0
    column_fraction = safe_columns - column0
    deposited = np.where(valid, values, 0.0)
    np.add.at(output, (row0, column0), deposited * (1.0 - row_fraction) * (1.0 - column_fraction))
    np.add.at(output, (row0, column1), deposited * (1.0 - row_fraction) * column_fraction)
    np.add.at(output, (row1, column0), deposited * row_fraction * (1.0 - column_fraction))
    np.add.at(output, (row1, column1), deposited * row_fraction * column_fraction)


def symmetric_streamline_deposit(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    trace_length_pixels: np.ndarray,
    *,
    steps: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Conservatively distribute each source flux along both streamline directions."""

    fx = _square_map(flux_x, "flux_x")
    fy = _square_map(flux_y, "flux_y")
    dx = _square_map(direction_x, "direction_x")
    dy = _square_map(direction_y, "direction_y")
    length = _square_map(trace_length_pixels, "trace_length_pixels")
    if not (fx.shape == fy.shape == dx.shape == dy.shape == length.shape):
        raise ValueError("streamline fields must have matching shapes")
    if int(steps) != steps or int(steps) < 2:
        raise ValueError("streamline steps must be an integer of at least two")
    if np.any(length < 0.0):
        raise ValueError("trace lengths must be nonnegative")
    norm = np.hypot(dx, dy)
    unit_x = np.zeros_like(dx)
    unit_y = np.zeros_like(dy)
    active = norm > 1e-12
    unit_x[active] = dx[active] / norm[active]
    unit_y[active] = dy[active] / norm[active]
    rows, columns = np.indices(fx.shape, dtype=np.float64)
    step_length = length / float(steps)
    maximum = float(fx.shape[0] - 1)
    destinations = []
    sample_counts = np.ones_like(fx)
    for sign in (-1.0, 1.0):
        current_rows = rows.copy()
        current_columns = columns.copy()
        for _ in range(int(steps)):
            local_x = ndimage.map_coordinates(
                unit_x, [current_rows, current_columns], order=1, mode="constant", cval=0.0
            )
            local_y = ndimage.map_coordinates(
                unit_y, [current_rows, current_columns], order=1, mode="constant", cval=0.0
            )
            local_norm = np.hypot(local_x, local_y)
            valid_direction = local_norm > 1e-12
            local_x[valid_direction] /= local_norm[valid_direction]
            local_y[valid_direction] /= local_norm[valid_direction]
            current_columns += sign * step_length * local_x
            current_rows += sign * step_length * local_y
            valid = (
                valid_direction
                & (current_rows >= 0.0)
                & (current_rows <= maximum)
                & (current_columns >= 0.0)
                & (current_columns <= maximum)
            )
            destinations.append((current_rows.copy(), current_columns.copy(), valid))
            sample_counts += valid
    shared_x = fx / sample_counts
    shared_y = fy / sample_counts
    deposited_x = shared_x.copy()
    deposited_y = shared_y.copy()
    for destination_rows, destination_columns, valid in destinations:
        _bilinear_deposit(deposited_x, shared_x, destination_rows, destination_columns, valid)
        _bilinear_deposit(deposited_y, shared_y, destination_rows, destination_columns, valid)
    original_rms = float(np.sqrt(np.mean(fx * fx + fy * fy)))
    difference_rms = float(
        np.sqrt(np.mean(np.square(deposited_x - fx) + np.square(deposited_y - fy)))
    )
    flux_sum_error = float(
        np.hypot(np.sum(deposited_x) - np.sum(fx), np.sum(deposited_y) - np.sum(fy))
        / max(np.sum(np.hypot(fx, fy)), np.finfo(float).tiny)
    )
    return deposited_x, deposited_y, {
        "streamline_steps": int(steps),
        "mean_samples_per_source_cell": float(np.mean(sample_counts)),
        "minimum_samples_per_source_cell": float(np.min(sample_counts)),
        "maximum_samples_per_source_cell": float(np.max(sample_counts)),
        "transport_relative_change_RMS": difference_rms
        / max(original_rms, np.finfo(float).tiny),
        "transport_flux_RMS_ratio": float(
            np.sqrt(np.mean(deposited_x * deposited_x + deposited_y * deposited_y))
            / max(original_rms, np.finfo(float).tiny)
        ),
        "transport_flux_sum_relative_error": flux_sum_error,
        "transport_is_source_conservative": True,
    }


def symmetric_field_line_diffusion(
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    trace_length_pixels: np.ndarray,
    conductance: np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
    maximum_iterations: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Implicitly diffuse vector flux on a symmetric field-aligned graph."""

    fx = _square_map(flux_x, "flux_x")
    fy = _square_map(flux_y, "flux_y")
    dx = _square_map(direction_x, "direction_x")
    dy = _square_map(direction_y, "direction_y")
    length = _square_map(trace_length_pixels, "trace_length_pixels")
    mobility = _square_map(conductance, "conductance")
    if not (fx.shape == fy.shape == dx.shape == dy.shape == length.shape == mobility.shape):
        raise ValueError("field-line diffusion inputs must have matching shapes")
    if np.any(length < 0.0) or np.any(mobility < 0.0) or np.any(mobility > 1.0):
        raise ValueError("trace lengths and conductance are outside their allowed ranges")
    if relative_tolerance <= 0.0 or int(maximum_iterations) < 1:
        raise ValueError("solver controls must be positive")
    norm = np.hypot(dx, dy)
    unit_x = np.zeros_like(dx)
    unit_y = np.zeros_like(dy)
    active = norm > 1e-12
    unit_x[active] = dx[active] / norm[active]
    unit_y[active] = dy[active] / norm[active]
    rows, columns = np.indices(fx.shape, dtype=np.float64)
    origins = np.arange(fx.size, dtype=np.intp).reshape(fx.shape)
    maximum = fx.shape[0] - 1
    edge_rows = []
    edge_columns = []
    edge_weights = []
    for sign in (-1.0, 1.0):
        destination_rows = rows + sign * unit_y
        destination_columns = columns + sign * unit_x
        valid_destination = (
            active
            & (destination_rows >= 0.0)
            & (destination_rows <= maximum)
            & (destination_columns >= 0.0)
            & (destination_columns <= maximum)
        )
        row0 = np.floor(np.clip(destination_rows, 0.0, maximum)).astype(np.intp)
        column0 = np.floor(np.clip(destination_columns, 0.0, maximum)).astype(np.intp)
        row1 = np.minimum(row0 + 1, maximum)
        column1 = np.minimum(column0 + 1, maximum)
        row_fraction = np.clip(destination_rows, 0.0, maximum) - row0
        column_fraction = np.clip(destination_columns, 0.0, maximum) - column0
        neighbors = (
            (row0, column0, (1.0 - row_fraction) * (1.0 - column_fraction)),
            (row0, column1, (1.0 - row_fraction) * column_fraction),
            (row1, column0, row_fraction * (1.0 - column_fraction)),
            (row1, column1, row_fraction * column_fraction),
        )
        for neighbor_rows, neighbor_columns, interpolation_weight in neighbors:
            destinations = origins[neighbor_rows, neighbor_columns]
            neighbor_mobility = mobility[neighbor_rows, neighbor_columns]
            weight = (
                0.5
                * np.square(length)
                * np.sqrt(mobility * neighbor_mobility)
                * interpolation_weight
            )
            selected = (
                valid_destination
                & (destinations != origins)
                & (weight > np.finfo(float).tiny)
            )
            edge_rows.append(origins[selected])
            edge_columns.append(destinations[selected])
            edge_weights.append(weight[selected])
    directed = sparse.coo_matrix(
        (
            np.concatenate(edge_weights),
            (np.concatenate(edge_rows), np.concatenate(edge_columns)),
        ),
        shape=(fx.size, fx.size),
    ).tocsr()
    adjacency = 0.5 * (directed + directed.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree) - adjacency
    operator = sparse.eye(fx.size, format="csr") + laplacian
    solved = []
    solver_information = []
    for component in (fx, fy):
        values, information = sparse_linalg.cg(
            operator,
            component.ravel(),
            rtol=float(relative_tolerance),
            atol=0.0,
            maxiter=int(maximum_iterations),
        )
        solved.append(values.reshape(fx.shape))
        solver_information.append(int(information))
    if any(information != 0 for information in solver_information):
        raise RuntimeError(f"field-line diffusion did not converge: {solver_information}")
    diffused_x, diffused_y = solved
    original_rms = float(np.sqrt(np.mean(fx * fx + fy * fy)))
    difference_rms = float(
        np.sqrt(np.mean(np.square(diffused_x - fx) + np.square(diffused_y - fy)))
    )
    flux_sum_error = float(
        np.hypot(np.sum(diffused_x) - np.sum(fx), np.sum(diffused_y) - np.sum(fy))
        / max(np.sum(np.hypot(fx, fy)), np.finfo(float).tiny)
    )
    component_scale = max(
        float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), np.finfo(float).tiny
    )
    overshoot = max(
        float(np.max(diffused_x) - np.max(fx)),
        float(np.min(fx) - np.min(diffused_x)),
        float(np.max(diffused_y) - np.max(fy)),
        float(np.min(fy) - np.min(diffused_y)),
        0.0,
    )
    return diffused_x, diffused_y, {
        "transport_operator": "symmetric_field_line_graph_diffusion",
        "transport_graph_edges": int(adjacency.nnz // 2),
        "transport_solver_relative_tolerance": float(relative_tolerance),
        "transport_solver_maximum_iterations": int(maximum_iterations),
        "transport_solver_information_x": solver_information[0],
        "transport_solver_information_y": solver_information[1],
        "transport_relative_change_RMS": difference_rms
        / max(original_rms, np.finfo(float).tiny),
        "transport_flux_RMS_ratio": float(
            np.sqrt(np.mean(diffused_x * diffused_x + diffused_y * diffused_y))
            / max(original_rms, np.finfo(float).tiny)
        ),
        "transport_flux_sum_relative_error": flux_sum_error,
        "transport_component_overshoot_fraction": overshoot / component_scale,
        "transport_is_source_conservative": True,
        "transport_is_self_adjoint_diffusion": True,
    }


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
