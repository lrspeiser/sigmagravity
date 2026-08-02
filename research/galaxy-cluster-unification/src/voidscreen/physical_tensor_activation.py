"""Resolution-aware physical tidal lengths for tensor-AQUAL activation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.geometric_transport import (
    KPC_M,
    ThinSheetField,
    component_angle_mismatch,
    high_acceleration_screen,
    thin_sheet_newtonian_field,
)
from voidscreen.tensor_aqual import simple_mu


@dataclass(frozen=True)
class PhysicalTensorActivation2D:
    sigma: np.ndarray
    transverse_mismatch: np.ndarray
    survival: np.ndarray
    high_acceleration_screen: np.ndarray
    trace_length_kpc: np.ndarray
    transport_direction_x: np.ndarray
    transport_direction_y: np.ndarray
    mu_newtonian_proxy: np.ndarray
    minimum_eigenvalue_proxy: np.ndarray
    maximum_eigenvalue_proxy: np.ndarray
    total_field: ThinSheetField


def _normalized_field_direction(field: ThinSheetField) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.asarray(field.magnitude_m_s2, dtype=float)
    active = magnitude > max(float(np.max(magnitude)) * 1e-12, np.finfo(float).tiny)
    direction_x = np.zeros_like(magnitude)
    direction_y = np.zeros_like(magnitude)
    direction_x[active] = field.acceleration_x_m_s2[active] / magnitude[active]
    direction_y[active] = field.acceleration_y_m_s2[active] / magnitude[active]
    return direction_x, direction_y


def forward_boundary_distance_kpc(
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    cell_kpc: float,
) -> np.ndarray:
    """Distance from each cell to the square boundary along a unit direction."""

    h_x = np.asarray(direction_x, dtype=float)
    h_y = np.asarray(direction_y, dtype=float)
    if h_x.ndim != 2 or h_x.shape != h_y.shape or cell_kpc <= 0.0:
        raise ValueError("directions must be matching 2D maps and cell_kpc positive")
    rows, columns = np.indices(h_x.shape, dtype=float)
    maximum_row = float(h_x.shape[0] - 1)
    maximum_column = float(h_x.shape[1] - 1)
    infinity = np.full_like(h_x, np.inf)
    distance_x = infinity.copy()
    positive_x = h_x > 1e-15
    negative_x = h_x < -1e-15
    distance_x[positive_x] = (
        maximum_column - columns[positive_x]
    ) * float(cell_kpc) / h_x[positive_x]
    distance_x[negative_x] = columns[negative_x] * float(cell_kpc) / (-h_x[negative_x])
    distance_y = infinity.copy()
    positive_y = h_y > 1e-15
    negative_y = h_y < -1e-15
    distance_y[positive_y] = (
        maximum_row - rows[positive_y]
    ) * float(cell_kpc) / h_y[positive_y]
    distance_y[negative_y] = rows[negative_y] * float(cell_kpc) / (-h_y[negative_y])
    distance = np.minimum(distance_x, distance_y)
    map_diagonal = float(np.hypot(maximum_row, maximum_column) * float(cell_kpc))
    return np.where(np.isfinite(distance), np.maximum(distance, 0.0), map_diagonal)


def physical_tidal_length_kpc(field: ThinSheetField, cell_kpc: float) -> np.ndarray:
    """Return ``min(|g|/||grad g||_F, boundary distance)`` in physical units."""

    if cell_kpc <= 0.0:
        raise ValueError("cell_kpc must be positive")
    step_m = float(cell_kpc) * KPC_M
    ax = np.asarray(field.acceleration_x_m_s2, dtype=float)
    ay = np.asarray(field.acceleration_y_m_s2, dtype=float)
    dax_dx = np.gradient(ax, step_m, axis=1, edge_order=2)
    dax_dy = np.gradient(ax, step_m, axis=0, edge_order=2)
    day_dx = np.gradient(ay, step_m, axis=1, edge_order=2)
    day_dy = np.gradient(ay, step_m, axis=0, edge_order=2)
    gradient_norm = np.sqrt(dax_dx**2 + dax_dy**2 + day_dx**2 + day_dy**2)
    maximum_gradient = float(np.max(gradient_norm))
    numerical_floor = max(maximum_gradient * 1e-12, np.finfo(float).tiny)
    raw_length_kpc = (
        np.asarray(field.magnitude_m_s2, dtype=float)
        / np.maximum(gradient_norm, numerical_floor)
        / KPC_M
    )
    direction_x, direction_y = _normalized_field_direction(field)
    boundary = forward_boundary_distance_kpc(direction_x, direction_y, cell_kpc)
    length = np.minimum(np.maximum(raw_length_kpc, 0.0), boundary)
    return np.where(np.isfinite(length), length, 0.0)


def _component_difference_direction(
    stars: ThinSheetField,
    gas: ThinSheetField,
    total: ThinSheetField,
) -> tuple[np.ndarray, np.ndarray]:
    tiny = np.finfo(float).tiny
    star_norm = np.maximum(stars.magnitude_m_s2, tiny)
    gas_norm = np.maximum(gas.magnitude_m_s2, tiny)
    difference_x = gas.acceleration_x_m_s2 / gas_norm - stars.acceleration_x_m_s2 / star_norm
    difference_y = gas.acceleration_y_m_s2 / gas_norm - stars.acceleration_y_m_s2 / star_norm
    difference_norm = np.hypot(difference_x, difference_y)
    fallback_x, fallback_y = _normalized_field_direction(total)
    active = difference_norm > 1e-12
    direction_x = np.where(
        active,
        difference_x / np.maximum(difference_norm, 1e-12),
        fallback_x,
    )
    direction_y = np.where(
        active,
        difference_y / np.maximum(difference_norm, 1e-12),
        fallback_y,
    )
    norm = np.hypot(direction_x, direction_y)
    valid = norm > 1e-12
    direction_x = np.where(valid, direction_x / np.maximum(norm, 1e-12), 1.0)
    direction_y = np.where(valid, direction_y / np.maximum(norm, 1e-12), 0.0)
    return direction_x, direction_y


def exact_physical_tensor_activation(
    stellar_surface_density_msun_kpc2: np.ndarray,
    gas_surface_density_msun_kpc2: np.ndarray,
    cell_kpc: float,
    *,
    a0_m_s2: float = 1.2e-10,
    coherence_length_kpc: float = 10.0,
    coherence_power: float = 2.0,
    mu_floor: float = 1e-6,
) -> PhysicalTensorActivation2D:
    stars = np.maximum(np.asarray(stellar_surface_density_msun_kpc2, dtype=float), 0.0)
    gas = np.maximum(np.asarray(gas_surface_density_msun_kpc2, dtype=float), 0.0)
    if stars.ndim != 2 or stars.shape != gas.shape or min(stars.shape) < 9:
        raise ValueError("stellar and gas inputs must be matching 2D maps")
    if not np.all(np.isfinite(stars)) or not np.all(np.isfinite(gas)):
        raise ValueError("surface-density maps must be finite")
    if cell_kpc <= 0.0 or a0_m_s2 <= 0.0 or coherence_length_kpc <= 0.0:
        raise ValueError("physical scales must be positive")
    if coherence_power <= 0.0 or mu_floor <= 0.0:
        raise ValueError("coherence power and mu floor must be positive")

    star_field = thin_sheet_newtonian_field(stars, cell_kpc)
    gas_field = thin_sheet_newtonian_field(gas, cell_kpc)
    total_field = thin_sheet_newtonian_field(stars + gas, cell_kpc)
    mismatch = component_angle_mismatch(
        star_field,
        gas_field,
        mode="transverse_tensor_mix",
    )
    trace_length = physical_tidal_length_kpc(total_field, cell_kpc)
    survival = 1.0 - np.exp(
        -np.power(
            np.maximum(trace_length / float(coherence_length_kpc), 0.0),
            float(coherence_power),
        )
    )
    screen = high_acceleration_screen(total_field.magnitude_m_s2, a0_m_s2)
    sigma = np.clip(screen * mismatch * survival, 0.0, 1.0)
    direction_x, direction_y = _component_difference_direction(
        star_field,
        gas_field,
        total_field,
    )
    mu_proxy = np.maximum(
        simple_mu(total_field.magnitude_m_s2 / float(a0_m_s2)),
        float(mu_floor),
    )
    minimum_eigenvalue = mu_proxy * (1.0 - sigma)
    return PhysicalTensorActivation2D(
        sigma=sigma,
        transverse_mismatch=mismatch,
        survival=survival,
        high_acceleration_screen=screen,
        trace_length_kpc=trace_length,
        transport_direction_x=direction_x,
        transport_direction_y=direction_y,
        mu_newtonian_proxy=mu_proxy,
        minimum_eigenvalue_proxy=minimum_eigenvalue,
        maximum_eigenvalue_proxy=mu_proxy,
        total_field=total_field,
    )
