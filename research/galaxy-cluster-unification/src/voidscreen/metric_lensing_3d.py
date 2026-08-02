"""Three-dimensional tensor activation and zero-slip photon deflection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.field_solvers import (
    FieldSolution,
    acceleration_magnitude,
    solve_newtonian,
    surface_density_to_volume,
)
from voidscreen.tensor_aqual import simple_mu

KPC_M = 3.085677581491367e19
M_SUN_KG = 1.98847e30
C_M_S = 299792458.0
RAD_TO_ARCSEC = 206264.80624709636


@dataclass(frozen=True)
class TensorActivation3D:
    sigma: np.ndarray
    transverse_mismatch: np.ndarray
    survival: np.ndarray
    high_acceleration_screen: np.ndarray
    trace_length: np.ndarray
    transport_direction: tuple[np.ndarray, np.ndarray, np.ndarray]
    mu_newtonian_proxy: np.ndarray
    minimum_eigenvalue_proxy: np.ndarray
    stellar_field: FieldSolution
    gas_field: FieldSolution
    total_field: FieldSolution


@dataclass(frozen=True)
class PhotonDeflection2D:
    alpha_x_radian: np.ndarray
    alpha_y_radian: np.ndarray
    alpha_x_arcsec: np.ndarray
    alpha_y_arcsec: np.ndarray
    distance_ratio: float
    zero_slip_multiplier: float


def projected_rms_radius(surface_density: np.ndarray, cell_size: float) -> float:
    surface = np.asarray(surface_density, dtype=float)
    if surface.ndim != 2 or min(surface.shape) < 5 or np.any(surface < 0.0):
        raise ValueError("surface_density must be a nonnegative 2D map")
    if not np.all(np.isfinite(surface)) or cell_size <= 0.0 or float(np.sum(surface)) <= 0.0:
        raise ValueError("surface density and cell size are invalid")
    axes = [
        (np.arange(count, dtype=float) - (count - 1.0) / 2.0) * float(cell_size)
        for count in surface.shape
    ]
    x, y = np.meshgrid(*axes, indexing="ij")
    return float(np.sqrt(np.sum(surface * (x * x + y * y)) / np.sum(surface)))


def lift_surface_density_msun_kpc2_to_si_volume(
    surface_density_msun_kpc2: np.ndarray,
    z_coordinates_kpc: np.ndarray,
    *,
    scale_height_kpc: float | None = None,
    cell_kpc: float,
) -> tuple[np.ndarray, float]:
    surface = np.asarray(surface_density_msun_kpc2, dtype=float)
    z_kpc = np.asarray(z_coordinates_kpc, dtype=float)
    scale_kpc = (
        projected_rms_radius(surface, cell_kpc)
        if scale_height_kpc is None
        else float(scale_height_kpc)
    )
    surface_si = surface * M_SUN_KG / KPC_M**2
    volume_si = surface_density_to_volume(
        surface_si,
        z_kpc * KPC_M,
        scale_height=scale_kpc * KPC_M,
    )
    return volume_si, scale_kpc


def _unit_direction(acceleration: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    magnitude = acceleration_magnitude(acceleration)
    floor = max(float(np.max(magnitude)) * 1e-12, np.finfo(float).tiny)
    active = magnitude > floor
    result = []
    for axis, component in enumerate(acceleration):
        default = 1.0 if axis == 0 else 0.0
        result.append(np.where(active, component / np.maximum(magnitude, floor), default))
    return tuple(result)


def forward_boundary_distance_3d(
    direction: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: float,
) -> np.ndarray:
    if len(direction) != 3 or spacing <= 0.0:
        raise ValueError("direction must have three components and spacing be positive")
    shape = np.asarray(direction[0]).shape
    if any(np.asarray(component).shape != shape for component in direction):
        raise ValueError("direction components must have matching shapes")
    coordinates = np.indices(shape, dtype=float)
    maximum = np.asarray(shape, dtype=float) - 1.0
    distances = []
    for axis, component in enumerate(direction):
        values = np.asarray(component, dtype=float)
        distance = np.full(shape, np.inf)
        positive = values > 1e-15
        negative = values < -1e-15
        distance[positive] = (
            maximum[axis] - coordinates[axis][positive]
        ) * float(spacing) / values[positive]
        distance[negative] = coordinates[axis][negative] * float(spacing) / (-values[negative])
        distances.append(distance)
    result = np.minimum(np.minimum(distances[0], distances[1]), distances[2])
    diagonal = float(np.linalg.norm(maximum) * float(spacing))
    return np.where(np.isfinite(result), np.maximum(result, 0.0), diagonal)


def physical_tidal_length_3d(field: FieldSolution, spacing: float) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    gradient_norm_squared = np.zeros_like(field.potential)
    for component in field.acceleration:
        for derivative in np.gradient(component, float(spacing), edge_order=2):
            gradient_norm_squared += derivative * derivative
    gradient_norm = np.sqrt(gradient_norm_squared)
    numerical_floor = max(float(np.max(gradient_norm)) * 1e-12, np.finfo(float).tiny)
    raw_length = acceleration_magnitude(field.acceleration) / np.maximum(
        gradient_norm,
        numerical_floor,
    )
    direction = _unit_direction(field.acceleration)
    boundary = forward_boundary_distance_3d(direction, spacing)
    return np.minimum(np.maximum(raw_length, 0.0), boundary)


def component_transverse_mismatch_3d(
    first: FieldSolution,
    second: FieldSolution,
) -> np.ndarray:
    first_magnitude = acceleration_magnitude(first.acceleration)
    second_magnitude = acceleration_magnitude(second.acceleration)
    denominator = first_magnitude + second_magnitude
    floor = max(float(np.max(denominator)) * 1e-12, np.finfo(float).tiny)
    active = (
        (denominator > floor)
        & (first_magnitude > floor)
        & (second_magnitude > floor)
    )
    dot = np.zeros_like(denominator)
    dot[active] = sum(
        left[active] * right[active]
        for left, right in zip(first.acceleration, second.acceleration, strict=True)
    ) / (first_magnitude[active] * second_magnitude[active])
    dot = np.clip(dot, -1.0, 1.0)
    mixing = np.zeros_like(denominator)
    mixing[active] = (
        2.0
        * np.sqrt(first_magnitude[active] * second_magnitude[active])
        / denominator[active]
    )
    mismatch = mixing * np.maximum(1.0 - dot * dot, 0.0)
    return np.clip(mismatch, 0.0, 1.0)


def _component_difference_direction_3d(
    stars: FieldSolution,
    gas: FieldSolution,
    total: FieldSolution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    star_unit = _unit_direction(stars.acceleration)
    gas_unit = _unit_direction(gas.acceleration)
    fallback = _unit_direction(total.acceleration)
    difference = tuple(
        gas_component - star_component
        for star_component, gas_component in zip(star_unit, gas_unit, strict=True)
    )
    norm = np.sqrt(sum(component * component for component in difference))
    active = norm > 1e-12
    result = []
    for component, fallback_component in zip(difference, fallback, strict=True):
        result.append(
            np.where(active, component / np.maximum(norm, 1e-12), fallback_component)
        )
    final_norm = np.sqrt(sum(component * component for component in result))
    valid = final_norm > 1e-12
    return tuple(
        np.where(
            valid,
            component / np.maximum(final_norm, 1e-12),
            1.0 if axis == 0 else 0.0,
        )
        for axis, component in enumerate(result)
    )


def exact_tensor_activation_3d(
    stellar_density: np.ndarray,
    gas_density: np.ndarray,
    spacing: float,
    *,
    gravitational_constant: float = 6.67430e-11,
    a0: float = 1.2e-10,
    coherence_length: float,
    coherence_power: float = 2.0,
    mu_floor: float = 1e-6,
) -> TensorActivation3D:
    stars = np.maximum(np.asarray(stellar_density, dtype=float), 0.0)
    gas = np.maximum(np.asarray(gas_density, dtype=float), 0.0)
    if stars.ndim != 3 or stars.shape != gas.shape or min(stars.shape) < 5:
        raise ValueError("stellar and gas densities must be matching 3D grids")
    if spacing <= 0.0 or a0 <= 0.0 or coherence_length <= 0.0 or coherence_power <= 0.0:
        raise ValueError("activation scales must be positive")
    stellar_field = solve_newtonian(
        stars,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    gas_field = solve_newtonian(
        gas,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    total_field = solve_newtonian(
        stars + gas,
        spacing,
        gravitational_constant=gravitational_constant,
    )
    mismatch = component_transverse_mismatch_3d(stellar_field, gas_field)
    trace_length = physical_tidal_length_3d(total_field, spacing)
    survival = 1.0 - np.exp(
        -np.power(
            np.maximum(trace_length / float(coherence_length), 0.0),
            float(coherence_power),
        )
    )
    magnitude = acceleration_magnitude(total_field.acceleration)
    screen = float(a0) / (float(a0) + magnitude)
    sigma = np.clip(screen * mismatch * survival, 0.0, 1.0)
    direction = _component_difference_direction_3d(stellar_field, gas_field, total_field)
    mu_proxy = np.maximum(simple_mu(magnitude / float(a0)), float(mu_floor))
    return TensorActivation3D(
        sigma=sigma,
        transverse_mismatch=mismatch,
        survival=survival,
        high_acceleration_screen=screen,
        trace_length=trace_length,
        transport_direction=direction,
        mu_newtonian_proxy=mu_proxy,
        minimum_eigenvalue_proxy=mu_proxy * (1.0 - sigma),
        stellar_field=stellar_field,
        gas_field=gas_field,
        total_field=total_field,
    )


def constitutive_tensor_components_3d(
    sigma: np.ndarray,
    direction: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, ...]:
    anisotropy = np.asarray(sigma, dtype=float)
    h_x, h_y, h_z = (np.asarray(component, dtype=float) for component in direction)
    return (
        1.0 - anisotropy * h_x * h_x,
        1.0 - anisotropy * h_y * h_y,
        1.0 - anisotropy * h_z * h_z,
        -anisotropy * h_x * h_y,
        -anisotropy * h_x * h_z,
        -anisotropy * h_y * h_z,
    )


def photon_deflection_zero_slip(
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray],
    dz: float,
    *,
    distance_ratio: float = 1.0,
    light_speed: float = C_M_S,
) -> PhotonDeflection2D:
    if len(acceleration) != 3 or dz <= 0.0 or light_speed <= 0.0 or distance_ratio <= 0.0:
        raise ValueError("photon-deflection inputs are invalid")
    if any(np.asarray(component).ndim != 3 for component in acceleration):
        raise ValueError("acceleration components must be 3D")
    multiplier = 2.0 * float(distance_ratio) / float(light_speed) ** 2
    alpha_x = -multiplier * np.trapezoid(acceleration[0], dx=float(dz), axis=2)
    alpha_y = -multiplier * np.trapezoid(acceleration[1], dx=float(dz), axis=2)
    return PhotonDeflection2D(
        alpha_x_radian=alpha_x,
        alpha_y_radian=alpha_y,
        alpha_x_arcsec=alpha_x * RAD_TO_ARCSEC,
        alpha_y_arcsec=alpha_y * RAD_TO_ARCSEC,
        distance_ratio=float(distance_ratio),
        zero_slip_multiplier=multiplier,
    )


def normalized_deflection_curl(
    alpha_x: np.ndarray,
    alpha_y: np.ndarray,
    spacing: float,
) -> float:
    curl = np.gradient(alpha_y, spacing, axis=0, edge_order=2) - np.gradient(
        alpha_x,
        spacing,
        axis=1,
        edge_order=2,
    )
    divergence = np.gradient(alpha_x, spacing, axis=0, edge_order=2) + np.gradient(
        alpha_y,
        spacing,
        axis=1,
        edge_order=2,
    )
    border = max(int(0.1 * min(alpha_x.shape)), 1)
    interior = (slice(border, -border), slice(border, -border))
    return float(np.sqrt(np.mean(curl[interior] ** 2))) / max(
        float(np.sqrt(np.mean(divergence[interior] ** 2))),
        np.finfo(float).tiny,
    )
