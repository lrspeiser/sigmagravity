"""Baryon-centered radial path potentials and zero-boundary routing corrections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from voidscreen.field_solvers import (
    acceleration_from_potential,
    cell_coordinates,
    laplacian,
    simple_mond_acceleration,
)
from voidscreen.spatial_qumond_3d import baryonic_center_of_mass


@dataclass(frozen=True)
class RadialPathPotentialSolution:
    potential: np.ndarray
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    equation_source: np.ndarray
    center_of_mass: tuple[float, float, float]
    quadrature_order: int
    interpolation_order: int


@dataclass(frozen=True)
class HybridPathRoutingSolution:
    potential: np.ndarray
    acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    equation_source: np.ndarray
    correction_potential: np.ndarray
    routing_fraction: float


def _spacing3(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        values = (float(spacing),) * 3
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("spacing must contain three positive values")
    return values


def _grid(values, *, name: str, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 5 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3D grid with at least five cells per axis")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def radial_path_potential_from_newtonian(
    density: np.ndarray,
    newtonian_potential: np.ndarray,
    newtonian_acceleration: Sequence[np.ndarray],
    spacing: float | Sequence[float],
    *,
    a0: float,
    quadrature_order: int = 24,
    interpolation_order: int = 1,
) -> RadialPathPotentialSolution:
    """Integrate the algebraically boosted Newtonian force along centroid rays."""

    rho = _grid(density, name="density", nonnegative=True)
    potential_n = _grid(newtonian_potential, name="newtonian_potential")
    if rho.shape != potential_n.shape:
        raise ValueError("density and newtonian_potential must have the same shape")
    if len(newtonian_acceleration) != 3:
        raise ValueError("newtonian_acceleration must contain three components")
    acceleration_n = tuple(
        _grid(component, name=f"newtonian_acceleration[{index}]")
        for index, component in enumerate(newtonian_acceleration)
    )
    if any(component.shape != rho.shape for component in acceleration_n):
        raise ValueError("newtonian acceleration components must match density")
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    order = int(quadrature_order)
    if order < 4:
        raise ValueError("quadrature_order must be at least four")
    interpolation = int(interpolation_order)
    if interpolation not in (1, 3):
        raise ValueError("interpolation_order must be one or three")
    steps = _spacing3(spacing)
    center = baryonic_center_of_mass(rho, steps)
    coordinates = cell_coordinates(rho.shape, steps)
    physical_displacement = tuple(
        coordinate - offset for coordinate, offset in zip(coordinates, center, strict=True)
    )
    index_axes = np.meshgrid(
        *[np.arange(count, dtype=float) for count in rho.shape],
        indexing="ij",
    )
    center_index = tuple(
        float(offset) / step + (count - 1.0) / 2.0
        for offset, step, count in zip(center, steps, rho.shape, strict=True)
    )
    center_coordinates = np.asarray(center_index, dtype=float)[:, None]
    center_potential = float(
        map_coordinates(
            potential_n,
            center_coordinates,
            order=interpolation,
            mode="nearest",
            prefilter=interpolation > 1,
        )[0]
    )
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    path_integral = np.zeros_like(rho)
    for node, weight in zip(nodes, weights, strict=True):
        sample_coordinates = [
            center_axis + float(node) * (index_grid - center_axis)
            for index_grid, center_axis in zip(index_axes, center_index, strict=True)
        ]
        sampled_acceleration = tuple(
            map_coordinates(
                component,
                sample_coordinates,
                order=interpolation,
                mode="nearest",
                prefilter=interpolation > 1,
            )
            for component in acceleration_n
        )
        magnitude = np.sqrt(sum(component * component for component in sampled_acceleration))
        algebraic_magnitude = simple_mond_acceleration(magnitude, float(a0))
        boost = np.divide(
            algebraic_magnitude,
            magnitude,
            out=np.zeros_like(magnitude),
            where=magnitude > 0.0,
        )
        force_dot_displacement = sum(
            component * displacement
            for component, displacement in zip(
                sampled_acceleration,
                physical_displacement,
                strict=True,
            )
        )
        path_integral += float(weight) * (-boost * force_dot_displacement)
    potential = center_potential + path_integral
    acceleration = acceleration_from_potential(potential, steps)
    source = laplacian(potential, steps)
    return RadialPathPotentialSolution(
        potential=potential,
        acceleration=acceleration,
        equation_source=source,
        center_of_mass=center,
        quadrature_order=order,
        interpolation_order=interpolation,
    )


def hybrid_path_routing_potential(
    path_solution: RadialPathPotentialSolution,
    local_potential: np.ndarray,
    routed_potential: np.ndarray,
    spacing: float | Sequence[float],
    routing_fraction: float,
) -> HybridPathRoutingSolution:
    """Add a declared fraction of the routed-minus-local potential difference."""

    local = _grid(local_potential, name="local_potential")
    routed = _grid(routed_potential, name="routed_potential")
    if local.shape != path_solution.potential.shape or routed.shape != local.shape:
        raise ValueError("path, local, and routed potentials must share one grid")
    fraction = float(routing_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("routing_fraction must be finite and lie in [0, 1]")
    steps = _spacing3(spacing)
    correction = routed - local
    potential = path_solution.potential + fraction * correction
    return HybridPathRoutingSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=laplacian(potential, steps),
        correction_potential=correction,
        routing_fraction=fraction,
    )


def normalized_acceleration_curl(
    acceleration: Sequence[np.ndarray],
    spacing: float | Sequence[float],
) -> float:
    """Return RMS curl divided by RMS divergence on a trimmed interior."""

    if len(acceleration) != 3:
        raise ValueError("acceleration must contain three components")
    fields = tuple(
        _grid(component, name=f"acceleration[{index}]")
        for index, component in enumerate(acceleration)
    )
    if len({field.shape for field in fields}) != 1:
        raise ValueError("acceleration components must share one shape")
    steps = _spacing3(spacing)
    derivatives = [np.gradient(field, *steps, edge_order=2) for field in fields]
    curl = (
        derivatives[2][1] - derivatives[1][2],
        derivatives[0][2] - derivatives[2][0],
        derivatives[1][0] - derivatives[0][1],
    )
    divergence = derivatives[0][0] + derivatives[1][1] + derivatives[2][2]
    border = max(int(0.1 * min(fields[0].shape)), 2)
    interior = (slice(border, -border),) * 3
    curl_rms = float(
        np.sqrt(np.mean(sum(component[interior] ** 2 for component in curl)))
    )
    divergence_rms = float(np.sqrt(np.mean(divergence[interior] ** 2)))
    return curl_rms / max(divergence_rms, np.finfo(float).tiny)
