"""Parameter-free alignment of a field with its barycentric inward direction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voidscreen.field_solvers import cell_coordinates
from voidscreen.spatial_qumond_3d import baryonic_center_of_mass


@dataclass(frozen=True)
class BarycentricRadialAlignmentSolution:
    alignment: np.ndarray
    inward_radial_acceleration: np.ndarray
    acceleration_magnitude: np.ndarray
    center_of_mass: tuple[float, float, float]


def _grid(values, *, name: str, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 5 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3D grid with at least five cells per axis")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _spacing3(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        steps = (float(spacing),) * 3
    else:
        steps = tuple(float(value) for value in spacing)
    if len(steps) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("spacing must contain three finite positive values")
    return steps


def vector_radial_alignment(
    displacement: Sequence[np.ndarray],
    acceleration: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inward alignment, inward radial component, and vector magnitude."""

    if len(displacement) != 3 or len(acceleration) != 3:
        raise ValueError("displacement and acceleration must each contain three components")
    offsets = tuple(
        _grid(component, name=f"displacement[{index}]")
        for index, component in enumerate(displacement)
    )
    fields = tuple(
        _grid(component, name=f"acceleration[{index}]")
        for index, component in enumerate(acceleration)
    )
    shapes = {component.shape for component in (*offsets, *fields)}
    if len(shapes) != 1:
        raise ValueError("all displacement and acceleration components must share one grid")
    radius = np.sqrt(sum(component * component for component in offsets))
    magnitude = np.sqrt(sum(component * component for component in fields))
    active = (radius > 0.0) & (magnitude > 0.0)
    radial_unit = tuple(
        np.divide(
            component,
            radius,
            out=np.zeros_like(component),
            where=radius > 0.0,
        )
        for component in offsets
    )
    inward = -sum(
        component * direction
        for component, direction in zip(fields, radial_unit, strict=True)
    )
    alignment = np.divide(
        np.maximum(inward, 0.0),
        magnitude,
        out=np.zeros_like(magnitude),
        where=active,
    )
    return np.clip(alignment, 0.0, 1.0), inward, magnitude


def barycentric_radial_alignment(
    density: np.ndarray,
    newtonian_acceleration: Sequence[np.ndarray],
    spacing: float | Sequence[float],
) -> BarycentricRadialAlignmentSolution:
    """Measure whether the summed Newtonian field points toward the barycenter."""

    rho = _grid(density, name="density", nonnegative=True)
    if float(np.sum(rho)) <= 0.0:
        raise ValueError("density must have positive mass")
    if len(newtonian_acceleration) != 3:
        raise ValueError("newtonian_acceleration must contain three components")
    acceleration = tuple(
        _grid(component, name=f"newtonian_acceleration[{index}]")
        for index, component in enumerate(newtonian_acceleration)
    )
    if any(component.shape != rho.shape for component in acceleration):
        raise ValueError("newtonian acceleration components must match density")
    steps = _spacing3(spacing)
    center = baryonic_center_of_mass(rho, steps)
    coordinates = cell_coordinates(rho.shape, steps)
    displacement = tuple(
        coordinate - offset
        for coordinate, offset in zip(coordinates, center, strict=True)
    )
    alignment, inward, magnitude = vector_radial_alignment(
        displacement,
        acceleration,
    )
    return BarycentricRadialAlignmentSolution(
        alignment=alignment,
        inward_radial_acceleration=inward,
        acceleration_magnitude=magnitude,
        center_of_mass=center,
    )
