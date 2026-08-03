"""Sparse multiscale parameter packages for resolved baryonic galaxy maps.

The Haar basis is local in both position and scale.  Unlike a global radial
Fourier model, it can encode bars, arms, clumps, cavities, mosaic boundaries,
and lopsided outskirts without using a velocity or gravity target.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import rfc8785


Array = np.ndarray
SCHEMA_VERSION = "1.0.0"
GENERATOR_NAME = "sparse-orthonormal-haar-2d"


def _regular_axis(axis_kpc: Array) -> tuple[Array, float]:
    axis = np.asarray(axis_kpc, dtype=float)
    if axis.ndim != 1 or axis.size < 9 or np.any(~np.isfinite(axis)):
        raise ValueError("axis_kpc must be a finite one-dimensional array")
    spacing = np.diff(axis)
    if np.any(spacing <= 0.0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
        raise ValueError("axis_kpc must be strictly increasing and regular")
    return axis, float(spacing[0])


def _surface(values: Array, cells: int) -> Array:
    surface = np.asarray(values, dtype=float)
    if surface.shape != (cells, cells):
        raise ValueError("surface must match the square axis")
    if np.any(~np.isfinite(surface)) or np.any(surface < 0.0) or np.sum(surface) <= 0.0:
        raise ValueError("surface must be finite, non-negative, and non-empty")
    return surface


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def haar2_forward(values: Array) -> Array:
    """Return the orthonormal 2D Haar transform of a power-of-two square."""

    transformed = np.asarray(values, dtype=float).copy()
    if transformed.ndim != 2 or transformed.shape[0] != transformed.shape[1]:
        raise ValueError("Haar input must be square")
    cells = transformed.shape[0]
    if cells < 2 or cells & (cells - 1):
        raise ValueError("Haar input size must be a power of two")
    root_two = np.sqrt(2.0)
    active = cells
    while active > 1:
        source = transformed[:active, :active].copy()
        half = active // 2
        rows = np.empty_like(source)
        rows[:half, :] = (source[0::2, :] + source[1::2, :]) / root_two
        rows[half:, :] = (source[0::2, :] - source[1::2, :]) / root_two
        columns = np.empty_like(rows)
        columns[:, :half] = (rows[:, 0::2] + rows[:, 1::2]) / root_two
        columns[:, half:] = (rows[:, 0::2] - rows[:, 1::2]) / root_two
        transformed[:active, :active] = columns
        active //= 2
    return transformed


def haar2_inverse(coefficients: Array) -> Array:
    """Invert :func:`haar2_forward`."""

    restored = np.asarray(coefficients, dtype=float).copy()
    if restored.ndim != 2 or restored.shape[0] != restored.shape[1]:
        raise ValueError("Haar coefficients must be square")
    cells = restored.shape[0]
    if cells < 2 or cells & (cells - 1):
        raise ValueError("Haar coefficient size must be a power of two")
    root_two = np.sqrt(2.0)
    active = 2
    while active <= cells:
        source = restored[:active, :active].copy()
        half = active // 2
        columns = np.empty_like(source)
        columns[:, 0::2] = (source[:, :half] + source[:, half:]) / root_two
        columns[:, 1::2] = (source[:, :half] - source[:, half:]) / root_two
        rows = np.empty_like(columns)
        rows[0::2, :] = (columns[:half, :] + columns[half:, :]) / root_two
        rows[1::2, :] = (columns[:half, :] - columns[half:, :]) / root_two
        restored[:active, :active] = rows
        active *= 2
    return restored


def _normalize_mass(surface: Array, target_mass: float, spacing_kpc: float) -> Array:
    clipped = np.clip(np.asarray(surface, dtype=float), 0.0, None)
    actual = float(np.sum(clipped) * spacing_kpc**2)
    if actual <= 0.0:
        raise ValueError("generated surface has no positive mass")
    return clipped * (float(target_mass) / actual)


def extract_component_parameters(
    surface_density: Array,
    axis_kpc: Array,
    *,
    coefficient_count: int,
) -> dict[str, Any]:
    axis, spacing = _regular_axis(axis_kpc)
    surface = _surface(surface_density, axis.size)
    padded_cells = _next_power_of_two(axis.size)
    if coefficient_count < 1 or coefficient_count > padded_cells**2:
        raise ValueError("coefficient_count is outside the padded map size")
    padded = np.zeros((padded_cells, padded_cells), dtype=float)
    padded[: axis.size, : axis.size] = surface
    coefficients = haar2_forward(padded)
    magnitudes = np.abs(coefficients).ravel()
    indices = np.argpartition(magnitudes, -coefficient_count)[-coefficient_count:]
    # Sorting by flat index makes serialization and hashes deterministic.
    indices = np.sort(indices)
    return {
        "mass_solar": float(np.sum(surface) * spacing**2),
        "source_cells": int(axis.size),
        "padded_cells": int(padded_cells),
        "coefficient_count": int(coefficient_count),
        "coefficient_flat_indices": indices.astype(int).tolist(),
        "coefficient_values": coefficients.ravel()[indices].tolist(),
        "negative_reconstruction_policy": "clip_to_zero_then_restore_component_mass",
        "padding_policy": "zero_pad_upper_index_edges_to_next_power_of_two",
    }


def render_component(parameters: Mapping[str, Any], axis_kpc: Array) -> Array:
    axis, spacing = _regular_axis(axis_kpc)
    source_cells = int(parameters["source_cells"])
    padded_cells = int(parameters["padded_cells"])
    if axis.size != source_cells or padded_cells < source_cells or padded_cells & (padded_cells - 1):
        raise ValueError("axis or padded size does not match the parameter package")
    indices = np.asarray(parameters["coefficient_flat_indices"], dtype=int)
    values = np.asarray(parameters["coefficient_values"], dtype=float)
    if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
        raise ValueError("coefficient indices and values are inconsistent")
    coefficients = np.zeros(padded_cells**2, dtype=float)
    coefficients[indices] = values
    restored = haar2_inverse(coefficients.reshape((padded_cells, padded_cells)))
    cropped = restored[:source_cells, :source_cells]
    return _normalize_mass(cropped, float(parameters["mass_solar"]), spacing)


def package_content_hash(package: Mapping[str, Any]) -> str:
    payload = dict(package)
    payload.pop("contentSha256", None)
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


def extract_galaxy_parameters(
    galaxy: str,
    axis_kpc: Array,
    gas_surface_density: Array,
    stellar_surface_density: Array,
    *,
    coefficient_count_per_component: int,
    source_observables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    axis, spacing = _regular_axis(axis_kpc)
    gas = _surface(gas_surface_density, axis.size)
    stars = _surface(stellar_surface_density, axis.size)
    package: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "galaxy": str(galaxy),
        "sourceObservables": dict(source_observables or {}),
        "grid": {
            "cellsPerAxis": int(axis.size),
            "minimumKpc": float(axis[0]),
            "maximumKpc": float(axis[-1]),
            "spacingKpc": spacing,
        },
        "extractionControls": {
            "basis": "orthonormal_Haar_2D",
            "coefficientCountPerComponent": int(coefficient_count_per_component),
            "selection": "largest_absolute_coefficients_with_deterministic_index_order",
        },
        "components": {
            "gas": extract_component_parameters(
                gas, axis, coefficient_count=coefficient_count_per_component
            ),
            "stars": extract_component_parameters(
                stars, axis, coefficient_count=coefficient_count_per_component
            ),
        },
        "gravityParameters": {},
        "velocityTargetsUsed": False,
        "verticalStructure": {
            "status": "assumed_prior_not_measured",
            "warning": "A surface-density map does not identify a unique 3D density.",
        },
    }
    package["contentSha256"] = package_content_hash(package)
    return package


def render_galaxy(package: Mapping[str, Any], axis_kpc: Array | None = None) -> dict[str, Array]:
    if axis_kpc is None:
        grid = package["grid"]
        axis_kpc = np.linspace(
            float(grid["minimumKpc"]), float(grid["maximumKpc"]), int(grid["cellsPerAxis"])
        )
    gas = render_component(package["components"]["gas"], axis_kpc)
    stars = render_component(package["components"]["stars"], axis_kpc)
    return {"gas": gas, "stars": stars, "total": gas + stars}

