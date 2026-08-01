"""Direction reconstruction from registered projected baryonic proxy maps."""

from __future__ import annotations

import numpy as np


def unit_directions(vectors, *, fallback=None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or np.any(~np.isfinite(vectors)):
        raise ValueError("vectors must be a finite Nx2 array")
    norm = np.linalg.norm(vectors, axis=1)
    missing = norm <= np.finfo(float).tiny
    if np.any(missing):
        if fallback is None:
            raise ValueError("zero direction without a fallback")
        fallback = np.asarray(fallback, dtype=float)
        if fallback.shape != vectors.shape:
            raise ValueError("fallback must match vectors")
        vectors = vectors.copy()
        vectors[missing] = fallback[missing]
        norm = np.linalg.norm(vectors, axis=1)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError("fallback also contains a zero direction")
    return vectors / norm[:, None]


def map_attraction_directions(
    axis,
    surface,
    positions,
    *,
    softening: float,
    distance_power: float = 2.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return unit directions toward a softened positive projected map."""
    axis = np.asarray(axis, dtype=float)
    image = np.asarray(surface, dtype=float)
    xy = np.asarray(positions, dtype=float)
    if axis.ndim != 1 or len(axis) < 8 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("axis must be a strictly increasing vector")
    if image.shape != (len(axis), len(axis)):
        raise ValueError("surface shape must match the square axis")
    if xy.ndim != 2 or xy.shape[1] != 2 or np.any(~np.isfinite(xy)):
        raise ValueError("positions must be a finite Nx2 array")
    if softening <= 0.0 or distance_power <= 0.0:
        raise ValueError("softening and distance_power must be positive")
    image = np.where(np.isfinite(image) & (image > 0.0), image, 0.0)
    total = float(np.sum(image))
    if total <= 0.0:
        raise ValueError("surface has no positive weight")
    image /= total
    grid_x, grid_y = np.meshgrid(axis, axis)
    cells = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    weights = image.ravel()
    keep = weights > 0.0
    cells, weights = cells[keep], weights[keep]
    centroid = np.sum(cells * weights[:, None], axis=0)
    vectors = []
    for position in xy:
        displacement = cells - position[None, :]
        distance2 = np.sum(np.square(displacement), axis=1)
        inverse = np.power(
            distance2 + float(softening) ** 2,
            -0.5 * (float(distance_power) + 1.0),
        )
        vectors.append(np.sum(displacement * (weights * inverse)[:, None], axis=0))
    vectors = np.asarray(vectors)
    fallback = centroid[None, :] - xy
    fallback_norm = np.linalg.norm(fallback, axis=1)
    fallback[fallback_norm <= np.finfo(float).tiny] = np.array([1.0, 0.0])
    directions = unit_directions(vectors, fallback=fallback)
    return directions, {
        "map_centroid": centroid,
        "positive_cells": int(len(weights)),
        "normalization_error": abs(float(np.sum(image)) - 1.0),
        "minimum_raw_vector_norm": float(np.min(np.linalg.norm(vectors, axis=1))),
    }


def blend_unit_directions(first, second, fraction: float) -> np.ndarray:
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    first = unit_directions(first)
    second = unit_directions(second)
    mixed = (1.0 - float(fraction)) * first + float(fraction) * second
    fallback = first + 1.0e-12 * second
    return unit_directions(mixed, fallback=fallback)
