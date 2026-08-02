"""Parameter-free three-dimensional baryonic planarity controller."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voidscreen.field_solvers import cell_coordinates


@dataclass(frozen=True)
class MassPlanarity:
    center_of_mass: tuple[float, float, float]
    covariance: np.ndarray
    eigenvalues: np.ndarray
    planarity: float


def _spacing_tuple(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        steps = (float(spacing),) * 3
    else:
        steps = tuple(float(value) for value in spacing)
    if len(steps) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("spacing must contain three finite positive values")
    return steps


def baryonic_mass_planarity(
    density: np.ndarray,
    spacing: float | Sequence[float],
) -> MassPlanarity:
    """Measure whether a mass distribution is a sheet rather than a filament or ball.

    Let lambda_1 <= lambda_2 <= lambda_3 be the eigenvalues of the mass-weighted
    spatial covariance tensor.  The dimensionless controller is

        P = (1 - lambda_1/lambda_2) (1 - lambda_1/lambda_3).

    P approaches one only when one direction is thin compared with both in-plane
    directions.  It approaches zero for spherical distributions and for filaments,
    whose two transverse eigenvalues are comparable.  No length scale, threshold,
    exponent, or object label enters the definition.
    """

    rho = np.asarray(density, dtype=float)
    if rho.ndim != 3 or min(rho.shape) < 3:
        raise ValueError("density must be a 3D grid with at least three cells per axis")
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("density must be finite and nonnegative")
    total = float(np.sum(rho))
    if total <= 0.0:
        raise ValueError("density must have positive mass")
    steps = _spacing_tuple(spacing)
    coordinates = cell_coordinates(rho.shape, steps)
    center = tuple(float(np.sum(rho * coordinate) / total) for coordinate in coordinates)
    covariance = np.empty((3, 3), dtype=float)
    centered = tuple(
        coordinate - offset
        for coordinate, offset in zip(coordinates, center, strict=True)
    )
    for row in range(3):
        for column in range(3):
            covariance[row, column] = float(
                np.sum(rho * centered[row] * centered[column]) / total
            )
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    scale = max(float(eigenvalues[-1]), np.finfo(float).tiny)
    if eigenvalues[1] <= np.finfo(float).eps * scale:
        planarity = 0.0
    else:
        planarity = (1.0 - eigenvalues[0] / eigenvalues[1]) * (
            1.0 - eigenvalues[0] / eigenvalues[2]
        )
    return MassPlanarity(
        center_of_mass=center,
        covariance=covariance,
        eigenvalues=eigenvalues,
        planarity=float(np.clip(planarity, 0.0, 1.0)),
    )


def planarity_blended_coherence(
    local_coherence: np.ndarray,
    planarity: float,
) -> np.ndarray:
    """Blend a local multi-center controller with the sheet-coherent endpoint.

    The equivalent source equation is

        S = P S_coh + (1-P)[C S_coh + (1-C) S_local]
          = [P + (1-P)C] S_coh + (1-P)(1-C) S_local.
    """

    coherence = np.asarray(local_coherence, dtype=float)
    if np.any(~np.isfinite(coherence)):
        raise ValueError("local_coherence must be finite")
    tolerance = 1e-12
    if np.min(coherence) < -tolerance or np.max(coherence) > 1.0 + tolerance:
        raise ValueError("local_coherence must lie in [0, 1]")
    weight = float(planarity)
    if not np.isfinite(weight) or weight < -tolerance or weight > 1.0 + tolerance:
        raise ValueError("planarity must lie in [0, 1]")
    coherence = np.clip(coherence, 0.0, 1.0)
    weight = float(np.clip(weight, 0.0, 1.0))
    return weight + (1.0 - weight) * coherence
