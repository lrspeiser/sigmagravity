"""RAR-coherent matter potential and a two-potential weak-field metric."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from voidscreen.coherent_monopole import CoherentMonopoleSolution
from voidscreen.field_solvers import acceleration_from_potential, cell_coordinates, laplacian
from voidscreen.spatial_qumond_3d import baryonic_center_of_mass


@dataclass(frozen=True)
class TwoPotentialMetric:
    time_potential: np.ndarray
    spatial_potential: np.ndarray
    weyl_potential: np.ndarray
    time_acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    spatial_acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    weyl_acceleration: tuple[np.ndarray, np.ndarray, np.ndarray]
    weyl_identity_relative_rms: float


def rar_acceleration(newtonian_acceleration, a0: float) -> np.ndarray:
    """Return the fixed RAR acceleration with its exponential Solar screening."""

    g_n = np.asarray(newtonian_acceleration, dtype=float)
    if np.any(~np.isfinite(g_n)) or np.any(g_n < 0.0):
        raise ValueError("newtonian_acceleration must be finite and nonnegative")
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("a0 must be finite and positive")
    root = np.sqrt(g_n / float(a0))
    denominator = -np.expm1(-root)
    return np.divide(g_n, denominator, out=np.zeros_like(g_n), where=denominator > 0.0)


def _spacing(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        steps = (float(spacing),) * 3
    else:
        steps = tuple(float(value) for value in spacing)
    if len(steps) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("spacing must contain three finite positive values")
    if not np.allclose(steps, steps[0], rtol=0.0, atol=1e-14 * steps[0]):
        raise ValueError("native radial shells require isotropic spacing")
    return steps


def rar_coherent_monopole_potential(
    density: np.ndarray,
    newtonian_potential: np.ndarray,
    newtonian_acceleration: Sequence[np.ndarray],
    spacing: float | Sequence[float],
    *,
    a0: float,
) -> CoherentMonopoleSolution:
    """Complete only the shared inward monopole with the fixed RAR relation."""

    rho = np.asarray(density, dtype=float)
    potential_n = np.asarray(newtonian_potential, dtype=float)
    acceleration_n = tuple(np.asarray(component, dtype=float) for component in newtonian_acceleration)
    if rho.ndim != 3 or min(rho.shape) < 5 or rho.shape != potential_n.shape:
        raise ValueError("density and potential must be matching 3D grids")
    if len(acceleration_n) != 3 or any(component.shape != rho.shape for component in acceleration_n):
        raise ValueError("newtonian acceleration must contain three matching grids")
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0) or float(np.sum(rho)) <= 0.0:
        raise ValueError("density must be finite, nonnegative, and have positive mass")
    if np.any(~np.isfinite(potential_n)) or any(np.any(~np.isfinite(component)) for component in acceleration_n):
        raise ValueError("Newtonian field must be finite")
    steps = _spacing(spacing)
    step = steps[0]
    center = baryonic_center_of_mass(rho, steps)
    coordinates = cell_coordinates(rho.shape, steps)
    displacement = tuple(coordinate - offset for coordinate, offset in zip(coordinates, center, strict=True))
    radius = np.sqrt(sum(component * component for component in displacement))
    safe_radius = np.maximum(radius, np.finfo(float).tiny)
    radial_unit = tuple(component / safe_radius for component in displacement)
    inward = -sum(component * direction for component, direction in zip(acceleration_n, radial_unit, strict=True))
    shell_index = np.rint(radius / step).astype(int)
    shell_count = int(np.max(shell_index)) + 1
    shell_radius = np.arange(shell_count, dtype=float) * step
    coherent_newtonian = np.full(shell_count, np.nan, dtype=float)
    for shell in range(shell_count):
        members = shell_index == shell
        if np.any(members):
            coherent_newtonian[shell] = max(float(np.mean(inward[members])), 0.0)
    coherent_newtonian[0] = 0.0
    finite = np.flatnonzero(np.isfinite(coherent_newtonian))
    missing = np.flatnonzero(~np.isfinite(coherent_newtonian))
    if finite.size < 2:
        raise ValueError("grid has too few occupied radial shells")
    if missing.size:
        coherent_newtonian[missing] = np.interp(missing, finite, coherent_newtonian[finite])
    completed = rar_acceleration(coherent_newtonian, float(a0))
    correction = completed - coherent_newtonian
    radial_correction_potential = cumulative_trapezoid(correction, shell_radius, initial=0.0)
    correction_potential = np.interp(radius, shell_radius, radial_correction_potential)
    potential = potential_n + correction_potential
    return CoherentMonopoleSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=laplacian(potential, steps),
        correction_potential=correction_potential,
        correction_acceleration=acceleration_from_potential(correction_potential, steps),
        center_of_mass=center,
        shell_radius=shell_radius,
        coherent_newtonian_acceleration=coherent_newtonian,
        coherent_completed_acceleration=completed,
        coherent_acceleration_correction=correction,
    )


def build_two_potential_metric(
    time_potential: np.ndarray,
    weyl_potential: np.ndarray,
    spacing: float | Sequence[float],
) -> TwoPotentialMetric:
    """Construct the spatial potential so the Weyl potential is exact.

    For ds^2=-(1+2 Psi/c^2)c^2dt^2+(1-2 Phi/c^2)dx^2, slow matter
    responds to Psi and light responds to W=(Psi+Phi)/2.  Given Psi and W,
    the spatial potential is uniquely Phi=2W-Psi.
    """

    time = np.asarray(time_potential, dtype=float)
    weyl = np.asarray(weyl_potential, dtype=float)
    if time.ndim != 3 or time.shape != weyl.shape or np.any(~np.isfinite(time)) or np.any(~np.isfinite(weyl)):
        raise ValueError("time and Weyl potentials must be finite matching 3D grids")
    steps = _spacing(spacing)
    spatial = 2.0 * weyl - time
    reconstructed = 0.5 * (time + spatial)
    scale = max(float(np.sqrt(np.mean(weyl**2))), np.finfo(float).tiny)
    identity = float(np.sqrt(np.mean((reconstructed - weyl) ** 2)) / scale)
    return TwoPotentialMetric(
        time_potential=time,
        spatial_potential=spatial,
        weyl_potential=weyl,
        time_acceleration=acceleration_from_potential(time, steps),
        spatial_acceleration=acceleration_from_potential(spatial, steps),
        weyl_acceleration=acceleration_from_potential(weyl, steps),
        weyl_identity_relative_rms=identity,
    )
