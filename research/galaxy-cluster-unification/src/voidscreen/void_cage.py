from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import KPC_M, PackedDataset


@dataclass(frozen=True)
class CageKernel:
    name: str
    family: str
    parameter: float


def kernel_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def power_law_hessian(
    offsets: np.ndarray,
    charges: np.ndarray,
    *,
    force_power: float,
) -> np.ndarray:
    """Return the unit potential Hessian for a repulsive 1/r**p force.

    The acceleration from one positive void charge is outward with magnitude
    proportional to ``1/r**force_power``. Relative acceleration is ``-H r``.
    Consequently, a positive eigenvalue of H is a compressive direction.
    """
    if force_power <= 0.0:
        raise ValueError("force_power must be positive")
    vectors = np.asarray(offsets, dtype=np.float64)
    weights = np.asarray(charges, dtype=np.float64)
    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii <= 0.0):
        raise ValueError("power-law sources must be separated from the field point")
    unit = vectors / radii[:, None]
    radial_weight = weights / np.power(radii, force_power + 1.0)
    outer = np.einsum("ni,nj->nij", unit, unit)
    identity = np.eye(3, dtype=np.float64)
    return np.einsum(
        "n,nij->ij",
        radial_weight,
        (force_power + 1.0) * outer - identity,
    )


def yukawa_hessian(
    offsets: np.ndarray,
    charges: np.ndarray,
    *,
    range_hmpc: float,
) -> np.ndarray:
    """Return the unit Hessian of exp(-r/lambda)/r from exterior void charges."""
    if range_hmpc <= 0.0:
        raise ValueError("range_hmpc must be positive")
    vectors = np.asarray(offsets, dtype=np.float64)
    weights = np.asarray(charges, dtype=np.float64)
    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii <= 0.0):
        raise ValueError("Yukawa sources must be separated from the field point")
    unit = vectors / radii[:, None]
    outer = np.einsum("ni,nj->nij", unit, unit)
    exponential = np.exp(-radii / range_hmpc)
    inverse = 1.0 / radii
    isotropic = -exponential * (inverse**3 + inverse**2 / range_hmpc)
    directional = exponential * (
        3.0 * inverse**3
        + 3.0 * inverse**2 / range_hmpc
        + inverse / range_hmpc**2
    )
    identity = np.eye(3, dtype=np.float64)
    return np.einsum("n,nij->ij", weights * directional, outer) + identity * float(
        np.sum(weights * isotropic)
    )


def tensor_metrics(tensor: np.ndarray) -> dict[str, float | bool | int]:
    values = np.linalg.eigvalsh(np.asarray(tensor, dtype=np.float64))
    trace = float(np.sum(values))
    kappa = trace / 3.0
    deviator = np.asarray(tensor, dtype=np.float64) - np.eye(3) * kappa
    denominator = math.sqrt(3.0) * abs(kappa)
    anisotropy = float(np.linalg.norm(deviator) / denominator) if denominator > 0.0 else math.inf
    return {
        "kappa_unit": kappa,
        "eigen_min": float(values[0]),
        "eigen_mid": float(values[1]),
        "eigen_max": float(values[2]),
        "anisotropy": anisotropy,
        "compressive_directions": int(np.sum(values > 0.0)),
        "fully_compressive": bool(np.all(values > 0.0)),
    }


def _axis_centers(side: int, box_size_hmpc: float) -> np.ndarray:
    voxel = box_size_hmpc / side
    return -box_size_hmpc / 2.0 + (np.arange(side, dtype=np.float64) + 0.5) * voxel


def _shell_sources(
    density_contrast: np.ndarray,
    point_hmpc: np.ndarray,
    *,
    box_size_hmpc: float,
    inner_hmpc: float,
    outer_hmpc: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(density_contrast, dtype=np.float64)
    if delta.ndim != 3 or len(set(delta.shape)) != 1:
        raise ValueError(f"Expected cubic density grid, found {delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError("Density grid contains non-finite values")
    if not 0.0 < inner_hmpc < outer_hmpc:
        raise ValueError("Require 0 < inner_hmpc < outer_hmpc")
    side = delta.shape[0]
    centers = _axis_centers(side, box_size_hmpc)
    selections = [
        np.flatnonzero(np.abs(centers - float(coordinate)) <= outer_hmpc)
        for coordinate in point_hmpc
    ]
    if any(indices.size == 0 for indices in selections):
        raise ValueError("Field point has no grid cells within the declared shell")
    ix, iy, iz = np.meshgrid(*selections, indexing="ij")
    source_positions = np.column_stack(
        [centers[ix.ravel()], centers[iy.ravel()], centers[iz.ravel()]]
    )
    # Offset points away from each void element. Its sign does not affect H, but
    # it gives the physical outward direction for the dipole diagnostic.
    offsets = np.asarray(point_hmpc, dtype=np.float64)[None, :] - source_positions
    radii = np.linalg.norm(offsets, axis=1)
    local_delta = delta[ix, iy, iz].ravel()
    voxel_volume = (box_size_hmpc / side) ** 3
    charges = np.maximum(-local_delta, 0.0) * voxel_volume
    keep = (radii >= inner_hmpc) & (radii <= outer_hmpc) & (charges > 0.0)
    if not np.any(keep):
        raise ValueError("No positive void charge in the declared exterior shell")
    return offsets[keep], charges[keep]


def cage_geometry_for_grid(
    density_contrast: np.ndarray,
    points_hmpc: np.ndarray,
    galaxy_names: tuple[str, ...] | list[str],
    *,
    grid_key: str,
    box_size_hmpc: float,
    inner_hmpc: float,
    outer_hmpc: float,
    power_law_values: tuple[float, ...] = (3.0,),
    yukawa_ranges_hmpc: tuple[float, ...] = (7.8125, 15.625, 31.25, 62.5),
) -> pd.DataFrame:
    points = np.asarray(points_hmpc, dtype=np.float64)
    if points.shape != (len(galaxy_names), 3):
        raise ValueError("points_hmpc must have one three-vector per galaxy")
    rows: list[dict[str, float | bool | int | str]] = []
    for name, point in zip(galaxy_names, points, strict=True):
        offsets, charges = _shell_sources(
            density_contrast,
            point,
            box_size_hmpc=box_size_hmpc,
            inner_hmpc=inner_hmpc,
            outer_hmpc=outer_hmpc,
        )
        radii = np.linalg.norm(offsets, axis=1)
        directions = offsets / radii[:, None]
        total_charge = float(np.sum(charges))
        dipole = np.sum(charges[:, None] * directions, axis=0) / total_charge
        quadrupole = np.einsum("n,ni,nj->ij", charges, directions, directions) / total_charge
        quadrupole -= np.eye(3) / 3.0
        row: dict[str, float | bool | int | str] = {
            "galaxy": str(name),
            f"{grid_key}_shell_void_charge": total_charge,
            f"{grid_key}_shell_void_cells": int(charges.size),
            f"{grid_key}_shell_mean_distance_hmpc": float(np.average(radii, weights=charges)),
            f"{grid_key}_shell_dipole": float(np.linalg.norm(dipole)),
            f"{grid_key}_shell_quadrupole": float(np.linalg.norm(quadrupole)),
        }
        for force_power in power_law_values:
            prefix = f"{grid_key}_power_p{kernel_label(force_power)}"
            metrics = tensor_metrics(
                power_law_hessian(offsets, charges, force_power=force_power)
            )
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        for range_hmpc in yukawa_ranges_hmpc:
            prefix = f"{grid_key}_yukawa_l{kernel_label(range_hmpc)}"
            metrics = tensor_metrics(
                yukawa_hessian(offsets, charges, range_hmpc=range_hmpc)
            )
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("galaxy", kind="stable").reset_index(drop=True)


def balanced_rank_folds(values: np.ndarray, folds: int) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if folds < 2 or folds > scores.size:
        raise ValueError("Invalid number of folds")
    if not np.isfinite(scores).all():
        raise ValueError("Fold scores must be finite")
    order = np.argsort(scores, kind="stable")
    assignment = np.empty(scores.size, dtype=np.int64)
    for rank, index in enumerate(order):
        block, within = divmod(rank, folds)
        assignment[index] = within if block % 2 == 0 else folds - 1 - within
    return assignment


def baryonic_velocity_squared(
    packed: PackedDataset,
    *,
    disk_mass_to_light: float = 0.5,
    bulge_mass_to_light: float = 0.7,
) -> np.ndarray:
    gas_v2 = np.sign(packed.velocity_gas_kms) * packed.velocity_gas_kms**2
    values = (
        gas_v2
        + disk_mass_to_light * packed.velocity_disk_unit_ml_kms**2
        + bulge_mass_to_light * packed.velocity_bulge_unit_ml_kms**2
    )
    return np.maximum(values, 1e-8)


def fixed_rar_velocity(
    packed: PackedDataset,
    baryonic_v2_km2_s2: np.ndarray,
    *,
    acceleration_scale_m_s2: float = 1.2e-10,
) -> np.ndarray:
    radius_m = packed.radius_kpc * KPC_M
    g_bar = baryonic_v2_km2_s2 * 1e6 / radius_m
    denominator = 1.0 - np.exp(-np.sqrt(g_bar / acceleration_scale_m_s2))
    g_predicted = g_bar / np.maximum(denominator, 1e-12)
    return np.sqrt(np.maximum(g_predicted * radius_m / 1e6, 1e-12))


def harmonic_cage_velocity(
    packed: PackedDataset,
    baryonic_v2_km2_s2: np.ndarray,
    *,
    log10_kappa_s2: float,
    environment_by_galaxy: np.ndarray | None = None,
    environment_exponent: float = 0.0,
) -> np.ndarray:
    radius_m = packed.radius_kpc * KPC_M
    if environment_by_galaxy is None:
        response = np.ones(packed.n_galaxies, dtype=np.float64)
    else:
        response = np.power(
            np.maximum(np.asarray(environment_by_galaxy, dtype=np.float64), 1e-12),
            environment_exponent,
        )
    extra_v2 = 10.0**log10_kappa_s2 * response[packed.galaxy_index] * radius_m**2 / 1e6
    return np.sqrt(np.maximum(baryonic_v2_km2_s2 + extra_v2, 1e-12))


def screened_cage_velocity(
    packed: PackedDataset,
    baryonic_v2_km2_s2: np.ndarray,
    *,
    log10_velocity_scale_km_s: float,
    log10_transition_scale_lengths: float,
    environment_by_galaxy: np.ndarray | None = None,
    environment_exponent: float = 0.0,
) -> np.ndarray:
    velocity_scale = 10.0**log10_velocity_scale_km_s
    transition = 10.0**log10_transition_scale_lengths * packed.disk_scale_kpc
    radius = packed.radius_kpc
    radial_activation = radius**2 / (
        radius**2 + transition[packed.galaxy_index] ** 2
    )
    if environment_by_galaxy is None:
        response = np.ones(packed.n_galaxies, dtype=np.float64)
    else:
        response = np.power(
            np.maximum(np.asarray(environment_by_galaxy, dtype=np.float64), 1e-12),
            environment_exponent,
        )
    extra_v2 = velocity_scale**2 * response[packed.galaxy_index] * radial_activation
    return np.sqrt(np.maximum(baryonic_v2_km2_s2 + extra_v2, 1e-12))
