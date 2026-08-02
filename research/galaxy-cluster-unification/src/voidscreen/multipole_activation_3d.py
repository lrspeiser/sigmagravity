"""Scale-free baryonic multipole gate for three-dimensional tensor activation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voidscreen.field_solvers import cell_coordinates
from voidscreen.metric_lensing_3d import TensorActivation3D, exact_tensor_activation_3d


@dataclass(frozen=True)
class ComponentMoments3D:
    mass: float
    centroid: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class MultipoleGate3D:
    gate: float
    dipole_squared: float
    quadrupole_squared: float
    stellar_moments: ComponentMoments3D
    gas_moments: ComponentMoments3D


@dataclass(frozen=True)
class MultipoleGatedActivation3D:
    sigma: np.ndarray
    minimum_eigenvalue_proxy: np.ndarray
    multipole: MultipoleGate3D
    local: TensorActivation3D


def component_moments_3d(density: np.ndarray, spacing: float) -> ComponentMoments3D:
    values = np.maximum(np.asarray(density, dtype=float), 0.0)
    if values.ndim != 3 or min(values.shape) < 5 or spacing <= 0.0:
        raise ValueError("density must be a 3D grid and spacing positive")
    mass = float(np.sum(values) * float(spacing) ** 3)
    if mass <= 0.0:
        raise ValueError("component density must have positive mass")
    coordinates = cell_coordinates(values.shape, spacing)
    total_weight = float(np.sum(values))
    centroid = np.asarray(
        [float(np.sum(values * coordinate) / total_weight) for coordinate in coordinates]
    )
    covariance = np.empty((3, 3), dtype=float)
    centered = [coordinate - centroid[axis] for axis, coordinate in enumerate(coordinates)]
    for first in range(3):
        for second in range(3):
            covariance[first, second] = float(
                np.sum(values * centered[first] * centered[second]) / total_weight
            )
    return ComponentMoments3D(mass=mass, centroid=centroid, covariance=covariance)


def baryonic_multipole_gate_3d(
    stellar_density: np.ndarray,
    gas_density: np.ndarray,
    spacing: float,
) -> MultipoleGate3D:
    stars = component_moments_3d(stellar_density, spacing)
    gas = component_moments_3d(gas_density, spacing)
    star_trace = float(np.trace(stars.covariance))
    gas_trace = float(np.trace(gas.covariance))
    denominator = star_trace + gas_trace
    if denominator <= np.finfo(float).tiny:
        raise ValueError("component second moments must be positive")
    dipole_squared = float(np.sum((stars.centroid - gas.centroid) ** 2) / denominator)
    normalized_star = stars.covariance / star_trace
    normalized_gas = gas.covariance / gas_trace
    quadrupole_squared = float(np.sum((normalized_star - normalized_gas) ** 2))
    invariant = max(dipole_squared + quadrupole_squared, 0.0)
    gate = float(-np.expm1(-invariant))
    return MultipoleGate3D(
        gate=gate,
        dipole_squared=dipole_squared,
        quadrupole_squared=quadrupole_squared,
        stellar_moments=stars,
        gas_moments=gas,
    )


def exact_multipole_gated_activation_3d(
    stellar_density: np.ndarray,
    gas_density: np.ndarray,
    spacing: float,
    **activation_kwargs,
) -> MultipoleGatedActivation3D:
    local = exact_tensor_activation_3d(
        stellar_density,
        gas_density,
        spacing,
        **activation_kwargs,
    )
    multipole = baryonic_multipole_gate_3d(stellar_density, gas_density, spacing)
    sigma = np.clip(local.sigma * multipole.gate, 0.0, 1.0)
    return MultipoleGatedActivation3D(
        sigma=sigma,
        minimum_eigenvalue_proxy=local.mu_newtonian_proxy * (1.0 - sigma),
        multipole=multipole,
        local=local,
    )
