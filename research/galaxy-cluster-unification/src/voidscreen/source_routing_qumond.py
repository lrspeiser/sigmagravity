"""Source-conserving baryonic routing built from QUMOND development fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from voidscreen.field_solvers import (
    FieldSolution,
    acceleration_from_potential,
    gradient,
    laplacian,
    normalized_residual_rms,
    solve_newtonian,
    solve_poisson_dirichlet,
)
from voidscreen.spatial_qumond_3d import (
    C_M_S,
    divergence_powered_qumond_gradient,
    path_geometry_from_newtonian,
    path_qumond_monopole_boundary,
)


@dataclass(frozen=True)
class SourceRoutingSolution:
    field: FieldSolution
    newtonian: FieldSolution
    base_source: np.ndarray
    local_generator_source: np.ndarray
    local_extra_source: np.ndarray
    positive_routed_source: np.ndarray
    negative_shell_source: np.ndarray
    routed_source: np.ndarray
    transition_shell_weight: np.ndarray
    potential_depth: np.ndarray
    local_channel_exponent: np.ndarray
    boundary_potential: np.ndarray
    positive_generator_strength: float


def _spacing3(spacing: float | Sequence[float]) -> tuple[float, float, float]:
    if np.isscalar(spacing):
        values = (float(spacing),) * 3
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError("spacing must contain three positive values")
    return values


def solve_source_conserving_baryonic_routing(
    density: np.ndarray,
    spacing: float | Sequence[float],
    *,
    gravitational_constant: float = 6.67430e-11,
    a0: float = 1.2e-10,
    transition_depth: float = 1e-6,
    transition_power: float = 4.0,
    extra_spatial_channels: float = 2.0,
    path_power: float = 0.5,
    light_speed: float = C_M_S,
) -> SourceRoutingSolution:
    rho = np.asarray(density, dtype=float)
    if rho.ndim != 3 or min(rho.shape) < 5 or np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("density must be a finite nonnegative 3D grid")
    steps = _spacing3(spacing)
    cell_volume = float(np.prod(steps))
    baryonic_mass = float(np.sum(rho) * cell_volume)
    if baryonic_mass <= 0.0:
        raise ValueError("density must have positive total mass")

    newtonian = solve_newtonian(
        rho,
        steps,
        gravitational_constant=gravitational_constant,
    )
    geometry = path_geometry_from_newtonian(
        newtonian.potential,
        newtonian.acceleration,
        steps,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=extra_spatial_channels,
        path_power=path_power,
        light_speed=light_speed,
    )
    base_source = divergence_powered_qumond_gradient(
        newtonian.potential,
        np.ones_like(rho),
        steps,
        a0=a0,
    )
    local_source = divergence_powered_qumond_gradient(
        newtonian.potential,
        geometry["channel_exponent"],
        steps,
        a0=a0,
    )
    local_extra = local_source - base_source
    positive_strength = float(np.sum(np.maximum(local_extra, 0.0)) * cell_volume)
    if not np.isfinite(positive_strength) or positive_strength <= 0.0:
        raise ValueError("local generator must have positive source strength")

    onset = geometry["potential_onset"]
    depth_gradient = gradient(geometry["potential_depth"], steps)
    depth_gradient_magnitude = np.sqrt(sum(component * component for component in depth_gradient))
    shell_weight = 4.0 * onset * (1.0 - onset) * depth_gradient_magnitude
    shell_integral = float(np.sum(shell_weight) * cell_volume)
    if not np.isfinite(shell_integral) or shell_integral <= 0.0:
        raise ValueError("transition shell must have positive integral")

    positive_route = positive_strength * rho / baryonic_mass
    negative_shell = positive_strength * shell_weight / shell_integral
    routed_source = base_source + positive_route - negative_shell
    boundary = path_qumond_monopole_boundary(
        rho,
        steps,
        gravitational_constant=gravitational_constant,
        a0=a0,
        transition_depth=transition_depth,
        transition_power=transition_power,
        extra_spatial_channels=0.0,
        path_power=path_power,
        light_speed=light_speed,
    )
    potential = solve_poisson_dirichlet(routed_source, steps, boundary)
    residual = laplacian(potential, steps) - routed_source
    residual_rms = normalized_residual_rms(residual, routed_source)
    field = FieldSolution(
        potential=potential,
        acceleration=acceleration_from_potential(potential, steps),
        equation_source=routed_source,
        normalized_residual_rms=residual_rms,
        converged=bool(newtonian.converged and np.isfinite(residual_rms)),
        metadata={
            "law": "source-conserving baryonic routing over fixed-RAR QUMOND",
            "spacing": steps,
            "a0": float(a0),
            "transition_depth": float(transition_depth),
            "transition_power": float(transition_power),
            "extra_spatial_channels": float(extra_spatial_channels),
            "path_power": float(path_power),
            "newtonian_residual_rms": newtonian.normalized_residual_rms,
        },
    )
    return SourceRoutingSolution(
        field=field,
        newtonian=newtonian,
        base_source=base_source,
        local_generator_source=local_source,
        local_extra_source=local_extra,
        positive_routed_source=positive_route,
        negative_shell_source=negative_shell,
        routed_source=routed_source,
        transition_shell_weight=shell_weight,
        potential_depth=geometry["potential_depth"],
        local_channel_exponent=geometry["channel_exponent"],
        boundary_potential=boundary,
        positive_generator_strength=positive_strength,
    )
