"""Three-dimensional AQUAL with two transversely suppressed eigenmodes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse

from voidscreen.field_solvers import boundary_mask
from voidscreen.tensor_aqual import simple_mu
from voidscreen.tensor_aqual_3d import (
    TensorAQUAL3DSolution,
    _isotropic_adjacency_3d,
    _linear_solve_3d,
    _map3,
    _normalized_direction3,
    _normalized_residual_3d,
    _oriented_adjacency_3d,
)

Array = np.ndarray


def confinement_eigenvalues_3d(
    mu: Array,
    sigma: Array,
) -> tuple[Array, Array, Array]:
    """Return eigenvalues along h and in its two perpendicular directions."""
    coefficient, confinement = np.broadcast_arrays(
        np.asarray(mu, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    if (
        np.any(coefficient <= 0.0)
        or np.any(confinement < 0.0)
        or np.any(confinement >= 1.0)
    ):
        raise ValueError("transverse confinement requires mu>0 and 0<=sigma<1")
    transverse = coefficient * (1.0 - confinement)
    return coefficient, transverse, transverse


def transverse_confinement_graph_laplacian_3d(
    mu: Array,
    sigma: Array,
    direction_x: Array,
    direction_y: Array,
    direction_z: Array,
    spacing: float,
) -> sparse.csr_matrix:
    """Discretize ``mu[(1-sigma)I + sigma h h]`` as a symmetric graph."""
    coefficient = _map3(mu, "mu")
    confinement = _map3(sigma, "sigma")
    if not (
        coefficient.shape
        == confinement.shape
        == np.asarray(direction_x).shape
        == np.asarray(direction_y).shape
        == np.asarray(direction_z).shape
    ):
        raise ValueError("3D confinement coefficient maps must have matching shapes")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    confinement_eigenvalues_3d(coefficient, confinement)
    direction = _normalized_direction3(direction_x, direction_y, direction_z)
    isotropic = _isotropic_adjacency_3d(
        coefficient * (1.0 - confinement),
        spacing,
    )
    routed = _oriented_adjacency_3d(
        coefficient * confinement,
        direction,
        spacing,
    )
    adjacency = isotropic + routed
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(degree) - adjacency


def solve_transverse_confinement_aqual_3d(
    source: Array,
    spacing: float,
    boundary_potential: Array,
    confinement_sigma: Array,
    transport_direction_x: Array,
    transport_direction_y: Array,
    transport_direction_z: Array,
    *,
    a0: float,
    mu_function: Callable[[Array], Array] = simple_mu,
    residual_tolerance: float = 1e-5,
    maximum_nonlinear_iterations: int = 80,
    maximum_linear_iterations: int = 5000,
    linear_relative_tolerance: float = 1e-10,
    damping: float = 0.65,
    mu_floor: float = 1e-6,
) -> TensorAQUAL3DSolution:
    values = _map3(source, "source")
    boundary = _map3(boundary_potential, "boundary_potential")
    sigma = _map3(confinement_sigma, "confinement_sigma")
    directions = tuple(
        _map3(component, name)
        for component, name in zip(
            (transport_direction_x, transport_direction_y, transport_direction_z),
            ("transport_direction_x", "transport_direction_y", "transport_direction_z"),
            strict=True,
        )
    )
    if not (
        values.shape
        == boundary.shape
        == sigma.shape
        == directions[0].shape
        == directions[1].shape
        == directions[2].shape
    ):
        raise ValueError("3D transverse-confinement maps must have matching shapes")
    if a0 <= 0.0 or spacing <= 0.0 or not 0.0 < damping <= 1.0 or mu_floor <= 0.0:
        raise ValueError("3D transverse-confinement scales and controls are invalid")
    if np.any(sigma < 0.0) or np.any(sigma >= 1.0):
        raise ValueError("confinement_sigma must lie in [0,1)")

    coefficient = np.ones_like(values)
    laplacian = transverse_confinement_graph_laplacian_3d(
        coefficient,
        sigma,
        *directions,
        spacing,
    )
    potential, initial_information = _linear_solve_3d(
        values,
        boundary,
        laplacian,
        relative_tolerance=linear_relative_tolerance,
        maximum_iterations=maximum_linear_iterations,
    )
    linear_information = [initial_information]
    residual = np.inf
    converged = False
    iterations = 0
    for iterations in range(1, int(maximum_nonlinear_iterations) + 1):
        gradients = np.gradient(potential, float(spacing), edge_order=2)
        magnitude = np.sqrt(sum(component * component for component in gradients))
        coefficient = np.maximum(
            np.asarray(mu_function(magnitude / float(a0)), dtype=float),
            float(mu_floor),
        )
        laplacian = transverse_confinement_graph_laplacian_3d(
            coefficient,
            sigma,
            *directions,
            spacing,
        )
        solved, information = _linear_solve_3d(
            values,
            boundary,
            laplacian,
            relative_tolerance=linear_relative_tolerance,
            maximum_iterations=maximum_linear_iterations,
            initial=potential,
        )
        linear_information.append(information)
        potential = float(damping) * solved + (1.0 - float(damping)) * potential
        mask = boundary_mask(values.shape)
        potential[mask] = boundary[mask]
        gradients = np.gradient(potential, float(spacing), edge_order=2)
        magnitude = np.sqrt(sum(component * component for component in gradients))
        coefficient = np.maximum(
            np.asarray(mu_function(magnitude / float(a0)), dtype=float),
            float(mu_floor),
        )
        laplacian = transverse_confinement_graph_laplacian_3d(
            coefficient,
            sigma,
            *directions,
            spacing,
        )
        residual = _normalized_residual_3d(potential, values, laplacian)
        if residual <= float(residual_tolerance) and information == 0:
            converged = True
            break
    acceleration = tuple(
        -component for component in np.gradient(potential, float(spacing), edge_order=2)
    )
    along, transverse_first, transverse_second = confinement_eigenvalues_3d(
        coefficient,
        sigma,
    )
    return TensorAQUAL3DSolution(
        potential=potential,
        acceleration=acceleration,
        coefficient_mu=coefficient,
        anisotropy_sigma=sigma,
        normalized_residual_rms=float(residual),
        converged=converged,
        nonlinear_iterations=iterations,
        metadata={
            "law": "three-dimensional transverse-confinement AQUAL",
            "equivalent_tensor": "mu[(1-sigma)I+sigma h h]",
            "minimum_constitutive_eigenvalue": float(
                np.min(np.minimum(transverse_first, transverse_second))
            ),
            "maximum_route_eigenvalue": float(np.max(along)),
            "linear_solver_information": linear_information,
            "damping": float(damping),
            "mu_floor": float(mu_floor),
            "spacing": float(spacing),
        },
    )
