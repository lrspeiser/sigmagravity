"""Variational projected tensor-AQUAL solver for baryonic transport tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

Array = np.ndarray


@dataclass(frozen=True)
class TensorAQUAL2DSolution:
    potential: Array
    acceleration_x: Array
    acceleration_y: Array
    coefficient_mu: Array
    anisotropy_sigma: Array
    normalized_residual_rms: float
    converged: bool
    nonlinear_iterations: int
    metadata: dict = field(default_factory=dict)


def _map2(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 9 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2D map with at least nine cells per axis")
    return array


def _boundary_mask(shape: tuple[int, int]) -> Array:
    mask = np.zeros(shape, dtype=bool)
    mask[0] = True
    mask[-1] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def simple_mu(x: Array) -> Array:
    values = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return values / (1.0 + values)


def constitutive_eigenvalues(mu: Array, sigma: Array) -> tuple[Array, Array]:
    coefficient = np.asarray(mu, dtype=np.float64)
    anisotropy = np.asarray(sigma, dtype=np.float64)
    coefficient, anisotropy = np.broadcast_arrays(coefficient, anisotropy)
    if np.any(coefficient <= 0.0) or np.any(anisotropy < 0.0) or np.any(anisotropy >= 1.0):
        raise ValueError("tensor AQUAL requires mu>0 and 0<=sigma<1")
    return coefficient * (1.0 - anisotropy), coefficient


def _normalized_direction(direction_x: Array, direction_y: Array) -> tuple[Array, Array]:
    x = _map2(direction_x, "direction_x")
    y = _map2(direction_y, "direction_y")
    if x.shape != y.shape:
        raise ValueError("direction maps must have matching shapes")
    norm = np.hypot(x, y)
    unit_x = np.zeros_like(x)
    unit_y = np.zeros_like(y)
    active = norm > 1e-12
    unit_x[active] = x[active] / norm[active]
    unit_y[active] = y[active] / norm[active]
    unit_x[~active] = 1.0
    return unit_x, unit_y


def _isotropic_adjacency(mobility: Array, spacing: float) -> sparse.csr_matrix:
    rows, columns = mobility.shape
    indices = np.arange(rows * columns, dtype=np.intp).reshape((rows, columns))
    edge_rows = []
    edge_columns = []
    edge_weights = []
    for first, second in (
        ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        left = indices[first].ravel()
        right = indices[second].ravel()
        weight = (0.5 * (mobility[first] + mobility[second]) / spacing**2).ravel()
        selected = weight > np.finfo(float).tiny
        edge_rows.extend([left[selected], right[selected]])
        edge_columns.extend([right[selected], left[selected]])
        edge_weights.extend([weight[selected], weight[selected]])
    return sparse.coo_matrix(
        (
            np.concatenate(edge_weights),
            (np.concatenate(edge_rows), np.concatenate(edge_columns)),
        ),
        shape=(rows * columns, rows * columns),
    ).tocsr()


def _oriented_adjacency(
    mobility: Array,
    direction_x: Array,
    direction_y: Array,
    spacing: float,
) -> sparse.csr_matrix:
    rows_count, columns_count = mobility.shape
    rows, columns = np.indices(mobility.shape, dtype=np.float64)
    indices = np.arange(mobility.size, dtype=np.intp).reshape(mobility.shape)
    maximum_row = rows_count - 1
    maximum_column = columns_count - 1
    edge_rows = []
    edge_columns = []
    edge_weights = []
    for sign in (-1.0, 1.0):
        destination_rows = rows + sign * direction_y
        destination_columns = columns + sign * direction_x
        valid_destination = (
            (destination_rows >= 0.0)
            & (destination_rows <= maximum_row)
            & (destination_columns >= 0.0)
            & (destination_columns <= maximum_column)
        )
        safe_rows = np.clip(destination_rows, 0.0, maximum_row)
        safe_columns = np.clip(destination_columns, 0.0, maximum_column)
        row0 = np.floor(safe_rows).astype(np.intp)
        column0 = np.floor(safe_columns).astype(np.intp)
        row1 = np.minimum(row0 + 1, maximum_row)
        column1 = np.minimum(column0 + 1, maximum_column)
        row_fraction = safe_rows - row0
        column_fraction = safe_columns - column0
        neighbors = (
            (row0, column0, (1.0 - row_fraction) * (1.0 - column_fraction)),
            (row0, column1, (1.0 - row_fraction) * column_fraction),
            (row1, column0, row_fraction * (1.0 - column_fraction)),
            (row1, column1, row_fraction * column_fraction),
        )
        for neighbor_rows, neighbor_columns, interpolation in neighbors:
            destinations = indices[neighbor_rows, neighbor_columns]
            neighbor_mobility = mobility[neighbor_rows, neighbor_columns]
            weight = (
                np.sqrt(mobility * neighbor_mobility)
                * interpolation
                / spacing**2
            )
            selected = (
                valid_destination
                & (destinations != indices)
                & (weight > np.finfo(float).tiny)
            )
            edge_rows.append(indices[selected])
            edge_columns.append(destinations[selected])
            edge_weights.append(weight[selected])
    if not edge_weights:
        return sparse.csr_matrix((mobility.size, mobility.size), dtype=np.float64)
    directed = sparse.coo_matrix(
        (
            np.concatenate(edge_weights),
            (np.concatenate(edge_rows), np.concatenate(edge_columns)),
        ),
        shape=(mobility.size, mobility.size),
    ).tocsr()
    adjacency = 0.5 * (directed + directed.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency


def tensor_graph_laplacian(
    mu: Array,
    sigma: Array,
    direction_x: Array,
    direction_y: Array,
    spacing: float,
) -> sparse.csr_matrix:
    coefficient = _map2(mu, "mu")
    anisotropy = _map2(sigma, "sigma")
    if not (
        coefficient.shape
        == anisotropy.shape
        == np.asarray(direction_x).shape
        == np.asarray(direction_y).shape
    ):
        raise ValueError("tensor coefficient maps must have matching shapes")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    constitutive_eigenvalues(coefficient, anisotropy)
    h_x, h_y = _normalized_direction(direction_x, direction_y)
    perpendicular_x = -h_y
    perpendicular_y = h_x
    isotropic = _isotropic_adjacency(coefficient * (1.0 - anisotropy), spacing)
    perpendicular = _oriented_adjacency(
        coefficient * anisotropy,
        perpendicular_x,
        perpendicular_y,
        spacing,
    )
    adjacency = isotropic + perpendicular
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(degree) - adjacency


def _linear_solve(
    source: Array,
    boundary_potential: Array,
    laplacian: sparse.csr_matrix,
    *,
    relative_tolerance: float,
    maximum_iterations: int,
    initial: Array | None = None,
) -> tuple[Array, int]:
    shape = source.shape
    boundary = _boundary_mask(shape)
    interior = ~boundary
    interior_indices = np.flatnonzero(interior.ravel())
    boundary_indices = np.flatnonzero(boundary.ravel())
    operator = laplacian[interior_indices][:, interior_indices].tocsr()
    boundary_block = laplacian[interior_indices][:, boundary_indices]
    rhs = -source.ravel()[interior_indices] - boundary_block @ boundary_potential.ravel()[
        boundary_indices
    ]
    diagonal = operator.diagonal()
    inverse_diagonal = 1.0 / np.maximum(diagonal, np.finfo(float).tiny)
    preconditioner = sparse_linalg.LinearOperator(
        operator.shape,
        matvec=lambda vector: inverse_diagonal * vector,
        dtype=np.float64,
    )
    initial_values = None if initial is None else initial.ravel()[interior_indices]
    values, information = sparse_linalg.cg(
        operator,
        rhs,
        x0=initial_values,
        rtol=float(relative_tolerance),
        atol=0.0,
        maxiter=int(maximum_iterations),
        M=preconditioner,
    )
    if information < 0:
        raise RuntimeError("tensor-AQUAL conjugate-gradient solve failed")
    potential = np.asarray(boundary_potential, dtype=np.float64).copy()
    potential.ravel()[interior_indices] = values
    return potential, int(information)


def _normalized_residual(
    potential: Array, source: Array, laplacian: sparse.csr_matrix
) -> float:
    interior = ~_boundary_mask(source.shape)
    residual = (-laplacian @ potential.ravel()).reshape(source.shape) - source
    numerator = float(np.sqrt(np.mean(np.square(residual[interior]))))
    denominator = float(np.sqrt(np.mean(np.square(source[interior]))))
    return numerator / max(denominator, np.finfo(float).tiny)


def solve_projected_tensor_aqual(
    source: Array,
    spacing: float,
    boundary_potential: Array,
    anisotropy_sigma: Array,
    transport_direction_x: Array,
    transport_direction_y: Array,
    *,
    a0: float,
    mu_function: Callable[[Array], Array] = simple_mu,
    residual_tolerance: float = 1e-5,
    maximum_nonlinear_iterations: int = 80,
    maximum_linear_iterations: int = 3000,
    linear_relative_tolerance: float = 1e-10,
    damping: float = 0.65,
    mu_floor: float = 1e-6,
) -> TensorAQUAL2DSolution:
    values = _map2(source, "source")
    boundary = _map2(boundary_potential, "boundary_potential")
    sigma = _map2(anisotropy_sigma, "anisotropy_sigma")
    direction_x = _map2(transport_direction_x, "transport_direction_x")
    direction_y = _map2(transport_direction_y, "transport_direction_y")
    if not (values.shape == boundary.shape == sigma.shape == direction_x.shape == direction_y.shape):
        raise ValueError("tensor-AQUAL maps must have matching shapes")
    if a0 <= 0.0 or spacing <= 0.0 or not 0.0 < damping <= 1.0 or mu_floor <= 0.0:
        raise ValueError("tensor-AQUAL scales and solver controls are invalid")
    if np.any(sigma < 0.0) or np.any(sigma >= 1.0):
        raise ValueError("anisotropy_sigma must lie in [0,1)")

    unit_mu = np.ones_like(values)
    initial_laplacian = tensor_graph_laplacian(
        unit_mu, sigma, direction_x, direction_y, spacing
    )
    potential, initial_information = _linear_solve(
        values,
        boundary,
        initial_laplacian,
        relative_tolerance=linear_relative_tolerance,
        maximum_iterations=maximum_linear_iterations,
    )
    linear_information = [initial_information]
    converged = False
    residual = np.inf
    coefficient = unit_mu
    iterations = 0
    for iterations in range(1, int(maximum_nonlinear_iterations) + 1):
        gradient_y, gradient_x = np.gradient(
            potential, float(spacing), float(spacing), edge_order=2
        )
        magnitude = np.hypot(gradient_x, gradient_y)
        coefficient = np.maximum(
            np.asarray(mu_function(magnitude / float(a0)), dtype=np.float64),
            float(mu_floor),
        )
        laplacian = tensor_graph_laplacian(
            coefficient, sigma, direction_x, direction_y, spacing
        )
        solved, information = _linear_solve(
            values,
            boundary,
            laplacian,
            relative_tolerance=linear_relative_tolerance,
            maximum_iterations=maximum_linear_iterations,
            initial=potential,
        )
        linear_information.append(information)
        updated = float(damping) * solved + (1.0 - float(damping)) * potential
        updated[_boundary_mask(values.shape)] = boundary[_boundary_mask(values.shape)]
        potential = updated
        gradient_y, gradient_x = np.gradient(
            potential, float(spacing), float(spacing), edge_order=2
        )
        coefficient = np.maximum(
            np.asarray(
                mu_function(np.hypot(gradient_x, gradient_y) / float(a0)),
                dtype=np.float64,
            ),
            float(mu_floor),
        )
        laplacian = tensor_graph_laplacian(
            coefficient, sigma, direction_x, direction_y, spacing
        )
        residual = _normalized_residual(potential, values, laplacian)
        if residual <= float(residual_tolerance) and information == 0:
            converged = True
            break
    acceleration_x = -np.gradient(potential, float(spacing), axis=1, edge_order=2)
    acceleration_y = -np.gradient(potential, float(spacing), axis=0, edge_order=2)
    minimum_eigenvalue, maximum_eigenvalue = constitutive_eigenvalues(
        coefficient, sigma
    )
    return TensorAQUAL2DSolution(
        potential=potential,
        acceleration_x=acceleration_x,
        acceleration_y=acceleration_y,
        coefficient_mu=coefficient,
        anisotropy_sigma=sigma,
        normalized_residual_rms=float(residual),
        converged=converged,
        nonlinear_iterations=iterations,
        metadata={
            "law": "projected tensor AQUAL",
            "equivalent_tensor": "mu(I-sigma h h)",
            "minimum_constitutive_eigenvalue": float(np.min(minimum_eigenvalue)),
            "maximum_constitutive_eigenvalue": float(np.max(maximum_eigenvalue)),
            "linear_solver_information": linear_information,
            "damping": float(damping),
            "mu_floor": float(mu_floor),
            "spacing": float(spacing),
        },
    )
