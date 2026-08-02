"""Variational three-dimensional tensor-AQUAL field solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from voidscreen.field_solvers import boundary_mask
from voidscreen.tensor_aqual import simple_mu

Array = np.ndarray


@dataclass(frozen=True)
class TensorAQUAL3DSolution:
    potential: Array
    acceleration: tuple[Array, Array, Array]
    coefficient_mu: Array
    anisotropy_sigma: Array
    normalized_residual_rms: float
    converged: bool
    nonlinear_iterations: int
    metadata: dict = field(default_factory=dict)


def _map3(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or min(array.shape) < 5 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3D map with at least five cells per axis")
    return array


def constitutive_eigenvalues_3d(mu: Array, sigma: Array) -> tuple[Array, Array, Array]:
    coefficient, anisotropy = np.broadcast_arrays(
        np.asarray(mu, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    if np.any(coefficient <= 0.0) or np.any(anisotropy < 0.0) or np.any(anisotropy >= 1.0):
        raise ValueError("3D tensor AQUAL requires mu>0 and 0<=sigma<1")
    return coefficient * (1.0 - anisotropy), coefficient, coefficient


def _normalized_direction3(
    direction_x: Array,
    direction_y: Array,
    direction_z: Array,
) -> tuple[Array, Array, Array]:
    x = _map3(direction_x, "direction_x")
    y = _map3(direction_y, "direction_y")
    z = _map3(direction_z, "direction_z")
    if not (x.shape == y.shape == z.shape):
        raise ValueError("direction maps must have matching shapes")
    norm = np.sqrt(x * x + y * y + z * z)
    active = norm > 1e-12
    unit_x = np.where(active, x / np.maximum(norm, 1e-12), 1.0)
    unit_y = np.where(active, y / np.maximum(norm, 1e-12), 0.0)
    unit_z = np.where(active, z / np.maximum(norm, 1e-12), 0.0)
    return unit_x, unit_y, unit_z


def perpendicular_basis_3d(
    direction_x: Array,
    direction_y: Array,
    direction_z: Array,
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
    """Return a deterministic orthonormal basis perpendicular to ``h``."""

    h_x, h_y, h_z = _normalized_direction3(direction_x, direction_y, direction_z)
    use_z_reference = np.abs(h_z) < 0.9
    n1_x = np.where(use_z_reference, h_y, -h_z)
    n1_y = np.where(use_z_reference, -h_x, 0.0)
    n1_z = np.where(use_z_reference, 0.0, h_x)
    n1_norm = np.sqrt(n1_x * n1_x + n1_y * n1_y + n1_z * n1_z)
    n1_x /= np.maximum(n1_norm, 1e-12)
    n1_y /= np.maximum(n1_norm, 1e-12)
    n1_z /= np.maximum(n1_norm, 1e-12)
    n2_x = h_y * n1_z - h_z * n1_y
    n2_y = h_z * n1_x - h_x * n1_z
    n2_z = h_x * n1_y - h_y * n1_x
    return (n1_x, n1_y, n1_z), (n2_x, n2_y, n2_z)


def _isotropic_adjacency_3d(mobility: Array, spacing: float) -> sparse.csr_matrix:
    shape = mobility.shape
    indices = np.arange(mobility.size, dtype=np.intp).reshape(shape)
    edge_rows = []
    edge_columns = []
    edge_weights = []
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        first_tuple = tuple(first)
        second_tuple = tuple(second)
        left = indices[first_tuple].ravel()
        right = indices[second_tuple].ravel()
        weight = (
            0.5 * (mobility[first_tuple] + mobility[second_tuple]) / float(spacing) ** 2
        ).ravel()
        selected = weight > np.finfo(float).tiny
        edge_rows.extend([left[selected], right[selected]])
        edge_columns.extend([right[selected], left[selected]])
        edge_weights.extend([weight[selected], weight[selected]])
    return sparse.coo_matrix(
        (
            np.concatenate(edge_weights),
            (np.concatenate(edge_rows), np.concatenate(edge_columns)),
        ),
        shape=(mobility.size, mobility.size),
    ).tocsr()


def _oriented_adjacency_3d(
    mobility: Array,
    direction: tuple[Array, Array, Array],
    spacing: float,
) -> sparse.csr_matrix:
    shape = mobility.shape
    coordinates = np.indices(shape, dtype=float)
    indices = np.arange(mobility.size, dtype=np.intp).reshape(shape)
    maximum = np.asarray(shape, dtype=float) - 1.0
    edge_rows = []
    edge_columns = []
    edge_weights = []
    for sign in (-1.0, 1.0):
        destinations = [coordinates[axis] + sign * direction[axis] for axis in range(3)]
        valid = np.ones(shape, dtype=bool)
        safe = []
        lower = []
        upper = []
        fraction = []
        for axis in range(3):
            valid &= (destinations[axis] >= 0.0) & (destinations[axis] <= maximum[axis])
            safe_axis = np.clip(destinations[axis], 0.0, maximum[axis])
            lower_axis = np.floor(safe_axis).astype(np.intp)
            safe.append(safe_axis)
            lower.append(lower_axis)
            upper.append(np.minimum(lower_axis + 1, int(maximum[axis])))
            fraction.append(safe_axis - lower_axis)
        for bit0 in (0, 1):
            for bit1 in (0, 1):
                for bit2 in (0, 1):
                    bits = (bit0, bit1, bit2)
                    neighbor_coordinates = tuple(
                        upper[axis] if bits[axis] else lower[axis] for axis in range(3)
                    )
                    interpolation = np.ones(shape, dtype=float)
                    for axis in range(3):
                        interpolation *= fraction[axis] if bits[axis] else 1.0 - fraction[axis]
                    neighbor_indices = indices[neighbor_coordinates]
                    neighbor_mobility = mobility[neighbor_coordinates]
                    weight = (
                        np.sqrt(mobility * neighbor_mobility)
                        * interpolation
                        / float(spacing) ** 2
                    )
                    selected = (
                        valid
                        & (neighbor_indices != indices)
                        & (weight > np.finfo(float).tiny)
                    )
                    edge_rows.append(indices[selected])
                    edge_columns.append(neighbor_indices[selected])
                    edge_weights.append(weight[selected])
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


def tensor_graph_laplacian_3d(
    mu: Array,
    sigma: Array,
    direction_x: Array,
    direction_y: Array,
    direction_z: Array,
    spacing: float,
) -> sparse.csr_matrix:
    coefficient = _map3(mu, "mu")
    anisotropy = _map3(sigma, "sigma")
    if not (
        coefficient.shape
        == anisotropy.shape
        == np.asarray(direction_x).shape
        == np.asarray(direction_y).shape
        == np.asarray(direction_z).shape
    ):
        raise ValueError("3D tensor coefficient maps must have matching shapes")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    constitutive_eigenvalues_3d(coefficient, anisotropy)
    n1, n2 = perpendicular_basis_3d(direction_x, direction_y, direction_z)
    isotropic = _isotropic_adjacency_3d(coefficient * (1.0 - anisotropy), spacing)
    perpendicular_mobility = coefficient * anisotropy
    adjacency = (
        isotropic
        + _oriented_adjacency_3d(perpendicular_mobility, n1, spacing)
        + _oriented_adjacency_3d(perpendicular_mobility, n2, spacing)
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(degree) - adjacency


def _linear_solve_3d(
    source: Array,
    boundary_potential: Array,
    laplacian: sparse.csr_matrix,
    *,
    relative_tolerance: float,
    maximum_iterations: int,
    initial: Array | None = None,
) -> tuple[Array, int]:
    shape = source.shape
    boundary = boundary_mask(shape)
    interior = ~boundary
    interior_indices = np.flatnonzero(interior.ravel())
    boundary_indices = np.flatnonzero(boundary.ravel())
    operator = laplacian[interior_indices][:, interior_indices].tocsr()
    boundary_block = laplacian[interior_indices][:, boundary_indices]
    rhs = -source.ravel()[interior_indices] - boundary_block @ boundary_potential.ravel()[
        boundary_indices
    ]
    inverse_diagonal = 1.0 / np.maximum(operator.diagonal(), np.finfo(float).tiny)
    preconditioner = sparse_linalg.LinearOperator(
        operator.shape,
        matvec=lambda vector: inverse_diagonal * vector,
        dtype=float,
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
        raise RuntimeError("3D tensor-AQUAL conjugate-gradient solve failed")
    potential = np.asarray(boundary_potential, dtype=float).copy()
    potential.ravel()[interior_indices] = values
    return potential, int(information)


def _normalized_residual_3d(
    potential: Array,
    source: Array,
    laplacian: sparse.csr_matrix,
) -> float:
    interior = ~boundary_mask(source.shape)
    residual = (-laplacian @ potential.ravel()).reshape(source.shape) - source
    numerator = float(np.sqrt(np.mean(residual[interior] ** 2)))
    denominator = float(np.sqrt(np.mean(source[interior] ** 2)))
    return numerator / max(denominator, np.finfo(float).tiny)


def solve_tensor_aqual_3d(
    source: Array,
    spacing: float,
    boundary_potential: Array,
    anisotropy_sigma: Array,
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
    sigma = _map3(anisotropy_sigma, "anisotropy_sigma")
    directions = tuple(
        _map3(component, name)
        for component, name in zip(
            (transport_direction_x, transport_direction_y, transport_direction_z),
            ("transport_direction_x", "transport_direction_y", "transport_direction_z"),
            strict=True,
        )
    )
    if not (values.shape == boundary.shape == sigma.shape == directions[0].shape == directions[1].shape == directions[2].shape):
        raise ValueError("3D tensor-AQUAL maps must have matching shapes")
    if a0 <= 0.0 or spacing <= 0.0 or not 0.0 < damping <= 1.0 or mu_floor <= 0.0:
        raise ValueError("3D tensor-AQUAL scales and controls are invalid")
    if np.any(sigma < 0.0) or np.any(sigma >= 1.0):
        raise ValueError("anisotropy_sigma must lie in [0,1)")

    coefficient = np.ones_like(values)
    laplacian = tensor_graph_laplacian_3d(
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
        laplacian = tensor_graph_laplacian_3d(
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
        laplacian = tensor_graph_laplacian_3d(
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
    minimum, middle, maximum = constitutive_eigenvalues_3d(coefficient, sigma)
    return TensorAQUAL3DSolution(
        potential=potential,
        acceleration=acceleration,
        coefficient_mu=coefficient,
        anisotropy_sigma=sigma,
        normalized_residual_rms=float(residual),
        converged=converged,
        nonlinear_iterations=iterations,
        metadata={
            "law": "three-dimensional tensor AQUAL",
            "equivalent_tensor": "mu(I-sigma h h)",
            "minimum_constitutive_eigenvalue": float(np.min(minimum)),
            "middle_constitutive_eigenvalue": float(np.max(middle)),
            "maximum_constitutive_eigenvalue": float(np.max(maximum)),
            "linear_solver_information": linear_information,
            "damping": float(damping),
            "mu_floor": float(mu_floor),
            "spacing": float(spacing),
        },
    )
