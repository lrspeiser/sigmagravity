from __future__ import annotations

import numpy as np

from voidscreen.tensor_aqual_3d import tensor_graph_laplacian_3d
from voidscreen.transverse_confinement_3d import (
    confinement_eigenvalues_3d,
    solve_transverse_confinement_aqual_3d,
    transverse_confinement_graph_laplacian_3d,
)


def maps(shape=(5, 5, 5)):
    direction_x = np.ones(shape)
    direction_y = np.zeros(shape)
    direction_z = np.zeros(shape)
    return direction_x, direction_y, direction_z


def test_confinement_eigenvalues_leave_route_open_and_suppress_two_modes():
    mu = np.asarray([0.4, 0.8])
    sigma = np.asarray([0.25, 0.75])
    along, first, second = confinement_eigenvalues_3d(mu, sigma)
    assert np.allclose(along, mu)
    assert np.allclose(first, mu * (1.0 - sigma))
    assert np.allclose(second, first)
    assert np.all(first > 0.0)


def test_zero_confinement_matches_isotropic_tensor_graph():
    direction = maps()
    mu = np.full(direction[0].shape, 0.6)
    sigma = np.zeros_like(mu)
    expected = tensor_graph_laplacian_3d(mu, sigma, *direction, 1.0)
    actual = transverse_confinement_graph_laplacian_3d(
        mu,
        sigma,
        *direction,
        1.0,
    )
    assert np.allclose((actual - expected).toarray(), 0.0)


def test_confinement_graph_is_invariant_to_route_reversal():
    direction = maps()
    mu = np.full(direction[0].shape, 0.6)
    sigma = np.full_like(mu, 0.3)
    forward = transverse_confinement_graph_laplacian_3d(
        mu,
        sigma,
        *direction,
        1.0,
    )
    reverse = transverse_confinement_graph_laplacian_3d(
        mu,
        sigma,
        *tuple(-component for component in direction),
        1.0,
    )
    assert np.allclose((forward - reverse).toarray(), 0.0)


def test_small_confinement_solve_is_finite_and_positive():
    shape = (7, 7, 7)
    direction = maps(shape)
    coordinates = np.indices(shape, dtype=float)
    radius_squared = sum((axis - 3.0) ** 2 for axis in coordinates)
    source = np.exp(-0.5 * radius_squared)
    boundary = np.zeros(shape)
    sigma = np.full(shape, 0.2)
    result = solve_transverse_confinement_aqual_3d(
        source,
        1.0,
        boundary,
        sigma,
        *direction,
        a0=1.0,
        residual_tolerance=1e-4,
        maximum_nonlinear_iterations=80,
    )
    assert result.converged
    assert np.all(np.isfinite(result.potential))
    assert result.metadata["minimum_constitutive_eigenvalue"] > 0.0
