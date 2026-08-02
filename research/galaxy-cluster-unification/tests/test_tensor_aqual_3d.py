from __future__ import annotations

import numpy as np

from voidscreen.tensor_aqual_3d import (
    perpendicular_basis_3d,
    solve_tensor_aqual_3d,
    tensor_graph_laplacian_3d,
)


def constant_mu(values):
    return np.ones_like(values)


def manufactured(cells=17, sigma_value=0.3):
    axis = np.linspace(-1.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    exact = np.cos(0.5 * np.pi * x) * np.cos(0.5 * np.pi * y) * np.cos(0.5 * np.pi * z)
    wave2 = (0.5 * np.pi) ** 2
    source = -wave2 * (3.0 - sigma_value) * exact
    sigma = np.full_like(exact, sigma_value)
    direction_x = np.ones_like(exact)
    direction_y = np.zeros_like(exact)
    direction_z = np.zeros_like(exact)
    solution = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(exact),
        sigma,
        direction_x,
        direction_y,
        direction_z,
        a0=1.0,
        mu_function=constant_mu,
    )
    return exact, solution, spacing


def test_manufactured_anisotropic_solution_is_accurate():
    exact, solution, _ = manufactured()
    error = np.sqrt(np.mean((solution.potential - exact) ** 2)) / np.sqrt(np.mean(exact**2))
    assert error < 0.005


def test_direction_reversal_and_axis_rotation_are_invariant():
    exact, solution, spacing = manufactured(cells=9)
    wave2 = (0.5 * np.pi) ** 2
    source = -wave2 * (3.0 - 0.3) * exact
    sigma = np.full_like(exact, 0.3)
    reversed_solution = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(exact),
        sigma,
        -np.ones_like(exact),
        np.zeros_like(exact),
        np.zeros_like(exact),
        a0=1.0,
        mu_function=constant_mu,
    )
    assert np.allclose(solution.potential, reversed_solution.potential, rtol=1e-12, atol=1e-12)
    rotated = solve_tensor_aqual_3d(
        np.swapaxes(source, 0, 1),
        spacing,
        np.zeros_like(exact),
        np.swapaxes(sigma, 0, 1),
        np.zeros_like(exact),
        np.ones_like(exact),
        np.zeros_like(exact),
        a0=1.0,
        mu_function=constant_mu,
    )
    assert np.allclose(solution.potential, np.swapaxes(rotated.potential, 0, 1), rtol=1e-11, atol=1e-11)


def test_perpendicular_basis_is_orthonormal_and_operator_symmetric():
    shape = (7, 7, 7)
    h_x = np.full(shape, 0.3)
    h_y = np.full(shape, -0.4)
    h_z = np.full(shape, np.sqrt(0.75))
    n1, n2 = perpendicular_basis_3d(h_x, h_y, h_z)
    for vector in (n1, n2):
        assert np.allclose(np.sqrt(sum(component**2 for component in vector)), 1.0)
        assert np.allclose(h_x * vector[0] + h_y * vector[1] + h_z * vector[2], 0.0)
    assert np.allclose(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2], 0.0)
    operator = tensor_graph_laplacian_3d(
        np.ones(shape),
        np.full(shape, 0.2),
        h_x,
        h_y,
        h_z,
        1.0,
    )
    difference = operator - operator.T
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-14


def test_nonlinear_solution_converges():
    cells = 9
    axis = np.linspace(-2.0, 2.0, cells)
    spacing = float(axis[1] - axis[0])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    source = np.exp(-0.5 * ((x / 0.6) ** 2 + (y / 0.8) ** 2 + (z / 1.0) ** 2))
    sigma = 0.15 * np.exp(-0.5 * (x * x + y * y + z * z))
    solution = solve_tensor_aqual_3d(
        source,
        spacing,
        np.zeros_like(source),
        sigma,
        np.ones_like(source),
        np.zeros_like(source),
        np.zeros_like(source),
        a0=0.2,
    )
    assert solution.converged
    assert solution.normalized_residual_rms < 1e-5
    assert solution.metadata["minimum_constitutive_eigenvalue"] > 0.0
