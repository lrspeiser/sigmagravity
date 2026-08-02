from __future__ import annotations

import numpy as np

from voidscreen.tensor_aqual import (
    constitutive_eigenvalues,
    solve_projected_tensor_aqual,
)


def constant_mu(values):
    return np.ones_like(np.asarray(values, dtype=float))


def manufactured(cells: int, sigma_value: float = 0.35):
    axis = np.linspace(-1.0, 1.0, cells)
    spacing = float(axis[1] - axis[0])
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    potential = np.sin(0.5 * np.pi * (xx + 1.0)) * np.sin(
        0.5 * np.pi * (yy + 1.0)
    )
    wave2 = (0.5 * np.pi) ** 2
    source = -wave2 * (2.0 - sigma_value) * potential
    sigma = np.full_like(source, sigma_value)
    direction_x = np.ones_like(source)
    direction_y = np.zeros_like(source)
    boundary = np.zeros_like(source)
    solution = solve_projected_tensor_aqual(
        source,
        spacing,
        boundary,
        sigma,
        direction_x,
        direction_y,
        a0=1.0,
        mu_function=constant_mu,
    )
    interior = (slice(1, -1), slice(1, -1))
    error = float(
        np.sqrt(np.mean(np.square(solution.potential[interior] - potential[interior])))
        / np.sqrt(np.mean(np.square(potential[interior])))
    )
    return solution, error, source, sigma, direction_x, direction_y


def test_constitutive_eigenvalues_are_positive_and_ordered():
    minimum, maximum = constitutive_eigenvalues(
        np.array([0.2, 0.8]), np.array([0.0, 0.35])
    )
    assert np.all(minimum > 0.0)
    assert np.all(maximum >= minimum)
    assert np.allclose(minimum, [0.2, 0.52])


def test_manufactured_anisotropic_solution_converges_at_second_order():
    errors = [manufactured(cells)[1] for cells in (17, 33, 65)]
    order_one = np.log(errors[0] / errors[1]) / np.log(2.0)
    order_two = np.log(errors[1] / errors[2]) / np.log(2.0)
    assert errors[-1] < 0.005
    assert order_one > 1.8
    assert order_two > 1.8


def test_solver_is_exactly_covariant_under_quarter_turn():
    original, _, source, sigma, _, _ = manufactured(33)
    rotated = solve_projected_tensor_aqual(
        np.rot90(source),
        2.0 / 32.0,
        np.zeros_like(source),
        np.rot90(sigma),
        np.zeros_like(source),
        np.ones_like(source),
        a0=1.0,
        mu_function=constant_mu,
    )
    relative = float(
        np.sqrt(np.mean(np.square(rotated.potential - np.rot90(original.potential))))
        / np.sqrt(np.mean(np.square(original.potential)))
    )
    assert relative < 1e-10


def test_solver_is_invariant_to_transport_direction_reversal():
    original, _, source, sigma, direction_x, direction_y = manufactured(33)
    reversed_direction = solve_projected_tensor_aqual(
        source,
        2.0 / 32.0,
        np.zeros_like(source),
        sigma,
        -direction_x,
        -direction_y,
        a0=1.0,
        mu_function=constant_mu,
    )
    assert np.allclose(
        original.potential, reversed_direction.potential, rtol=1e-10, atol=1e-10
    )


def test_zero_anisotropy_removes_all_direction_dependence():
    _, _, source, _, direction_x, direction_y = manufactured(33)
    sigma = np.zeros_like(source)
    first = solve_projected_tensor_aqual(
        source,
        2.0 / 32.0,
        np.zeros_like(source),
        sigma,
        direction_x,
        direction_y,
        a0=0.15,
    )
    second = solve_projected_tensor_aqual(
        source,
        2.0 / 32.0,
        np.zeros_like(source),
        sigma,
        -direction_y,
        direction_x,
        a0=0.15,
    )
    relative = float(
        np.sqrt(np.mean(np.square(first.potential - second.potential)))
        / np.sqrt(np.mean(np.square(first.potential)))
    )
    assert relative < 1e-10


def test_nonlinear_tensor_aqual_converges_with_positive_eigenvalue():
    cells = 49
    axis = np.linspace(-2.0, 2.0, cells)
    spacing = float(axis[1] - axis[0])
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    source = np.exp(-0.5 * ((xx / 0.45) ** 2 + (yy / 0.7) ** 2))
    sigma = 0.25 * np.exp(-0.5 * (xx**2 + yy**2) / 1.2**2)
    direction_x = np.cos(0.35 * yy)
    direction_y = np.sin(0.35 * yy)
    solution = solve_projected_tensor_aqual(
        source,
        spacing,
        np.zeros_like(source),
        sigma,
        direction_x,
        direction_y,
        a0=0.15,
    )
    assert solution.converged is True
    assert solution.normalized_residual_rms < 1e-5
    assert solution.metadata["minimum_constitutive_eigenvalue"] > 0.0
    assert all(value == 0 for value in solution.metadata["linear_solver_information"])
