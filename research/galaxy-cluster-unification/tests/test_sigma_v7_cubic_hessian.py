from __future__ import annotations

import numpy as np
import pytest

from voidscreen.field_solvers import cell_coordinates
from voidscreen.sigma_v7_cubic_hessian import (
    cubic_hessian_invariant,
    cubic_hessian_operator,
    ellipticity_eigenvalues,
    hessian_components,
    solve_cubic_hessian_dirichlet,
    temporal_kinetic_coefficient,
)


def test_quadratic_spherical_solution_has_exact_hessian_and_source() -> None:
    shape = (11, 11, 11)
    spacing = 0.5
    x, y, z = cell_coordinates(shape, spacing)
    amplitude = 0.02
    potential = amplitude * (x**2 + y**2 + z**2)
    components = hessian_components(potential, spacing)
    interior = (slice(1, -1),) * 3
    for diagonal in components[:3]:
        assert np.max(np.abs(diagonal[interior] - 2.0 * amplitude)) < 1.0e-13
    for cross in components[3:]:
        assert np.max(np.abs(cross[interior])) < 1.0e-13
    expected_invariant = 24.0 * amplitude**2
    expected_source = 18.0 * amplitude + expected_invariant
    assert np.max(
        np.abs(cubic_hessian_invariant(potential, spacing)[interior] - expected_invariant)
    ) < 1.0e-13
    assert np.max(
        np.abs(cubic_hessian_operator(potential, spacing)[interior] - expected_source)
    ) < 3.0e-13


def test_solver_recovers_analytic_spherical_manufactured_solution() -> None:
    shape = (11, 11, 11)
    spacing = 0.5
    x, y, z = cell_coordinates(shape, spacing)
    amplitude = 0.02
    expected = amplitude * (x**2 + y**2 + z**2)
    source = np.full(shape, 18.0 * amplitude + 24.0 * amplitude**2)
    solution = solve_cubic_hessian_dirichlet(
        source,
        spacing,
        expected,
        relaxation=0.2,
        tolerance=1.0e-10,
        max_iterations=1000,
    )
    interior = (slice(1, -1),) * 3
    assert solution.converged
    assert solution.residual_rms < 1.0e-10
    assert np.max(np.abs(solution.potential[interior] - expected[interior])) < 2.0e-11
    assert solution.minimum_temporal_kinetic_coefficient > 0.0
    assert solution.minimum_ellipticity_eigenvalue > 0.0


def test_separated_sources_are_nonadditive_on_a_healthy_branch() -> None:
    shape = (11, 11, 11)
    spacing = 0.5
    x, y, z = cell_coordinates(shape, spacing)
    boundary = np.zeros(shape)

    def source(center: float) -> np.ndarray:
        return 30.0 * np.exp(
            -((x - center) ** 2 + y**2 + z**2) / (2.0 * 0.5**2)
        )

    first_source = source(-0.75)
    second_source = source(0.75)
    options = {
        "relaxation": 0.005,
        "tolerance": 1.0e-4,
        "max_iterations": 5000,
    }
    first = solve_cubic_hessian_dirichlet(first_source, spacing, boundary, **options)
    second = solve_cubic_hessian_dirichlet(second_source, spacing, boundary, **options)
    combined = solve_cubic_hessian_dirichlet(
        first_source + second_source, spacing, boundary, **options
    )
    interior = (slice(1, -1),) * 3
    difference = combined.potential[interior] - (
        first.potential[interior] + second.potential[interior]
    )
    relative_nonadditivity = float(
        np.sqrt(np.mean(difference**2))
        / np.sqrt(np.mean(combined.potential[interior] ** 2))
    )
    assert first.converged and second.converged and combined.converged
    assert relative_nonadditivity > 0.05
    assert combined.minimum_temporal_kinetic_coefficient > 0.0
    assert combined.minimum_ellipticity_eigenvalue > 0.0


def test_perturbation_coefficients_match_quadratic_background() -> None:
    shape = (7, 7, 7)
    spacing = 0.5
    x, y, z = cell_coordinates(shape, spacing)
    amplitude = 0.03
    potential = amplitude * (x**2 + y**2 + z**2)
    temporal = temporal_kinetic_coefficient(potential, spacing)
    spatial = ellipticity_eigenvalues(potential, spacing)
    assert np.allclose(temporal[1:-1, 1:-1, 1:-1], 3.0 + 12.0 * amplitude)
    assert np.allclose(spatial, 3.0 + 8.0 * amplitude)


def test_invalid_cubic_hessian_inputs_are_rejected() -> None:
    grid = np.zeros((7, 7, 7))
    with pytest.raises(ValueError):
        cubic_hessian_operator(grid, 1.0, kappa=-1.0)
    with pytest.raises(ValueError):
        temporal_kinetic_coefficient(grid, 1.0, kappa=-1.0)
    with pytest.raises(ValueError):
        solve_cubic_hessian_dirichlet(grid, 1.0, grid, relaxation=0.0)
    with pytest.raises(ValueError):
        solve_cubic_hessian_dirichlet(grid, 1.0, grid, tolerance=0.0)
