from __future__ import annotations

import numpy as np

from voidscreen.field_solvers import (
    acceleration_magnitude,
    cell_coordinates,
    radial_boundary_from_acceleration,
    simple_mond_acceleration,
    solve_aqual,
    solve_newtonian,
    solve_poisson_dirichlet,
    solve_qumond,
    surface_density_to_volume,
)


def plummer_fixture(cells: int = 41, half_width: float = 12.0):
    spacing = 2.0 * half_width / (cells - 1)
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    radius = np.sqrt(x * x + y * y + z * z)
    mass = 1.0
    scale = 1.0
    density = 3.0 * mass * scale**2 / (4.0 * np.pi * (radius**2 + scale**2) ** 2.5)
    newtonian_potential = -mass / np.sqrt(radius**2 + scale**2)
    newtonian_boundary = np.where(
        (np.indices(radius.shape) == 0).any(axis=0)
        | (np.indices(radius.shape) == cells - 1).any(axis=0),
        newtonian_potential,
        0.0,
    )
    return spacing, radius, density, newtonian_boundary


def radial_profile(values: np.ndarray) -> np.ndarray:
    center = tuple((count - 1) // 2 for count in values.shape)
    return values[center[0] :, center[1], center[2]]


def test_surface_density_lift_preserves_every_column():
    surface = np.arange(1.0, 21.0).reshape(4, 5)
    z = np.linspace(-4.0, 4.0, 81)
    volume = surface_density_to_volume(surface, z, scale_height=0.7)
    reconstructed = np.sum(volume, axis=2) * (z[1] - z[0])
    assert np.allclose(reconstructed, surface, rtol=1e-12, atol=1e-12)


def test_dirichlet_poisson_solver_is_exact_for_a_quadratic_field():
    shape = (17, 19, 21)
    spacing = (0.3, 0.4, 0.5)
    x, y, z = cell_coordinates(shape, spacing)
    exact = x * x + 2.0 * y * y + 3.0 * z * z
    source = np.full(shape, 12.0)
    boundary = np.zeros(shape)
    edge = np.zeros(shape, dtype=bool)
    edge[[0, -1], :, :] = True
    edge[:, [0, -1], :] = True
    edge[:, :, [0, -1]] = True
    boundary[edge] = exact[edge]
    solved = solve_poisson_dirichlet(source, spacing, boundary)
    assert np.max(np.abs(solved - exact)) < 2e-12


def test_newtonian_solver_recovers_plummer_force_and_equation():
    spacing, radius, density, boundary = plummer_fixture(cells=65)
    solution = solve_newtonian(
        density,
        spacing,
        gravitational_constant=1.0,
        boundary_potential=boundary,
    )
    measured = radial_profile(acceleration_magnitude(solution.acceleration))
    radial = radial_profile(radius)
    expected = radial / np.power(radial * radial + 1.0, 1.5)
    valid = (radial >= 2.0 * spacing) & (radial <= 7.0)
    relative = np.abs(measured[valid] / expected[valid] - 1.0)
    assert np.median(relative) < 0.015
    assert np.quantile(relative, 0.95) < 0.03
    assert solution.normalized_residual_rms < 1e-10


def test_qumond_solves_second_field_equation_for_spherical_source():
    spacing, radius, density, newtonian_boundary = plummer_fixture(cells=65)
    a0 = 0.03

    def expected_acceleration(radial_distance):
        newtonian = radial_distance / np.power(radial_distance**2 + 1.0, 1.5)
        return simple_mond_acceleration(newtonian, a0)

    mond_boundary = radial_boundary_from_acceleration(
        density.shape,
        spacing,
        expected_acceleration,
    )
    solution = solve_qumond(
        density,
        spacing,
        a0=a0,
        gravitational_constant=1.0,
        newtonian_boundary=newtonian_boundary,
        mond_boundary=mond_boundary,
    )
    measured = radial_profile(acceleration_magnitude(solution.acceleration))
    radial = radial_profile(radius)
    expected = expected_acceleration(radial)
    valid = (radial >= 3.0 * spacing) & (radial <= 6.0)
    relative = np.abs(measured[valid] / expected[valid] - 1.0)
    assert np.median(relative) < 0.02
    assert solution.normalized_residual_rms < 1e-10
    assert solution.converged


def test_aqual_solves_nonlinear_field_equation_for_spherical_source():
    spacing, radius, density, _ = plummer_fixture(cells=49)
    a0 = 0.03

    def expected_acceleration(radial_distance):
        newtonian = radial_distance / np.power(radial_distance**2 + 1.0, 1.5)
        return simple_mond_acceleration(newtonian, a0)

    mond_boundary = radial_boundary_from_acceleration(
        density.shape,
        spacing,
        expected_acceleration,
    )
    solution = solve_aqual(
        density,
        spacing,
        a0=a0,
        gravitational_constant=1.0,
        boundary_potential=mond_boundary,
        residual_tolerance=1e-5,
        maximum_nonlinear_iterations=60,
    )
    measured = radial_profile(acceleration_magnitude(solution.acceleration))
    radial = radial_profile(radius)
    expected = expected_acceleration(radial)
    valid = (radial >= 3.0 * spacing) & (radial <= 6.0)
    relative = np.abs(measured[valid] / expected[valid] - 1.0)
    assert solution.converged
    assert solution.normalized_residual_rms < 1e-5
    assert np.median(relative) < 0.02
