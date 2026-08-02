from __future__ import annotations

import numpy as np

from voidscreen.field_solvers import (
    boundary_mask,
    cell_coordinates,
    solve_qumond,
)
from voidscreen.spatial_qumond_3d import (
    divergence_powered_qumond_gradient,
    path_qumond_monopole_boundary,
    solve_path_diluted_qumond,
)


def compact_density(cells: int = 21):
    spacing = 0.8
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 2.0 + y * y / 1.4 + z * z / 0.8))
    density /= np.sum(density) * spacing**3
    return spacing, density


def rar_nu(y):
    values = np.asarray(y, dtype=float)
    return 1.0 / (-np.expm1(-np.sqrt(values)))


def test_zero_extra_channels_exactly_reduces_to_fixed_rar_qumond():
    spacing, density = compact_density()
    a0 = 0.03
    boundary = path_qumond_monopole_boundary(
        density,
        spacing,
        gravitational_constant=1.0,
        a0=a0,
        transition_depth=1e-6,
        transition_power=4.0,
        extra_spatial_channels=0.0,
        path_power=0.5,
        light_speed=1000.0,
    )
    candidate = solve_path_diluted_qumond(
        density,
        spacing,
        gravitational_constant=1.0,
        a0=a0,
        transition_depth=1e-6,
        transition_power=4.0,
        extra_spatial_channels=0.0,
        path_power=0.5,
        light_speed=1000.0,
        modified_boundary=boundary,
    )
    control = solve_qumond(
        density,
        spacing,
        gravitational_constant=1.0,
        a0=a0,
        mond_boundary=boundary,
        nu_function=rar_nu,
    )
    assert np.allclose(candidate.field.equation_source, control.equation_source)
    assert np.allclose(candidate.field.potential, control.potential)
    assert np.allclose(candidate.channel_exponent, 1.0)
    assert candidate.field.normalized_residual_rms < 1e-10


def test_powered_face_flux_is_covariant_under_axis_permutation():
    spacing, density = compact_density(cells=17)
    solved = solve_path_diluted_qumond(
        density,
        spacing,
        gravitational_constant=1.0,
        a0=0.03,
        transition_depth=1e-4,
        transition_power=4.0,
        extra_spatial_channels=2.0,
        path_power=0.5,
        light_speed=1000.0,
    )
    source = divergence_powered_qumond_gradient(
        solved.newtonian.potential,
        solved.channel_exponent,
        spacing,
        a0=0.03,
    )
    permuted = divergence_powered_qumond_gradient(
        np.transpose(solved.newtonian.potential, (1, 0, 2)),
        np.transpose(solved.channel_exponent, (1, 0, 2)),
        spacing,
        a0=0.03,
    )
    assert np.allclose(permuted, np.transpose(source, (1, 0, 2)), rtol=1e-12, atol=1e-12)


def test_locked_solution_is_finite_bounded_and_honors_boundary():
    spacing, density = compact_density()
    solution = solve_path_diluted_qumond(
        density,
        spacing,
        gravitational_constant=1.0,
        a0=0.03,
        transition_depth=1e-4,
        transition_power=4.0,
        extra_spatial_channels=2.0,
        path_power=0.5,
        light_speed=1000.0,
    )
    edge = boundary_mask(density.shape)
    assert solution.field.converged
    assert solution.field.normalized_residual_rms < 1e-10
    assert np.all(np.isfinite(solution.potential_path_ratio))
    assert float(np.min(solution.channel_exponent)) >= 1.0
    assert float(np.max(solution.channel_exponent)) <= 3.0
    assert np.array_equal(
        solution.field.potential[edge],
        solution.boundary_potential[edge],
    )
