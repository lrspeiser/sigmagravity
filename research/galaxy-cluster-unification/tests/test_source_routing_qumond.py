from __future__ import annotations

import numpy as np

from voidscreen.field_solvers import boundary_mask, cell_coordinates
from voidscreen.source_routing_qumond import (
    normalized_baryonic_quadrupole,
    solve_multipole_gated_source_routing,
    solve_source_conserving_baryonic_routing,
)


def test_source_routing_conserves_added_source_and_honors_boundary():
    cells = 21
    spacing = 0.8
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 2.0 + y * y / 1.4 + z * z / 0.8))
    density /= np.sum(density) * spacing**3
    solution = solve_source_conserving_baryonic_routing(
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
    positive = float(np.sum(solution.positive_routed_source) * spacing**3)
    negative = float(np.sum(solution.negative_shell_source) * spacing**3)
    net = float(
        np.sum(solution.positive_routed_source - solution.negative_shell_source) * spacing**3
    )
    edge = boundary_mask(density.shape)
    assert solution.field.converged
    assert solution.field.normalized_residual_rms < 1e-10
    assert solution.positive_generator_strength > 0.0
    assert np.isclose(positive, solution.positive_generator_strength, rtol=1e-13)
    assert np.isclose(negative, solution.positive_generator_strength, rtol=1e-13)
    assert abs(net) / solution.positive_generator_strength < 1e-13
    assert np.array_equal(solution.field.potential[edge], solution.boundary_potential[edge])


def test_normalized_quadrupole_has_sphere_and_line_limits():
    cells = 41
    spacing = 0.4
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    sphere = np.exp(-(x * x + y * y + z * z))
    line = np.exp(-(x * x / 16.0 + y * y / 0.02 + z * z / 0.02))
    sphere_q, _ = normalized_baryonic_quadrupole(sphere, spacing)
    line_q, _ = normalized_baryonic_quadrupole(line, spacing)
    assert sphere_q < 1e-12
    assert line_q > 0.98


def test_multipole_gated_source_is_exact_declared_linear_mixture():
    cells = 17
    spacing = 0.8
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 2.0 + y * y / 1.4 + z * z / 0.8))
    density /= np.sum(density) * spacing**3
    solution = solve_multipole_gated_source_routing(
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
    expected = (
        (1.0 - solution.quadrupole_fraction) * solution.routing.local_generator_source
        + solution.quadrupole_fraction * solution.routing.routed_source
    )
    assert 0.0 <= solution.quadrupole_fraction <= 1.0
    assert np.array_equal(solution.mixed_source, expected)
    assert solution.field.converged
    assert solution.field.normalized_residual_rms < 1e-10
