from __future__ import annotations

import numpy as np

from voidscreen.field_solvers import cell_coordinates, solve_newtonian
from voidscreen.radial_path_potential import (
    hybrid_path_routing_potential,
    normalized_acceleration_curl,
    radial_path_potential_from_newtonian,
)


def test_radial_path_potential_is_finite_curl_free_and_hybrid_is_exact():
    cells = 17
    spacing = 0.5
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 2.0 + y * y / 1.2 + z * z / 0.8))
    density /= np.sum(density) * spacing**3
    newtonian = solve_newtonian(density, spacing, gravitational_constant=1.0)
    path = radial_path_potential_from_newtonian(
        density,
        newtonian.potential,
        newtonian.acceleration,
        spacing,
        a0=0.03,
        quadrature_order=8,
        interpolation_order=1,
    )
    assert np.all(np.isfinite(path.potential))
    assert all(np.all(np.isfinite(component)) for component in path.acceleration)
    assert normalized_acceleration_curl(path.acceleration, spacing) < 1e-10
    local = path.potential + 0.1 * x
    routed = path.potential - 0.2 * y
    hybrid = hybrid_path_routing_potential(path, local, routed, spacing, 0.25)
    assert np.array_equal(
        hybrid.potential,
        path.potential + 0.25 * (routed - local),
    )
    assert normalized_acceleration_curl(hybrid.acceleration, spacing) < 1e-10


def test_radial_path_potential_rejects_invalid_numerics():
    density = np.ones((9, 9, 9))
    potential = np.zeros_like(density)
    acceleration = (np.zeros_like(density),) * 3
    with np.testing.assert_raises(ValueError):
        radial_path_potential_from_newtonian(
            density,
            potential,
            acceleration,
            1.0,
            a0=1.0,
            quadrature_order=3,
        )
    with np.testing.assert_raises(ValueError):
        radial_path_potential_from_newtonian(
            density,
            potential,
            acceleration,
            1.0,
            a0=1.0,
            interpolation_order=2,
        )
