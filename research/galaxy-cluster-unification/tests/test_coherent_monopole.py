from __future__ import annotations

import numpy as np

from voidscreen.coherent_monopole import (
    coherent_monopole_potential,
    hybrid_coherent_routing_potential,
)
from voidscreen.field_solvers import cell_coordinates, solve_newtonian
from voidscreen.radial_path_potential import normalized_acceleration_curl


def test_coherent_monopole_is_finite_curl_free_radial_and_hybrid_is_exact():
    cells = 17
    spacing = 0.5
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 2.0 + y * y / 1.2 + z * z / 0.8))
    density /= np.sum(density) * spacing**3
    newtonian = solve_newtonian(density, spacing, gravitational_constant=1.0)
    coherent = coherent_monopole_potential(
        density,
        newtonian.potential,
        newtonian.acceleration,
        spacing,
        a0=0.03,
    )
    assert np.all(np.isfinite(coherent.potential))
    assert all(np.all(np.isfinite(component)) for component in coherent.acceleration)
    assert np.array_equal(
        coherent.potential,
        newtonian.potential + coherent.correction_potential,
    )
    assert np.all(coherent.coherent_acceleration_correction >= 0.0)
    assert normalized_acceleration_curl(coherent.acceleration, spacing) < 1e-10
    local = coherent.potential + 0.1 * x
    routed = coherent.potential - 0.2 * y
    hybrid = hybrid_coherent_routing_potential(
        coherent,
        local,
        routed,
        spacing,
        0.25,
    )
    assert np.array_equal(
        hybrid.potential,
        coherent.potential + 0.25 * (routed - local),
    )
    assert normalized_acceleration_curl(hybrid.acceleration, spacing) < 1e-10


def test_coherent_monopole_rejects_anisotropic_grid_and_invalid_fraction():
    density = np.ones((9, 9, 9))
    potential = np.zeros_like(density)
    acceleration = (np.zeros_like(density),) * 3
    with np.testing.assert_raises(ValueError):
        coherent_monopole_potential(
            density,
            potential,
            acceleration,
            (1.0, 1.0, 1.1),
            a0=1.0,
        )
    coherent = coherent_monopole_potential(
        density,
        potential,
        acceleration,
        1.0,
        a0=1.0,
    )
    with np.testing.assert_raises(ValueError):
        hybrid_coherent_routing_potential(
            coherent,
            potential,
            potential,
            1.0,
            1.1,
        )
