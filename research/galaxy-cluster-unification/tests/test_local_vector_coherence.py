from __future__ import annotations

import numpy as np

from voidscreen.coherent_monopole import coherent_monopole_potential
from voidscreen.field_solvers import cell_coordinates, laplacian, solve_newtonian
from voidscreen.local_vector_coherence import (
    baryonic_vector_coherence,
    base_boundary_relative_mismatch,
    coherence_gated_source_potential,
    hybrid_coherence_routing_potential,
)
from voidscreen.radial_path_potential import normalized_acceleration_curl


def test_pairwise_coherence_obeys_bounds_and_cancels_between_equal_sources():
    cells = 15
    density = np.zeros((cells,) * 3)
    middle = cells // 2
    density[middle - 2, middle, middle] = 1.0
    density[middle + 2, middle, middle] = 1.0
    solution = baryonic_vector_coherence(
        density,
        1.0,
        gravitational_constant=1.0,
    )
    assert np.min(solution.coherence) >= 0.0
    assert np.max(solution.coherence) <= 1.0
    assert solution.maximum_triangle_inequality_excess < 1e-12
    assert solution.coherence[middle, middle, middle] < 1e-12
    assert solution.coherence[middle + 5, middle, middle] > 0.9


def test_gated_source_endpoints_boundary_curl_and_hybrid_identity():
    cells = 15
    spacing = 0.5
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 1.2 + y * y / 0.9 + z * z / 0.7))
    density /= np.sum(density) * spacing**3
    newtonian = solve_newtonian(density, spacing, gravitational_constant=1.0)
    coherent = coherent_monopole_potential(
        density,
        newtonian.potential,
        newtonian.acceleration,
        spacing,
        a0=0.03,
    )
    local_source = laplacian(coherent.potential + 0.01 * x * x, spacing)
    endpoint = coherence_gated_source_potential(
        coherent,
        local_source,
        np.ones_like(density),
        spacing,
    )
    interior = (slice(1, -1),) * 3
    assert np.allclose(
        endpoint.potential[interior],
        coherent.potential[interior],
        rtol=1e-11,
        atol=1e-11,
    )
    assert base_boundary_relative_mismatch(endpoint, coherent) == 0.0
    assert normalized_acceleration_curl(endpoint.acceleration, spacing) < 1e-10
    local_potential = endpoint.potential + 0.1 * x
    routed_potential = endpoint.potential - 0.2 * y
    hybrid = hybrid_coherence_routing_potential(
        endpoint,
        local_potential,
        routed_potential,
        spacing,
        0.25,
    )
    assert np.array_equal(
        hybrid.potential,
        endpoint.potential + 0.25 * (routed_potential - local_potential),
    )
    assert normalized_acceleration_curl(hybrid.acceleration, spacing) < 1e-10


def test_vector_coherence_rejects_invalid_numerics():
    density = np.ones((9, 9, 9))
    with np.testing.assert_raises(ValueError):
        baryonic_vector_coherence(
            density,
            (1.0, 1.0, 1.1),
            gravitational_constant=1.0,
        )
    with np.testing.assert_raises(ValueError):
        baryonic_vector_coherence(
            density,
            1.0,
            gravitational_constant=0.0,
        )
