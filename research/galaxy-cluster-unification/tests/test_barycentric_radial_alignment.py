from __future__ import annotations

import numpy as np

from voidscreen.barycentric_radial_alignment import (
    barycentric_radial_alignment,
    vector_radial_alignment,
)
from voidscreen.field_solvers import cell_coordinates, solve_newtonian


def test_explicit_radial_and_tangential_limits():
    cells = 9
    x, y, z = cell_coordinates((cells,) * 3, 1.0)
    radius = np.sqrt(x * x + y * y + z * z)
    radial = tuple(
        np.divide(component, radius, out=np.zeros_like(component), where=radius > 0.0)
        for component in (x, y, z)
    )
    inward, _, _ = vector_radial_alignment((x, y, z), tuple(-item for item in radial))
    outward, _, _ = vector_radial_alignment((x, y, z), radial)
    tangential_vector = (-radial[1], radial[0], np.zeros_like(x))
    tangential, _, _ = vector_radial_alignment((x, y, z), tangential_vector)
    active = radius > 0.0
    assert np.allclose(inward[active], 1.0, rtol=0.0, atol=1e-15)
    assert np.max(outward) == 0.0
    assert np.max(tangential) < 1e-15
    assert inward[cells // 2, cells // 2, cells // 2] == 0.0


def test_barycentric_alignment_is_finite_bounded_and_rotation_covariant():
    cells = 15
    spacing = 0.5
    x, y, z = cell_coordinates((cells,) * 3, spacing)
    density = np.exp(-(x * x / 1.5 + y * y / 0.8 + z * z / 0.6))
    density /= np.sum(density) * spacing**3
    newtonian = solve_newtonian(density, spacing, gravitational_constant=1.0)
    solution = barycentric_radial_alignment(
        density,
        newtonian.acceleration,
        spacing,
    )
    rotated_density = np.swapaxes(density, 0, 1)
    rotated_newtonian = solve_newtonian(
        rotated_density,
        spacing,
        gravitational_constant=1.0,
    )
    rotated = barycentric_radial_alignment(
        rotated_density,
        rotated_newtonian.acceleration,
        spacing,
    )
    assert np.all(np.isfinite(solution.alignment))
    assert np.min(solution.alignment) >= 0.0
    assert np.max(solution.alignment) <= 1.0
    assert np.allclose(
        rotated.alignment,
        np.swapaxes(solution.alignment, 0, 1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_barycentric_alignment_rejects_invalid_fields():
    density = np.ones((9, 9, 9))
    acceleration = (np.zeros_like(density),) * 3
    with np.testing.assert_raises(ValueError):
        barycentric_radial_alignment(
            density,
            acceleration[:2],
            1.0,
        )
    with np.testing.assert_raises(ValueError):
        barycentric_radial_alignment(
            -density,
            acceleration,
            1.0,
        )
