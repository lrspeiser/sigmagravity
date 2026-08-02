from __future__ import annotations

import numpy as np
import pytest

from voidscreen.geometric_transport import thin_sheet_newtonian_field
from voidscreen.physical_tensor_activation import (
    exact_physical_tensor_activation,
    forward_boundary_distance_kpc,
    physical_tidal_length_kpc,
)


def gaussian_map(cells=65, offset=0.0, scale=0.4, mass=1.0e9):
    axis = np.linspace(-4.0, 4.0, cells)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values = np.exp(-0.5 * (((xx - offset) / scale) ** 2 + (yy / scale) ** 2))
    return axis, values * mass / (np.sum(values) * (axis[1] - axis[0]) ** 2)


def test_boundary_distance_uses_physical_direction_and_extent():
    direction_x = np.ones((9, 9))
    direction_y = np.zeros((9, 9))
    distance = forward_boundary_distance_kpc(direction_x, direction_y, 0.5)
    assert distance[4, 0] == 4.0
    assert distance[4, 4] == 2.0
    assert distance[4, -1] == 0.0


def test_physical_tidal_length_scales_with_coordinate_size():
    axis, surface = gaussian_map()
    first = physical_tidal_length_kpc(
        thin_sheet_newtonian_field(surface, axis[1] - axis[0]),
        axis[1] - axis[0],
    )
    factor = 3.0
    second = physical_tidal_length_kpc(
        thin_sheet_newtonian_field(surface, factor * (axis[1] - axis[0])),
        factor * (axis[1] - axis[0]),
    )
    active = first > 1e-8
    relative = np.median(np.abs(second[active] / (factor * first[active]) - 1.0))
    assert relative < 1e-10


def test_activation_is_bounded_elliptic_and_radial_null():
    axis, stars = gaussian_map(scale=0.35, mass=3.0e9)
    _, gas = gaussian_map(scale=0.7, mass=7.0e9)
    radial = exact_physical_tensor_activation(stars, gas, axis[1] - axis[0])
    assert np.max(radial.sigma[8:-8, 8:-8]) < 1e-9
    _, offset_gas = gaussian_map(offset=0.7, scale=0.7, mass=7.0e9)
    result = exact_physical_tensor_activation(stars, offset_gas, axis[1] - axis[0])
    assert np.min(result.sigma) >= 0.0
    assert np.max(result.sigma) < 1.0
    assert np.min(result.minimum_eigenvalue_proxy) > 0.0
    assert np.allclose(
        np.hypot(result.transport_direction_x, result.transport_direction_y),
        1.0,
    )


def test_invalid_physical_inputs_are_rejected():
    axis, stars = gaussian_map()
    with pytest.raises(ValueError):
        exact_physical_tensor_activation(stars, stars[:-1], axis[1] - axis[0])
    with pytest.raises(ValueError):
        forward_boundary_distance_kpc(np.ones((9, 9)), np.ones((8, 9)), 1.0)
