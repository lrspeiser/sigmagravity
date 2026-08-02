from __future__ import annotations

import numpy as np
import pytest

from voidscreen.geometric_transport import aperture_weighted_statistics
from voidscreen.tensor_activation import (
    constitutive_tensor_components,
    exact_tensor_activation,
)


def gaussian_map(cells=33, offset=0.0, scale=0.4, mass=1.0e9):
    axis = np.linspace(-2.0, 2.0, cells)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values = np.exp(-0.5 * (((xx - offset) / scale) ** 2 + (yy / scale) ** 2))
    return axis, values * mass / (np.sum(values) * (axis[1] - axis[0]) ** 2)


def test_exact_activation_is_finite_bounded_and_elliptic():
    axis, stars = gaussian_map(offset=-0.35, scale=0.3, mass=3.0e9)
    _, gas = gaussian_map(offset=0.4, scale=0.65, mass=7.0e9)
    result = exact_tensor_activation(stars, gas, axis[1] - axis[0])
    assert np.all(np.isfinite(result.sigma))
    assert np.min(result.sigma) >= 0.0
    assert np.max(result.sigma) < 1.0
    assert np.min(result.minimum_eigenvalue_proxy) > 0.0
    assert np.allclose(
        np.hypot(result.transport_direction_x, result.transport_direction_y),
        1.0,
    )


def test_cocentered_radial_components_have_zero_transverse_activation():
    axis, stars = gaussian_map(scale=0.3, mass=3.0e9)
    _, gas = gaussian_map(scale=0.65, mass=7.0e9)
    result = exact_tensor_activation(stars, gas, axis[1] - axis[0])
    statistics = aperture_weighted_statistics(
        result.sigma,
        stars + gas,
        result.total_field.magnitude_m_s2,
        axis[1] - axis[0],
    )
    assert statistics["weighted_mean"] < 1e-10


def test_rotation_covariance_and_direction_reversal_invariance():
    axis, stars = gaussian_map(offset=-0.35, scale=0.3, mass=3.0e9)
    _, gas = gaussian_map(offset=0.4, scale=0.65, mass=7.0e9)
    result = exact_tensor_activation(stars, gas, axis[1] - axis[0])
    rotated = exact_tensor_activation(np.rot90(stars), np.rot90(gas), axis[1] - axis[0])
    assert np.allclose(result.sigma, np.rot90(rotated.sigma, -1), rtol=1e-12, atol=1e-12)
    direct = constitutive_tensor_components(
        result.sigma,
        result.transport_direction_x,
        result.transport_direction_y,
    )
    reversed_tensor = constitutive_tensor_components(
        result.sigma,
        -result.transport_direction_x,
        -result.transport_direction_y,
    )
    for first, second in zip(direct, reversed_tensor, strict=True):
        assert np.array_equal(first, second)


def test_invalid_maps_or_scales_are_rejected():
    axis, stars = gaussian_map()
    with pytest.raises(ValueError):
        exact_tensor_activation(stars, stars[:-1], axis[1] - axis[0])
    with pytest.raises(ValueError):
        exact_tensor_activation(stars, stars, 0.0)
