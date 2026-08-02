from __future__ import annotations

import numpy as np

from voidscreen.metric_lensing_3d import (
    exact_tensor_activation_3d,
    lift_surface_density_msun_kpc2_to_si_volume,
    normalized_deflection_curl,
    photon_deflection_zero_slip,
)


def test_surface_lift_preserves_physical_column_density():
    axis = np.linspace(-2.0, 2.0, 17)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    surface = np.exp(-0.5 * ((xx / 0.4) ** 2 + (yy / 0.7) ** 2))
    z = np.linspace(-4.0, 4.0, 33)
    volume, scale = lift_surface_density_msun_kpc2_to_si_volume(
        surface,
        z,
        cell_kpc=axis[1] - axis[0],
    )
    assert scale > 0.0
    reconstructed = np.sum(volume, axis=2) * (z[1] - z[0]) * 3.085677581491367e19
    expected = surface * 1.98847e30 / 3.085677581491367e19**2
    assert np.allclose(reconstructed, expected, rtol=1e-12, atol=0.0)


def test_zero_slip_deflection_is_linear_rotation_covariant_and_curl_free():
    cells = 17
    axis = np.linspace(-2.0, 2.0, cells)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    potential = 0.5 * (x * x + 2.0 * y * y + 0.3 * z * z)
    acceleration = tuple(-component for component in np.gradient(potential, axis[1] - axis[0]))
    first = photon_deflection_zero_slip(acceleration, axis[1] - axis[0], light_speed=1.0)
    doubled = photon_deflection_zero_slip(
        tuple(2.0 * component for component in acceleration),
        axis[1] - axis[0],
        light_speed=1.0,
    )
    assert np.allclose(doubled.alpha_x_radian, 2.0 * first.alpha_x_radian)
    assert normalized_deflection_curl(
        first.alpha_x_radian,
        first.alpha_y_radian,
        axis[1] - axis[0],
    ) < 1e-12


def test_three_dimensional_activation_is_bounded_and_positive():
    cells = 17
    axis = np.linspace(-4.0, 4.0, cells)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    stars = np.exp(-0.5 * (((x + 0.5) / 0.6) ** 2 + (y / 0.7) ** 2 + (z / 0.8) ** 2))
    gas = np.exp(-0.5 * (((x - 0.7) / 1.0) ** 2 + (y / 1.1) ** 2 + (z / 1.2) ** 2))
    activation = exact_tensor_activation_3d(
        stars,
        gas,
        axis[1] - axis[0],
        gravitational_constant=1.0,
        a0=0.1,
        coherence_length=1.0,
    )
    assert np.min(activation.sigma) >= 0.0
    assert np.max(activation.sigma) < 1.0
    assert np.min(activation.minimum_eigenvalue_proxy) > 0.0
    assert all(
        np.allclose(component**2, component**2)
        for component in activation.transport_direction
    )
