from __future__ import annotations

import numpy as np
import pytest

from voidscreen.observational_resampling import common_resolution_surface_density


def test_common_resolution_conserves_physical_surface_integral():
    axis = np.linspace(-4.0, 4.0, 65)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    surface = np.exp(-0.5 * ((xx / 0.7) ** 2 + (yy / 1.1) ** 2))
    result = common_resolution_surface_density(surface, 33)
    assert result.downsampling_ratio == 2.0
    assert np.isclose(result.added_native_gaussian_sigma_pixels, np.sqrt(3.0) / 2.0)
    assert result.filtered_mass_relative_error < 1e-14
    assert result.coarse_mass_relative_error < 1e-14
    assert np.isclose(
        np.sum(result.filtered_native),
        np.sum(result.coarse) * result.downsampling_ratio**2,
        rtol=1e-14,
    )


def test_common_resolution_suppresses_unresolved_checkerboard_alias():
    rows, columns = np.indices((65, 65))
    checkerboard = 1.0 + 0.9 * ((rows + columns) % 2)
    result = common_resolution_surface_density(checkerboard, 33)
    unfiltered_sample = checkerboard[::2, ::2]
    central = result.coarse[3:-3, 3:-3]
    true_mean = float(np.mean(checkerboard))
    assert abs(float(np.mean(central)) - true_mean) < 0.01
    assert abs(float(np.mean(central)) - true_mean) < abs(
        float(np.mean(unfiltered_sample)) - true_mean
    )


def test_invalid_common_resolution_requests_are_rejected():
    surface = np.ones((65, 65))
    with pytest.raises(ValueError):
        common_resolution_surface_density(surface, 65)
    with pytest.raises(ValueError):
        common_resolution_surface_density(surface, 32)
