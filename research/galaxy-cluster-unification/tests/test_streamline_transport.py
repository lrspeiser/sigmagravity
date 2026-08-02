from __future__ import annotations

import numpy as np
import pytest

from voidscreen.geometric_transport import symmetric_streamline_average


def fields(cells=33):
    shape = (cells, cells)
    direction_x = np.ones(shape)
    direction_y = np.zeros(shape)
    length = np.full(shape, 8.0)
    return direction_x, direction_y, length


def test_constant_flux_is_fixed_point_even_at_boundaries():
    direction_x, direction_y, length = fields()
    flux_x = np.full(direction_x.shape, 2.0)
    flux_y = np.full(direction_x.shape, -0.5)
    averaged_x, averaged_y, audit = symmetric_streamline_average(
        flux_x, flux_y, direction_x, direction_y, length, steps=8
    )
    assert np.allclose(averaged_x, flux_x)
    assert np.allclose(averaged_y, flux_y)
    assert audit["transport_relative_change_RMS"] < 1e-15


def test_direction_reversal_leaves_symmetric_transport_unchanged():
    direction_x, direction_y, length = fields()
    rows, columns = np.indices(direction_x.shape)
    flux_x = np.sin(columns / 5.0) * np.exp(-np.square(rows - 16) / 80.0)
    flux_y = np.cos(rows / 7.0)
    forward = symmetric_streamline_average(
        flux_x, flux_y, direction_x, direction_y, length, steps=8
    )[:2]
    reversed_result = symmetric_streamline_average(
        flux_x, flux_y, -direction_x, -direction_y, length, steps=8
    )[:2]
    assert np.allclose(forward[0], reversed_result[0])
    assert np.allclose(forward[1], reversed_result[1])


def test_impulse_spreads_only_along_streamline():
    direction_x, direction_y, length = fields()
    flux_x = np.zeros_like(direction_x)
    flux_y = np.zeros_like(direction_y)
    flux_x[16, 16] = 1.0
    averaged_x, averaged_y, audit = symmetric_streamline_average(
        flux_x, flux_y, direction_x, direction_y, length, steps=8
    )
    assert np.count_nonzero(averaged_x[16]) > 1
    assert np.count_nonzero(averaged_x[:16]) == 0
    assert np.count_nonzero(averaged_x[17:]) == 0
    assert np.max(np.abs(averaged_y)) == 0.0
    assert audit["transport_relative_change_RMS"] > 0.0


def test_invalid_steps_are_rejected():
    direction_x, direction_y, length = fields()
    with pytest.raises(ValueError):
        symmetric_streamline_average(
            direction_x, direction_y, direction_x, direction_y, length, steps=1
        )
