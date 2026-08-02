from __future__ import annotations

import numpy as np
import pytest

from voidscreen.accumulated_lensing import zero_pad_square_component_maps


def test_zero_padding_preserves_map_values_mass_and_spacing():
    axis = np.arange(-2.0, 3.0, 1.0)
    stars = np.arange(25.0).reshape(5, 5)
    gas = np.flipud(stars)
    padded_axis, (padded_stars, padded_gas) = zero_pad_square_component_maps(
        axis, stars, gas, padding_cells=3
    )
    assert padded_axis[0] == -5.0
    assert padded_axis[-1] == 5.0
    assert np.allclose(np.diff(padded_axis), 1.0)
    assert np.array_equal(padded_stars[3:-3, 3:-3], stars)
    assert np.array_equal(padded_gas[3:-3, 3:-3], gas)
    assert np.sum(padded_stars) == np.sum(stars)
    assert np.sum(padded_gas) == np.sum(gas)
    assert np.count_nonzero(padded_stars[:3]) == 0
    assert np.count_nonzero(padded_gas[:, -3:]) == 0


def test_zero_padding_rejects_incompatible_inputs():
    axis = np.arange(5.0)
    square = np.ones((5, 5))
    with pytest.raises(ValueError, match="non-negative integer"):
        zero_pad_square_component_maps(axis, square, padding_cells=-1)
    with pytest.raises(ValueError, match="match the square axis"):
        zero_pad_square_component_maps(axis, np.ones((4, 4)), padding_cells=1)
