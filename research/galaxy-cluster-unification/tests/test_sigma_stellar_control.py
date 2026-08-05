from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_stellar_control import (
    cloud_in_cell_light_map,
    logical_pixels_to_common_kpc,
    region_light_percentile_ranks,
    smooth_light_draws,
)


def test_logical_pixel_inverse_matches_v19x4_coordinate_convention() -> None:
    east, north = logical_pixels_to_common_kpc(
        [99.0, 89.0],
        [199.0, 219.0],
        center_logical_x=100.0,
        center_logical_y=200.0,
        native_pixel_kpc=2.0,
    )
    assert east == pytest.approx([0.0, 20.0])
    assert north == pytest.approx([0.0, 40.0])


def test_cloud_in_cell_conserves_and_normalizes_light() -> None:
    axis = np.arange(-2.0, 3.0)
    result = cloud_in_cell_light_map(
        [-0.5, 0.25],
        [-0.5, 0.25],
        [1.0, 3.0],
        axis,
    )
    assert float(np.sum(result)) == pytest.approx(1.0, abs=1e-14)
    assert np.all(result >= 0.0)


def test_cic_rejects_members_without_four_grid_neighbors() -> None:
    with pytest.raises(ValueError, match="outside"):
        cloud_in_cell_light_map([2.0], [0.0], [1.0], np.arange(-2.0, 3.0))


def test_batch_smoothing_keeps_draws_separate_and_conserves_light() -> None:
    maps = np.zeros((2, 11, 11))
    maps[0, 3, 3] = 1.0
    maps[1, 7, 7] = 1.0
    smoothed = smooth_light_draws(maps, sigma_pixels=1.2)
    assert np.sum(smoothed, axis=(-2, -1)) == pytest.approx([1.0, 1.0])
    assert np.unravel_index(np.argmax(smoothed[0]), smoothed[0].shape) == (3, 3)
    assert np.unravel_index(np.argmax(smoothed[1]), smoothed[1].shape) == (7, 7)


def test_region_light_ranks_are_within_draw_and_filter_scale_invariant() -> None:
    labels = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]])
    maps = np.asarray(
        [
            [[1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 4.0, 4.0]],
            [[30.0, 30.0, 10.0, 10.0], [30.0, 30.0, 10.0, 10.0]],
        ]
    )
    means, ranks = region_light_percentile_ranks(maps, labels, [0, 1])
    np.testing.assert_allclose(means, [[1.0, 4.0], [30.0, 10.0]])
    np.testing.assert_allclose(ranks, [[0.25, 0.75], [0.75, 0.25]])
    _, rescaled = region_light_percentile_ranks(17.0 * maps, labels, [0, 1])
    np.testing.assert_allclose(rescaled, ranks)
