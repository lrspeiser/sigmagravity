import numpy as np

from voidscreen.baryon_morphology import (
    blend_unit_directions,
    map_attraction_directions,
)


def test_map_attraction_points_toward_offset_peak_and_normalizes():
    axis = np.linspace(-5.0, 5.0, 21)
    surface = np.zeros((21, 21))
    surface[10, 16] = 3.0
    directions, audit = map_attraction_directions(
        axis, surface, [[0.0, 0.0], [0.0, 1.0]], softening=1.0
    )
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions[0, 0] > 0.99
    assert directions[1, 0] > 0.9
    assert directions[1, 1] < 0.0
    assert audit["normalization_error"] < 1e-14


def test_direction_blend_respects_endpoints():
    first = np.array([[1.0, 0.0], [0.0, 1.0]])
    second = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(blend_unit_directions(first, second, 0.0), first)
    assert np.allclose(blend_unit_directions(first, second, 1.0), second)
    middle = blend_unit_directions(first, second, 0.5)
    assert np.allclose(middle, np.sqrt(0.5))
