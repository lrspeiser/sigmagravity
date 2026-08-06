import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_sigma_v19cy_a2319_response_inputs as preparation


def test_frozen_inputs_validate_and_sealed_assets_stay_closed():
    config = preparation.validate_config()
    assert config["authorization"]["access_A3667_validation"] is False
    assert config["authorization"]["access_A754_holdout"] is False


def test_interval_clipping_and_intersection_are_exact():
    parent = np.asarray([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    clipped = preparation.clip_intervals(parent, 5.0, 45.0)
    np.testing.assert_array_equal(clipped, [[5.0, 10.0], [20.0, 30.0], [40.0, 45.0]])
    environment = np.asarray([[8.0, 22.0], [25.0, 42.0]])
    np.testing.assert_array_equal(
        preparation.intersect_intervals(clipped, environment),
        [[8.0, 10.0], [20.0, 22.0], [25.0, 30.0], [40.0, 42.0]],
    )


def test_touching_and_overlapping_intervals_normalize_without_zero_width_rows():
    normalized = preparation.normalize_intervals(
        np.asarray([5.0, 0.0, 10.0, 30.0]),
        np.asarray([12.0, 5.0, 20.0, 30.0]),
    )
    np.testing.assert_array_equal(normalized, [[0.0, 20.0]])


def test_event_selection_includes_closed_gti_endpoints():
    times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    selected = preparation.select_event_rows(times, np.asarray([[1.0, 2.0], [4.0, 5.0]]))
    np.testing.assert_array_equal(selected, [False, True, True, False, True])


def test_pixlist_uses_hyphenated_contiguous_runs():
    assert preparation.compress_pixlist([0, 1, 2, 4, 7, 8]) == "0-2,4,7-8"
