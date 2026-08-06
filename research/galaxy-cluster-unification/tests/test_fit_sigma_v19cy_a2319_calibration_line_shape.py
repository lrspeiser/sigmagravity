import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_sigma_v19cy_a2319_calibration_line_shape as line_shape


def test_frozen_line_shape_scope_and_seals_are_exact() -> None:
    config, application = line_shape.validate_inputs(line_shape.DEFAULT_CONFIG)
    assert config["line_template"]["rows"] == 8
    assert config["fit"]["bin_width_ev"] == 0.5
    assert config["candidate_selection"]["maximum_absolute_statistical_z_per_observable"] == 5.0
    assert application["terminal_gate_passed"]
    assert not application["cluster_sky_event_accessed"]


def test_expected_counts_are_positive_and_area_normalized() -> None:
    template = {
        "energy": np.asarray([5890.0, 5900.0]),
        "width": np.asarray([2.0, 2.0]),
        "area": np.asarray([0.4, 0.6]),
    }
    centers = np.arange(5860.0, 5930.0, 0.5)
    expected = line_shape.expected_counts(
        centers,
        0.5,
        template,
        np.asarray([0.2, 4.5, 10000.0, 1.0]),
    )
    assert np.isfinite(expected).all()
    assert (expected > 0).all()


def test_fit_recovers_manufactured_shift_and_width() -> None:
    template = {
        "energy": np.asarray([5887.0, 5899.0]),
        "width": np.asarray([2.0, 2.0]),
        "area": np.asarray([0.25, 0.75]),
    }
    centers = np.arange(5846.75, 5942.75, 0.5)
    truth = np.asarray([0.3, 4.7, 100000.0, 2.0])
    observed = line_shape.expected_counts(centers, 0.5, template, truth)
    bounds = {
        "common_centroid_shift_ev": [-2.0, 2.0],
        "instrument_gaussian_fwhm_ev": [2.0, 8.0],
        "line_normalization": [1.0, 10000000.0],
        "constant_background_per_ev": [0.0, 100000.0],
    }
    fit = line_shape.fit_histogram(observed, centers, 0.5, template, bounds)
    assert fit["converged"]
    assert abs(fit["centroid_shift_ev"] - truth[0]) < 0.02
    assert abs(fit["instrument_fwhm_ev"] - truth[1]) < 0.05
