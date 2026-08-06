import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_sigma_v19cy_a2319_calibration_line_shape as line_shape  # noqa: I001


TERMINAL_REPORT = (
    ROOT
    / "results"
    / "sigma_v19cy_direct_icm_velocity_evidence"
    / "development_calibration_line_shape.json"
)


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


def test_terminal_line_shape_result_stops_before_cluster_events() -> None:
    report = json.loads(TERMINAL_REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == "stop_before_cluster_event_application"
    assert not report["line_shape_gate_passed"]
    assert report["selected_candidate"] is None
    assert not report["cluster_sky_event_accessed"]
    assert not report["cluster_velocity_fit"]
    assert not report["validation_or_holdout_accessed"]
    assert all(not allowed for allowed in report["authorization"].values())

    summaries = {row["candidate"]: row for row in report["selection"]["summaries"]}
    assert summaries["branch_linear_common_mode"]["score"] == 5772.581543531192
    assert summaries["branch_linear_common_mode"]["maximum_absolute_z"] == 73.07926100138947
    assert summaries["branch_linear_common_mode"]["passed"] is False

    best = report["fit_results"]["branch_linear_common_mode"]
    assert best["000101000"]["centroid_shift_ev"] == 0.32079261001389475
    assert best["000101000"]["instrument_fwhm_ev"] == 4.778993234074812
    assert best["000102000"]["centroid_shift_ev"] == 0.15713701047640277
    assert best["000102000"]["instrument_fwhm_ev"] == 4.544768450237075
    assert best["000103000"]["centroid_shift_ev"] == -0.05391332232601177
    assert best["000103000"]["instrument_fwhm_ev"] == 4.598311596867162
