import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_sigma_v19cy_a2319_common_differential_gain as closure  # noqa: I001


TERMINAL_REPORT = (
    ROOT
    / "results"
    / "sigma_v19cy_direct_icm_velocity_evidence"
    / "development_common_differential_gain_closure.json"
)


def test_frozen_common_differential_scope_is_exact() -> None:
    config, topology, _, _ = closure.validate_inputs()
    assert len(topology["segment_sets"]["10800"]) == 4
    assert config["physical_model"]["free_per_observation_parameters"] == 0
    assert config["physical_model"]["free_per_pixel_performance_parameters"] == 0
    assert config["cross_validation"]["folds"] == 4
    assert config["terminal_gate"][
        "maximum_absolute_candidate_minus_control_whole_array_centroid_ev"
    ] == 0.3
    assert not config["inputs"]["science_event_allowed"]
    for key in (
        "read_cluster_sky_event_rows",
        "apply_gain_to_cluster_sky_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        assert not config["authorization"][key]


def test_differential_model_recovers_manufactured_common_mode() -> None:
    dtype = [("TIME", "f8"), ("PIXEL", "i2"), ("TEMP_FIT", "f8")]
    fe55 = {}
    pxcal = {}
    for obsid_index in range(4):
        base = obsid_index * 1000.0
        common_times = base + np.arange(5, dtype=float) * 100.0
        common_rows = np.zeros(5, dtype=dtype)
        common_rows["TIME"] = common_times
        common_rows["PIXEL"] = 12
        common_rows["TEMP_FIT"] = 0.05 + 3e-6 * common_times
        pxcal[str(obsid_index)] = common_rows
        rows = np.zeros(36 * 3, dtype=dtype)
        row = 0
        for time in common_times[1:4]:
            common = 0.05 + 3e-6 * time
            for pixel in range(36):
                rows["TIME"][row] = time
                rows["PIXEL"][row] = pixel
                rows["TEMP_FIT"][row] = common + 1e-4 * pixel + 2e-8 * time
                row += 1
        fe55[str(obsid_index)] = rows
    models = closure.fit_differential_models("1", fe55, pxcal)
    predicted = closure.predict_differential(models[7], np.asarray([2500.0]))[0]
    assert abs(predicted - (7e-4 + 2e-8 * 2500.0)) < 1e-12
    assert closure.predict_differential(models[12], np.asarray([2500.0]))[0] == 0.0


def _fits(shift: float, width: float) -> dict:
    pixels = {
        str(pixel): {"centroid_shift_ev": shift, "instrument_fwhm_ev": width}
        for pixel in range(36)
        if pixel != 12
    }
    return {
        "whole_array": {"centroid_shift_ev": shift, "instrument_fwhm_ev": width},
        "per_pixel": pixels,
    }


def test_closure_comparison_uses_frozen_whole_and_per_pixel_gates() -> None:
    config = closure.load_json(closure.DEFAULT_CONFIG)
    control = _fits(0.0, 4.5)
    candidate = _fits(0.2, 4.7)
    comparison = closure.compare_fits(candidate, control, config["terminal_gate"])
    assert comparison["passed"]
    assert comparison["whole_array_centroid_delta_ev"] == 0.2
    for pixel in (0, 1, 2, 3):
        candidate["per_pixel"][str(pixel)]["centroid_shift_ev"] = 0.8
    comparison = closure.compare_fits(candidate, control, config["terminal_gate"])
    assert not comparison["passed"]
    assert comparison["per_pixel_absolute_centroid_delta_p90_ev"] > 0.5


def test_terminal_closure_preserves_mixed_scenario_result() -> None:
    report = json.loads(TERMINAL_REPORT.read_text(encoding="utf-8"))
    assert not report["terminal_gate_passed"]
    assert report["decision"] == "stop_without_science_branch_application"
    assert not report["cluster_sky_event_accessed"]
    assert not report["cluster_velocity_fit"]
    assert not report["validation_or_holdout_accessed"]
    assert len(report["commands"]) == 12
    assert all(command["exit_code"] == 0 for command in report["commands"])
    folds = {fold["segment"]: fold for fold in report["folds"]}
    assert not folds[0]["comparison"]["passed"]
    assert folds[1]["comparison"]["passed"]
    assert folds[2]["comparison"]["passed"]
    assert not folds[3]["comparison"]["passed"]
    assert folds[1]["validation_role"] == "interior_interpolation"
    assert folds[2]["validation_role"] == "interior_interpolation"
    assert folds[0]["validation_role"] == "endpoint_extrapolation"
    assert folds[3]["validation_role"] == "endpoint_extrapolation"
    assert folds[1]["comparison"]["whole_array_centroid_delta_ev"] == (
        -0.007665482864770368
    )
    assert folds[2]["comparison"]["whole_array_centroid_delta_ev"] == (
        -0.087579930902936
    )
    assert not report["authorization"]["freeze_science_branch_application_protocol"]
