import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a383_frozen_calibration_gate_fails_on_arc_rms_without_tuning():
    report = json.loads((ROOT / "results/r1_a383_gmos_calibrations/report.json").read_text())
    assert report["scope"] == "calibrations_only_before_any_science_frame_processing"
    assert len(report["biases"]) == 1
    assert report["biases"][0]["input_count"] == 5
    assert len(report["flats"]) == 5
    assert len(report["arcs"]) == 2
    assert all(report["gates"][name] for name in (
        "bias_construction_passed",
        "bias_overscan_residuals_passed",
        "flat_normalization_passed",
    ))
    assert report["arcs"][0]["wavelength_solution_rms_angstrom"] > 0.2
    assert report["arcs"][1]["wavelength_solution_rms_angstrom"] > 0.2
    assert report["gates"]["arc_wavelength_solutions_passed"] is False
    assert report["gates"]["P2a_calibration_products_gate_passed"] is False
    assert report["authorization"]["process_four_science_frames_with_frozen_mapping"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
