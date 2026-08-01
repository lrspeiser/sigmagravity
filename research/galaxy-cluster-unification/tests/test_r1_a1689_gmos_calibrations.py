from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a1689_gmos_calibration_gate() -> None:
    report = json.loads((ROOT / "results/r1_a1689_gmos_calibrations/report.json").read_text())
    assert report["scope"] == "calibrations_only_before_any_science_frame_processing"
    assert len(report["biases"]) == 2
    assert all(row["input_count"] == 5 for row in report["biases"])
    assert all(row["sigma_clipped_median_recorded"] for row in report["biases"])
    assert all(row["first_input_reproduction_matches_processed_headers"] for row in report["biases"])
    assert len(report["flats"]) == 4
    assert len(report["arcs"]) == 3
    assert min(row["matched_cuar_lines"] for row in report["arcs"]) >= 12
    assert max(row["wavelength_solution_rms_angstrom"] for row in report["arcs"]) <= 0.2
    assert report["gates"]["P2a_calibration_products_gate_passed"] is True
    assert report["authorization"]["process_four_science_frames_with_frozen_mapping"] is True
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
