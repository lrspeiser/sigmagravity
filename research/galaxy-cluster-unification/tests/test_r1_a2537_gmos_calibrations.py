from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2537_control_calibration_gate_is_binding() -> None:
    report = json.loads((ROOT / "results/r1_a2537_gmos_calibrations/report.json").read_text())
    assert report["scope"] == "calibrations_only_before_any_science_frame_processing"
    assert report["disturbed_control"] is True
    assert report["counts_as_non_disturbed_pilot"] is False
    assert len(report["biases"]) == 2
    assert all(row["input_count"] == 5 for row in report["biases"])
    assert len(report["flats"]) == 4
    assert len(report["arcs"]) == 2
    assert report["authorization"]["fit_stellar_kinematics"] is False
    assert report["authorization"]["count_as_non_disturbed_pilot"] is False
    assert report["gates"]["gravity_response_fit_authorized"] is False
