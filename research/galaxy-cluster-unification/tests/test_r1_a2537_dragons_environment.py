import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2537_control_environment_passes_exact_versions_and_bpm():
    report = json.loads((ROOT / "results/r1_a2537_dragons_environment/report.json").read_text())
    assert report["science_arrays_opened"] is False
    assert report["bpm_checksum_passed"] is True
    assert all(report["recognition_gates"].values())
    assert report["runtime"]["packages"]["dragons"] == "4.2.2"
    assert report["runtime"]["packages"]["ppxf"] == "9.4.8"
    assert report["gates"]["C1_environment_and_bpm_gate_passed"] is True
    assert report["authorization"]["execute_frozen_C2_calibration_reduction"] is True
    assert report["authorization"]["count_as_non_disturbed_pilot"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
