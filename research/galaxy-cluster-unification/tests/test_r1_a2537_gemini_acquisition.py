import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2537_raw_acquisition_is_exact_header_only_and_control_labeled():
    report = json.loads((ROOT / "results/r1_a2537_gemini_acquisition/report.json").read_text())
    assert report["science_arrays_opened"] is False
    assert report["files"] == 21
    assert report["science_position_angles_deg"] == [124.0, 124.0, 124.0, 124.0]
    assert report["first_header_audit_semantic_correction"]["science_pixels_read"] is False
    assert report["first_header_audit_semantic_correction"]["science_threshold_or_target_changed"] is False
    assert report["disturbed_control"] is True
    assert report["counts_as_non_disturbed_pilot"] is False
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_disturbed_control_reduction_protocol_freeze"
    assert report["authorization"]["freeze_reduction_and_covariance_protocol"] is True
    assert report["authorization"]["count_as_non_disturbed_pilot"] is False
    assert report["authorization"]["reduce_spectra"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
