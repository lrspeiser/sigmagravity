import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a383_raw_acquisition_is_exact_and_header_only():
    report = json.loads((ROOT / "results/r1_a383_gemini_acquisition/report.json").read_text())
    assert report["science_arrays_opened"] is False
    assert report["files"] == 17
    assert report["science_position_angles_deg"] == [2.0]
    assert report["science_detectors"] == ["GMOS + Blue1 + new CCD1"]
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_reduction_protocol_freeze"
    assert report["authorization"]["freeze_reduction_and_covariance_protocol"] is True
    assert report["authorization"]["reduce_spectra"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
