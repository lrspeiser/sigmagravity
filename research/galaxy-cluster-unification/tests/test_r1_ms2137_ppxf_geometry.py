import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ms2137_P2_report_preserves_frozen_outer_support_and_stop_logic():
    report = json.loads((ROOT / "results/r1_ms2137_ppxf_geometry/report.json").read_text())
    assert report["data_and_variance_arrays_read"] is True
    assert report["ppxf_run"] is False
    assert len(report["annuli"]) == 9
    assert report["annuli"][-1]["inner_arcsec"] == 8.5
    assert report["annuli"][-1]["outer_arcsec"] == 14.0
    passed = report["gates"]["P2_geometry_and_signal_gate_passed"]
    assert report["authorization"]["execute_P3_baseline_ppxf"] is passed
    assert report["decision"] == ("authorize_P3_baseline_ppxf" if passed else "stop_MS2137_at_P2_geometry_and_signal")
    assert report["authorization"]["change_support_or_thresholds"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
