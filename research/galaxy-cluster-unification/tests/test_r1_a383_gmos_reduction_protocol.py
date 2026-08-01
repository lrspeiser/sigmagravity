import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a383_reduction_protocol_was_frozen_pre_pixel():
    config = json.loads((ROOT / "configs/r1_a383_gmos_reduction_covariance_protocol.json").read_text())
    report = json.loads((ROOT / "results/r1_a383_gmos_reduction_protocol/report.json").read_text())
    assert report["science_arrays_inspected_at_freeze"] is False
    assert report["science_products_present_at_freeze"] == []
    assert report["outer_edge_arcsec"] == 10.5
    assert report["required_image_radius_arcsec"] < report["outer_edge_arcsec"]
    assert report["signed_bins_frozen"] == 9
    assert report["bootstrap_replicates_frozen"] == 200
    assert report["gates"]["protocol_freeze_gate_passed"] is True
    assert config["authorization"]["fit_new_force_or_action"] is False


def test_a383_environment_gate_passes_exact_versions_and_bpm():
    report = json.loads((ROOT / "results/r1_a383_dragons_environment/report.json").read_text())
    assert report["bpm_checksum_passed"] is True
    assert all(report["recognition_gates"].values())
    assert report["runtime"]["packages"]["dragons"] == "4.2.2"
    assert report["runtime"]["packages"]["ppxf"] == "9.4.8"
    assert report["gates"]["P1_environment_and_bpm_gate_passed"] is True
    assert report["authorization"]["execute_frozen_P2_calibration_reduction"] is True
    assert report["authorization"]["fit_new_force_or_action"] is False
