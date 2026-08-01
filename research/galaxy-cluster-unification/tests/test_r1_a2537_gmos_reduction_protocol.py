import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_a2537_control_protocol_is_pre_pixel_fixed_and_not_science_pilot():
    config = json.loads((ROOT / "configs/r1_a2537_gmos_reduction_covariance_protocol.json").read_text())
    report = json.loads((ROOT / "results/r1_a2537_gmos_reduction_protocol/report.json").read_text())
    assert report["science_arrays_inspected_at_freeze"] is False
    assert report["science_products_present_at_freeze"] == []
    assert report["disturbed_control"] is True
    assert report["counts_as_non_disturbed_pilot"] is False
    assert report["outer_edge_arcsec"] == 16.0
    assert report["required_image_radius_arcsec"] < report["outer_edge_arcsec"]
    assert report["signed_bins_frozen"] == 9
    assert report["bootstrap_replicates_frozen"] == 200
    assert report["pointing_step_semantics_frozen"] is True
    assert config["spatial_extraction"]["signed_bin_edges_arcsec"] == [-16.0, -11.0, -7.0, -3.0, -0.5, 0.5, 3.0, 7.0, 11.0, 16.0]
    assert config["calibration_acceptance"]["maximum_reconstructed_bcg_center_range_between_exposures_arcsec"] == 0.3
    assert report["gates"]["protocol_freeze_gate_passed"] is True
    assert report["authorization"]["audit_environment"] is True
    assert report["authorization"]["execute_calibration_reduction"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
