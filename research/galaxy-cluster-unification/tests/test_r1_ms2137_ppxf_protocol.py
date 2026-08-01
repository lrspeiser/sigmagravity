import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ms2137_ppxf_protocol_freezes_support_and_theory_block():
    config = json.loads((ROOT / "configs/r1_ms2137_ppxf_covariance_protocol.json").read_text())
    assert config["status"] == "frozen_before_first_MS2137_DATA_or_STAT_array_read"
    assert config["spatial_extraction"]["annulus_count"] == 9
    assert config["spatial_extraction"]["annulus_edges_arcsec"][-1] == 14.0
    assert config["spectral_fit"]["template_family_baseline"] == "XSL"
    assert config["spectral_fit"]["additive_polynomial_degree"] == 5
    assert config["spectral_fit"]["multiplicative_polynomial_degree"] == 3
    assert config["stage_gates"]["P2_geometry_and_signal"]["minimum_median_signal_to_noise_each_annulus"] == 10.0
    assert config["stage_gates"]["P4_covariance_and_systematics"]["maximum_template_resolution_or_mask_shift_fraction_each_bin"] == 0.1
    assert config["authorization"]["change_support_or_thresholds_after_result"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False


def test_ms2137_ppxf_protocol_audit_passes_before_array_read():
    report = json.loads((ROOT / "results/r1_ms2137_ppxf_protocol/report.json").read_text())
    assert report["science_or_variance_arrays_read"] is False
    assert report["annulus_edges_arcsec"][-1] == 14.0
    assert all(report["gates"].values())
    assert report["decision"] == "authorize_P2_geometry_and_signal"
    assert report["authorization"]["execute_P2_geometry_and_signal"] is True
    assert report["authorization"]["execute_P3_ppxf"] is False
