import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_XMM_acquisition_is_complete_and_environment_locked():
    report = json.loads((ROOT / "results/r1_rxj2129_xmm_acquisition/report.json").read_text())
    assert report["obsid"] == "0093030201"
    assert report["XMM_pixels_inspected"] is False
    assert report["local_files"] == 2824
    assert report["local_bytes"] == 645765075
    assert report["category_counts"] == {"4XMM": 276, "ODF": 284, "om_mosaic": 9, "PPS": 2255}
    assert report["partial_files"] == []
    assert report["gates"]["archive_manifest_ETag_size_and_SHA_provenance_passed"] is True
    assert report["gates"]["raw_ODF_and_PPS_classes_present"] is True
    assert report["gates"]["exact_SAS_HEASoft_environment_present"] is False
    assert report["gates"]["R1B3_XMM_acquisition_gate_passed"] is True
    assert report["gates"]["R1B3_XMM_reduction_environment_gate_passed"] is False
    assert report["authorization"]["freeze_and_install_SAS_HEASoft_CCF_environment"] is True
    assert report["authorization"]["run_cifbuild_or_odfingest"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
