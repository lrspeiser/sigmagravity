import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_R1B3_feasibility_is_pre_pixel_and_numeric():
    config = json.loads((ROOT / "configs/r1_rxj2129_strict_observable_next_stage.json").read_text())
    report = json.loads((ROOT / "results/r1_rxj2129_strict_observable_feasibility/report.json").read_text())
    assert report["XMM_pixels_downloaded_or_inspected"] is False
    assert report["HST_arc_pixels_measured"] is False
    assert report["xmm_metadata"]["obsid"] == "0093030201"
    assert report["xmm_metadata"]["duration"] >= 50000
    assert report["xmm_located_data"]["content_length"] < 1000000000
    assert len(report["hst_header_audit"]) == 2
    assert report["lens_ledger"]["spectroscopic_images"] == 21
    assert report["lens_ledger"]["spectroscopic_families"] == 7
    assert report["lens_ledger"]["inner_image_ids"] == ["5.2", "6.3", "8.2"]
    assert config["xmm_reduction_gate"]["minimum_cleaned_exposure_seconds_each_passing_instrument"] == 15000
    assert config["gas_likelihood_gate"]["minimum_accepted_annuli"] == 5
    assert config["hst_astrometric_covariance_gate"]["minimum_all_field_images_accepted"] == 18
    assert all(report["gates"].values())
    assert report["authorization"]["download_exact_XMM_observation"] is True
    assert report["authorization"]["infer_dynamical_or_Weyl_response"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
