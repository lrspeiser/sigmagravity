import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r1_j1402_dinos_interface_correction_protocol.json"


def correction() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_correction_was_frozen_before_any_forward_result() -> None:
    item = correction()
    assert item["status"].endswith("before_software_install_or_forward_model_evaluation")
    assert not item["science_fit_seen_before_correction"]
    assert not item["forward_model_seen_before_correction"]
    assert not item["lens_response_seen_before_correction"]
    assert "does not change" in item["reason_for_correction"]


def test_mask_and_PSF_corrections_are_exact_interface_semantics() -> None:
    item = correction()
    mask = item["mask_correction"]
    psf = item["PSF_correction"]
    assert mask["released_Dolphin_evidence"]["commit"] == "1593c573541d26ae5791835430c68858988a969b"
    assert mask["released_Dolphin_evidence"]["blob_sha1"] == "f329ac8cd464a024dffaba35ba016cf7b26f027b"
    assert "one minus" in mask["corrected_invariants"][0]
    assert psf["released_lenstronomy_evidence"]["tag"] == "v1.11.5"
    assert psf["released_lenstronomy_evidence"]["blob_sha1"] == "8f4c6f95c089eafa46a61b56413b7b5214be38f3"
    assert "1e-12" in psf["corrected_invariants"][2]
    assert len(item["unchanged_scientific_gates"]) == 6
    packaging = item["Dolphin_packaging_correction"]
    assert "omits" in packaging["released_source_cause"]
    assert packaging["corrected_environment_method"].startswith("Safely extract")
    assert not packaging["source_modifications_allowed"]
    assert not packaging["forward_model_seen_before_correction"]
    astropy = item["Astropy_compatibility_correction"]
    assert astropy["corrected_pin"] == "astropy==5.3.4"
    assert not astropy["forward_model_seen_before_correction"]
    assert not astropy["scientific_threshold_changed"]
    boundary = item["candidate_bundle_interface_boundary"]
    assert "ImageData, PSFData, and ModelConfig" in boundary["corrected_audit_method"]
    assert not boundary["new_science_input_added"]
    assert not boundary["forward_model_seen_before_correction"]
    rounding = item["PSF_float32_normalization_correction"]
    assert rounding["superseded_interface_tolerance"] == 1e-12
    assert rounding["corrected_maximum_absolute_sum_error"] == 5e-7
    assert not rounding["scientific_threshold_changed"]
    assert not rounding["forward_model_seen_before_correction"]


def test_correction_does_not_authorize_model_or_science_promotion() -> None:
    auth = correction()["authorization"]
    assert auth["rerun_only_the_corrected_no_fit_structural_audit"]
    assert not auth["evaluate_forward_model_before_environment_gate"]
    assert not auth["compute_lens_response"]
    assert not auth["reduce_KCWI"]
    assert not auth["infer_gravity_response"]
    assert not auth["authorize_R2"]
