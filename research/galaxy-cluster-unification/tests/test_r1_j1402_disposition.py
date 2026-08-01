import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "r1_j1402_final_disposition" / "report.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_j1402_failure_is_specific_and_does_not_erase_the_predictive_passes() -> None:
    item = report()
    assert item["completed_gates"]["exact_stored_chain_likelihood_replay"]
    assert item["completed_gates"]["six_sector_pixelwise_prediction"]
    assert item["completed_gates"]["all_coordinate_negative_controls"]
    assert item["failed_gate"]["name"] == "maximum_coherent_heldout_residual_sigma"
    assert item["failed_gate"]["observed_sigma"] > item["failed_gate"][
        "maximum_allowed_sigma"
    ]
    assert not item["failed_gate"][
        "optimizer_mask_or_threshold_rerun_after_failure"
    ]


def test_third_external_nonpromotion_triggers_the_frozen_rethink() -> None:
    item = report()
    checkpoint = item["external_search_checkpoint"]
    assert len(checkpoint["completed_candidates"]) == 3
    assert checkpoint["promoted_candidates"] == []
    assert checkpoint["frozen_rethink_triggered"]
    assert not item["authorization"]["select_fourth_external_one_off_candidate"]
    assert item["authorization"]["reassess_ten_system_public_data_premise"]
    assert item["authorization"]["continue_RXJ2129_strict_observable_package"]


def test_disposition_authorizes_no_response_or_gravity_fit() -> None:
    authorization = report()["authorization"]
    assert not authorization["compute_J1402_lens_response_Jacobian"]
    assert not authorization["reduce_J1402_KCWI"]
    assert not authorization["count_J1402_toward_ten_system_target"]
    assert not authorization["infer_dynamical_or_Weyl_response"]
    assert not authorization["fit_gravity_response"]
    assert not authorization["fit_new_force_or_action"]
    assert not authorization["authorize_R2"]
