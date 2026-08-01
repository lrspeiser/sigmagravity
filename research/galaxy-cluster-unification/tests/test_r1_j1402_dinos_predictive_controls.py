import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "r1_j1402_dinos_predictive_controls_protocol.json"
CORRECTION = (
    ROOT / "configs" / "r1_j1402_dinos_predictive_controls_implementation_correction.json"
)
REPORT = ROOT / "results" / "r1_j1402_dinos_predictive_controls" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_predictive_protocol_was_frozen_before_sector_scores() -> None:
    protocol = load(PROTOCOL)
    assert protocol["sector_scores_seen_at_freeze"] is False
    assert protocol["coordinate_corruption_scores_seen_at_freeze"] is False
    assert protocol["sector_geometry"]["count"] == 6
    assert protocol["sector_geometry"]["phase_degrees"] == 0.0
    assert protocol["predictive_metrics"][
        "maximum_six_sector_heldout_reduced_chi_square"
    ] == 1.5
    assert protocol["predictive_metrics"][
        "maximum_coherent_heldout_residual_sigma"
    ] == 5.0


def test_predictive_decision_is_derived_from_every_frozen_check() -> None:
    report = load(REPORT)
    assert report["predictive_coordinate_gate_pass"] == all(report["checks"].values())
    assert report["authorization"]["compute_frozen_lens_response_Jacobian"] == report[
        "predictive_coordinate_gate_pass"
    ]


def test_masked_solver_return_correction_is_explicit_and_guarded() -> None:
    protocol = load(PROTOCOL)
    correction = load(CORRECTION)
    report = load(REPORT)
    assert correction["scientific_protocol_changed"] is False
    assert correction["sectors_masks_thresholds_coordinates_and_parameters_changed"] is False
    assert protocol["implementation_contract"]["heldout_prediction"].startswith(
        "after the solve updates the linear amplitudes"
    )
    assert report["implementation_correction"]["id"] == correction["correction_id"]
    assert report["checks"][
        "linear_solver_masked_return_is_zero_filled_on_every_heldout_sector"
    ]
    assert report["checks"][
        "complete_forward_model_is_nonzero_on_every_heldout_sector"
    ]


def test_all_six_baseline_sector_scores_are_reported_per_band() -> None:
    baseline = load(REPORT)["released_and_corrupted_results"]["baseline"]
    assert [item["sector"] for item in baseline["sectors"]] == list(range(6))
    assert all(
        [band["band"] for band in sector["bands"]]
        == ["F435W", "F555W", "F814W"]
        for sector in baseline["sectors"]
    )
    assert baseline["aggregate_heldout_pixels"] == 25807


def test_every_frozen_coordinate_control_is_scored_without_physics_authorization() -> None:
    report = load(REPORT)
    assert set(report["negative_controls"]) == {
        "scalar_0p04",
        "zero_shifts",
        "swap_F555W_F814W_coordinate_maps",
    }
    authorization = report["authorization"]
    assert not authorization["infer_gravity_response"]
    assert not authorization["fit_new_force_or_action"]
    assert not authorization["authorize_R2"]


def test_j1402_stops_only_on_the_frozen_coherent_residual_gate() -> None:
    report = load(REPORT)
    checks = report["checks"]
    assert checks["maximum_six_sector_heldout_reduced_chi_square_passes"]
    assert checks["every_coordinate_corruption_worsens_heldout_likelihood"]
    assert checks["every_coordinate_corruption_changes_instantiated_coordinates"]
    assert not checks["maximum_coherent_heldout_residual_passes"]
    assert [name for name, passed in checks.items() if not passed] == [
        "maximum_coherent_heldout_residual_passes"
    ]
    assert report["released_and_corrupted_results"]["baseline"][
        "maximum_PSF_matched_coherent_residual_sigma"
    ] > report["predictive_thresholds"][
        "maximum_coherent_heldout_residual_sigma"
    ]
    assert report["predictive_coordinate_gate_pass"] is False
    assert report["decision"] == (
        "stop_J1402_lens_promotion_after_predictive_or_coordinate_control_failure"
    )
