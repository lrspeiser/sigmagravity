from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def report(stage: str) -> dict:
    return json.loads((ROOT / "results" / stage / "report.json").read_text(encoding="utf-8"))


def test_p0710_acquired_the_complete_frozen_product_inventory() -> None:
    acquisition = report("p0710_external_target_acquisition")
    assert acquisition["status"] == "pass"
    assert acquisition["requested_products"] == acquisition["received_products"] == 51
    assert acquisition["failed_products"] == []
    assert acquisition["total_bytes"] == 1_004_204_557
    assert acquisition["P0633_sample_now_spent"] is True
    assert acquisition["formula_changes_after_this_run_are_validation"] is False

    supplement = report("p0710b_iorio_online_tables_acquisition")
    assert supplement["status"] == "pass"
    assert supplement["sha256"] == (
        "967110269d59357ee3a94d1d6e46c2402aef38da3f674180d42044ceaf094173"
    )
    assert supplement["members"] == ["BTFR_data.txt", "results.zip"]


def test_p0711_rotation_curve_holdout_passed_without_target_refits() -> None:
    outcome = report("p0711_external_galaxy_rotation_validation")
    assert outcome["status"] == "pass"
    assert outcome["valid_galaxies"] == 13
    assert outcome["best_frozen_MOND_model"] == "QUMOND_simple_nu_3D"
    assert outcome["candidate_to_best_MOND_RMSE_ratio"] < 1.0
    assert outcome["candidate_to_best_MOND_RMSE_ratio"] <= 1.05
    assert outcome["maximum_morphology_bin_ratio"] <= 1.25
    assert outcome["per_object_gravity_parameters"] == 0
    assert outcome["target_refits"] == 0


def test_p0712_velocity_field_holdout_passed_but_newtonian_pixel_residual_is_lower() -> None:
    outcome = report("p0712_external_galaxy_velocity_field_validation")
    assert outcome["status"] == "pass"
    assert outcome["valid_galaxies"] == 13
    assert outcome["candidate_to_best_MOND_RMSE_ratio"] <= 1.05
    assert (
        outcome["sample_weighted_RMSE_km_s"]["Newtonian_3D"]
        < outcome["sample_weighted_RMSE_km_s"]["P0707_time_potential"]
    )
    assert outcome["ordinary_observational_coordinates"]["handedness_selected_by_model_RMSE"] is False
    assert outcome["photometric_distance_inclination_PA_refits"] == 0


def test_p0713_records_the_predeclared_cluster_readiness_failure() -> None:
    outcome = report("p0713_external_cluster_readiness_audit")
    assert outcome["status"] == "fail_data_readiness"
    assert outcome["formula_scored"] is False
    assert outcome["ready_clusters"] == 2
    assert outcome["required_ready_clusters"] == 4
    rows = {row["cluster"]: row for row in outcome["cluster_rows"]}
    assert rows["AS295"]["ready"] is True
    assert rows["PLCKG287"]["ready"] is True
    assert rows["MACS0025"]["secure_families"] == 2
    assert rows["MACS0025"]["secure_images"] == 7
    assert rows["MACS0159"]["spectroscopic_families"] == 0
    assert rows["MACS0025"]["ready"] is False
    assert rows["MACS0159"]["ready"] is False


def test_p0714_ready_subset_has_no_candidate_multiple_image_topology() -> None:
    outcome = report("p0714_ready_subset_raw_lensing")
    assert outcome["status"] == "completed_exploratory_ready_subset"
    assert outcome["validation_status"].startswith("not_a_P0633_validation")
    assert outcome["ready_clusters"] == ["AS295", "PLCKG287"]
    assert outcome["frozen_candidate_all_heldout_roots_converged"] is False
    assert outcome["frozen_candidate_all_heldout_topologies_correct"] is False
    assert outcome["critical_curve_gate"].startswith("not_independently_observable")

    family = pd.read_csv(
        ROOT / "results/p0714_ready_subset_raw_lensing/family_model_scores.csv"
    )
    candidate = family[family.model == "P0707_Weyl_frozen_axis_contract"]
    repaired = family[family.model == "P0707_Weyl_axis_repaired_exploratory"]
    assert (candidate.observed_images >= 2).all()
    assert (candidate.global_roots == 1).all()
    assert (repaired.global_roots == 1).all()
    assert not candidate.topology_correct.any()

    scores = pd.read_csv(
        ROOT / "results/p0714_ready_subset_raw_lensing/cluster_model_scores.csv"
    )
    as295_halo = scores[
        (scores.cluster == "AS295") & (scores.model == "glafic_v2_compact_halo")
    ].iloc[0]
    assert as295_halo.root_convergence_fraction == 1.0
    assert as295_halo.heldout_image_RMS_arcsec < 1.0
