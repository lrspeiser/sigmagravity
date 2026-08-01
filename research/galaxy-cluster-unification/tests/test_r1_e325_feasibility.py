from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_e325_authorizes_a_jacobian_protocol_but_not_rank_three_promotion() -> None:
    report = json.loads(
        (ROOT / "results/r1_e325_feasibility/report.json").read_text()
    )
    inventory = pd.read_csv(ROOT / report["outputs"]["archive_inventory"])
    queue = pd.read_csv(ROOT / report["outputs"]["candidate_queue"])
    candidate = queue.loc[queue["system"] == "ESO 325-G004"]
    final = json.loads(
        (ROOT / "results/r1_e325_final_disposition/report.json").read_text()
    )

    assert report["selection_blind"] is True
    assert report["science_pixels_downloaded_or_inspected"] is False
    assert all(report["primary_source_checks"].values())
    assert report["primary_source_archive_hash_check_passed"] is True
    assert report["archive_inventory"]["eso_muse_product_count"] == 3
    assert report["archive_inventory"]["eso_muse_science_cube_count"] == 1
    assert report["archive_inventory"]["eso_muse_sky_cube_count"] == 2
    assert report["archive_inventory"]["eso_muse_calibration_level_3_count"] == 0
    assert report["archive_inventory"]["hst_expected_public_groups_found"] == 2
    assert report["geometry"]["accepted_dynamics_outer_radius_arcsec"] == 4.0
    assert report["geometry"]["published_einstein_radius_arcsec"] == 2.95
    assert report["geometry"]["einstein_radius_inside_accepted_support"] is True
    assert report["geometry"]["ring_only_radial_rank_upper_bound"] == 1
    assert report["geometry"]["published_radial_magnification_sensitivity"] is True
    assert report["geometry"]["pixel_count_used_as_rank"] is False
    assert report["geometry"]["full_image_level_nuisance_marginalized_rank"] == "not_established"
    assert report["gates"]["public_raw_archive_metadata_passed"] is True
    assert report["gates"]["pre_pixel_acquisition_and_jacobian_protocol_authorized"] is True
    assert report["gates"]["rank_three_candidate_admission_passed"] is False
    assert report["ten_system_effect"]["updated_structural_ceiling"] == 3
    assert report["ten_system_effect"]["minimum_new_rank_three_systems_still_required"] == 7
    assert len(inventory) == 16
    assert len(candidate) == 1
    assert bool(candidate.iloc[0]["counts_toward_ten_system_target"]) is False
    assert candidate.iloc[0]["next_authorized_stage"] == (
        "none_for_E325_promotion_retain_hash_locked_data_as_control"
    )
    assert final["decision"] == (
        "retain_as_hash_locked_extended_arc_control_not_a_rank_three_promotion"
    )
    assert final["authorization"]["continue_E325_promotion_work"] is False
    assert final["authorization"]["retain_E325_data_as_lower_rank_control"] is True
    assert report["authorization"]["freeze_acquisition_and_image_level_jacobian_protocol"] is True
    assert report["authorization"]["download_science_pixels_under_current_protocol"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False


def test_e325_j1_freezes_exact_products_and_rank_controls() -> None:
    config = json.loads(
        (ROOT / "configs/r1_e325_acquisition_jacobian_protocol.json").read_text()
    )
    products = config["acquisition"]["products"]
    rank = config["image_level_jacobian"]["projection_and_rank"]

    assert config["science_arrays_seen_at_freeze"] is False
    assert len(products) == 5
    assert sum(product["archive"] == "HST_MAST" for product in products) == 4
    assert sum(product["archive"] == "ESO_SODA" for product in products) == 1
    assert all(int(product["expected_bytes"]) > 0 for product in products)
    assert config["image_level_jacobian"]["response_basis"]["number_of_basis_directions"] == 4
    assert rank["minimum_rank_for_promotion"] == 3
    assert rank["singular_direction_detection_threshold"] == 3.0
    assert config["image_level_jacobian"]["controls"]["negative_mask"]
    assert config["dynamics_follow_on_gate"]["support_limit_arcsec"] == 4.0
    assert config["authorization"]["count_toward_ten_system_target_before_rank_and_dynamics_pass"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False
