from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_j0946_is_public_but_does_not_pass_the_rank_three_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_j0946_jackpot_feasibility/report.json").read_text()
    )
    inventory = pd.read_csv(ROOT / report["outputs"]["archive_inventory"])
    queue = pd.read_csv(ROOT / report["outputs"]["candidate_queue"])

    assert report["selection_blind"] is True
    assert report["science_pixels_downloaded_or_inspected"] is False
    assert all(report["primary_source_checks"].values())
    assert all(report["primary_source_archive_hash_checks"].values())
    assert report["archive_inventory"]["eso_muse_product_count"] == 11
    assert report["archive_inventory"]["eso_muse_calibration_level_3_count"] == 1
    assert report["archive_inventory"]["hst_expected_public_observations_found"] == 5
    assert report["geometry"]["accepted_dynamics_outer_radius_arcsec"] == 1.95
    assert report["geometry"]["ring_scales_inside_accepted_support_arcsec"] == [1.4]
    assert report["geometry"]["ring_scales_outside_accepted_support_arcsec"] == [2.1, 2.5]
    assert report["geometry"]["current_pre_fit_ring_scale_rank_upper_bound"] == 1
    assert report["gates"]["public_raw_archive_metadata_passed"] is True
    assert report["gates"]["three_ring_scales_inside_accepted_dynamics_support_passed"] is False
    assert report["gates"]["rank_three_candidate_admission_passed"] is False
    assert report["ten_system_effect"]["updated_structural_ceiling"] == 3
    assert report["ten_system_effect"]["minimum_new_rank_three_systems_still_required"] == 7
    assert len(inventory) == 16
    candidate = queue.loc[queue["system"] == "SDSS J0946+1006"]
    assert len(candidate) == 1
    assert bool(candidate.iloc[0]["counts_toward_ten_system_target"]) is False
    assert report["authorization"]["download_science_pixels"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
