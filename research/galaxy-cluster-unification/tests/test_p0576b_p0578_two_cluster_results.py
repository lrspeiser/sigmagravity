import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def report(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "report.json").read_text(encoding="utf-8"))


def test_p0576b_extended_source_plane_scan_runs_to_a_more_extreme_boundary():
    result = report("p0576b_fractional_boundary_extension")
    assert result["selected"]["candidate_id"] == "p2.6__f1"
    assert result["result"]["primary_improvement_vs_local_fraction"] > 0.93
    assert result["result"]["splits_improved"] == 6
    assert not result["gates"]["selected_power_interior_pass"]
    assert not result["gates"]["second_cluster_lock_authorized"]


def test_p0576c_proves_the_apparent_gain_is_mass_sheet_source_collapse():
    result = report("p0576c_source_plane_degeneracy_audit")
    assert result["result"]["p2p6_global_mass_sheet_R2"] > 0.9999
    assert result["result"]["p2p6_source_radius_ratio"] < 0.01
    assert result["result"]["fractional_within_family_RMS_monotonically_decreases"]
    assert result["gates"]["mass_sheet_resistant_metric_required"]
    assert not result["gates"]["fractional_source_plane_gain_is_non_degenerate"]


def test_p0576d_image_plane_metric_rejects_the_smacs_fractional_lock():
    result = report("p0576d_linearized_image_plane")
    assert result["selected"]["candidate_id"] == "p1.75__f1"
    assert result["result"]["improvement_vs_local_fraction"] > 0.15
    assert result["result"]["heldout_families_improved"] == 1
    assert result["result"]["selected_mass_sheet_R2"] > 0.99
    assert not result["gates"]["second_cluster_lock_authorized"]


def test_p0577_uses_seventeen_secure_second_cluster_positions():
    result = report("p0577_spt0615_raw_response")
    assert result["coverage"]["raw_images"] == 17
    assert result["coverage"]["subfamilies"] == 5
    assert result["coverage"]["calibration_images"] == 7
    assert result["coverage"]["heldout_images"] == 10


def test_p0577_fractional_power_does_not_transfer_between_clusters():
    result = report("p0577_spt0615_raw_response")
    assert result["SPT_selected"]["candidate_id"] == "p2__f1"
    assert result["result"]["SPT_selected_improvement_fraction"] < 0.10
    assert result["result"]["SMACS_locked_improvement_fraction"] < -0.60
    assert result["result"]["SMACS_locked_subfamilies_improved"] == 1
    assert not result["gates"]["cross_cluster_propagator_pattern_supported"]


def test_p0578_broadening_helps_only_one_cluster():
    result = report("p0578_two_cluster_baryon_broadening")
    assert result["selected"]["candidate_id"] == "w125__f1"
    assert result["result"]["improvement_vs_B100_fraction"] == pytest.approx(0.11954873064385174)
    assert result["result"]["clusters_improved"] == 1
    assert result["result"]["heldout_subfamilies_improved_fraction"] == 0.4
    per_cluster = {row["cluster"]: row for row in result["per_cluster"]}
    assert per_cluster["SMACS J0723.3-7327"]["improvement_fraction"] < 0.0
    assert per_cluster["SPT-CL J0615-5746"]["improvement_fraction"] > 0.26
    assert not result["gates"]["universal_broadening_supported"]


def test_p0578_keeps_angular_solar_and_sparc_nulls():
    cross = report("p0578_two_cluster_baryon_broadening")["cross_domain"]
    assert cross["solar_broad_fraction"] == 0.0
    assert cross["SPARC_angular_velocity_change_km_s"] == 0.0
