import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def assert_protocol(report):
    assert report["protocol"]["sha256"] == digest(report["protocol"]["path"])


def test_balanced_and_capacity_backtracks_are_complete_and_numerically_valid():
    balanced = load("results/p0554_halo_backtrack/report.json")
    capacity = load("results/p0554_halo_backtrack_capacity/report.json")
    assert_protocol(balanced)
    assert_protocol(capacity)
    assert balanced["coverage"]["systems"] == 5
    assert balanced["coverage"]["cluster_scale_halos"] == 7
    assert balanced["coverage"]["transport_solutions"] == 50
    assert balanced["coverage"]["radial_angle_controls"] == 160
    assert balanced["coverage"]["minimum_posterior_center_fraction_inside_target_aperture"] == 1.0
    assert balanced["coverage"]["maximum_source_marginal_error"] < 1e-6
    assert capacity["coverage"]["capacity_solutions"] == 20
    assert capacity["coverage"]["q1_maximum_RMS_difference_from_parent_kpc"] == 0.0
    assert capacity["origin_stability"]["halos_with_same_top_origin_at_q2_q4_q8"] == 7


def test_wide_field_failure_and_member_aperture_diagnostic_are_preserved():
    legacy = load("results/p0554_macs1931_wide_field/report.json")
    aperture = load("results/p0554_macs1931_member_aperture/report.json")
    assert_protocol(legacy)
    assert_protocol(aperture)
    assert legacy["status"] == "input_inadequate_no_density_score"
    assert legacy["density_claim_made"] is False
    assert legacy["coverage"]["objects_with_usable_r_flux"] == 0
    assert aperture["coverage"]["published_members_total"] == 120
    assert aperture["qcap4_aperture_effect"]["rms_improvement_fraction"] > 0.15
    assert aperture["qcap4_aperture_effect"]["top_origin_distance_450kpc"] > 400.0


def test_subaru_endpoint_result_obeys_frozen_gate_and_provenance():
    endpoint = load("results/p0554_macs1931_subaru_endpoint/report.json")
    robustness = load("results/p0554_macs1931_endpoint_robustness/report.json")
    provenance = load("data/raw/p0554_macs1931_subaru/provenance.json")
    assert_protocol(endpoint)
    assert_protocol(robustness)
    products = {row["role"]: row for row in provenance["products"]}
    assert products["bpz_photometric_redshifts"]["lines"] == 108689
    assert endpoint["catalog"]["catalog_rows"] == 108658
    assert endpoint["catalog"]["halo_posterior_inside_catalog_bounds_fraction"] == 1.0
    primary = endpoint["primary_endpoint_test"]
    assert primary["rotation_p_value"] < 0.05
    assert primary["density_ratio"] < 1.5
    assert endpoint["frozen_counterpart_gate_passed"] is False
    assert endpoint["outcome"] == "no_frozen_significant_baryonic_counterpart"
    robust = robustness["robust_endpoint_test"]
    assert robust["density_ratio"] > 1.5
    assert robust["rotation_p_value"] > 0.05
    assert robustness["robustness_gate_passed"] is False
    assert robustness["outcome"] == "count_excess_not_robust"


def test_subaru_transport_is_descriptive_but_shorter_than_truncated_source_inverse():
    endpoint = load("results/p0554_macs1931_subaru_endpoint/report.json")
    previous = endpoint["transport"]["previous_full_published_member_qcap4_rms_kpc"]
    scores = endpoint["transport"]["subaru_scores"]
    assert len(scores) == 2
    assert all(row["rms_transport_kpc"] < previous for row in scores)
    assert all(row["target_marginal_max_error"] < 1e-6 for row in scores)
    assert all(row["maximum_source_capacity_excess"] < 1e-8 for row in scores)
