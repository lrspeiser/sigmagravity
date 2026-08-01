import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_report(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "report.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p0579_protocol_and_candidate_coverage_are_frozen():
    result = load_report("p0579_extent_gated_return_raw")
    protocol = ROOT / result["protocol"]["path"]
    assert result["protocol"]["sha256"] == sha256(protocol)
    assert result["coverage"]["candidates"] == 432
    assert result["coverage"]["raw_images"] == 29
    assert result["coverage"]["heldout_subfamilies"] == 5


def test_p0579_locked_inverse_route_improves_both_raw_clusters():
    result = load_report("p0579_extent_gated_return_raw")
    values = result["result"]
    assert result["primary_inverse_candidate_id"] == "K0338"
    assert values["primary_inverse_improvement_vs_B100_fraction"] == pytest.approx(
        0.1220148933597065
    )
    assert values["primary_inverse_clusters_improved"] == 2
    assert values["primary_inverse_heldout_subfamilies_improved_fraction"] == 0.6
    per_cluster = {row["cluster"]: row for row in result["per_cluster"]}
    assert per_cluster["SMACS J0723.3-7327"]["primary_inverse_improvement_fraction"] > 0.08
    assert per_cluster["SPT-CL J0615-5746"]["primary_inverse_improvement_fraction"] > 0.15


def test_p0579_locked_route_misses_only_the_mass_sheet_gate():
    gates = load_report("p0579_extent_gated_return_raw")["primary_inverse_replay_gates"]
    assert gates["equal_cluster_heldout_improvement_pass"]
    assert gates["cluster_count_pass"]
    assert gates["heldout_subfamily_fraction_pass"]
    assert gates["conservation_pass"]
    assert gates["solar_point_collapse_pass"]
    assert not gates["mass_sheet_pass"]
    assert not gates["all_primary_replay_gates_pass"]


def test_p0579_calibration_search_overfits_and_route_mode_is_largest_holdout_coordinate():
    result = load_report("p0579_extent_gated_return_raw")
    selected = result["selected"]
    assert selected["candidate_id"] == "K0107"
    assert selected["equal_cluster_calibration_RMS_arcsec"] < 2.01
    assert selected["equal_cluster_heldout_RMS_arcsec"] > 14.7
    impacts = result["parameter_impacts"]
    assert impacts[0]["parameter"] == "width_over_R80"  # calibration ranking
    route = next(row for row in impacts if row["parameter"] == "route_mode")
    assert route["heldout_RMS_span_arcsec_posthoc"] > 8.5


def test_p0580_conservative_return_has_the_right_sign_but_insufficient_amplitude():
    result = load_report("p0580_conservative_return_sparc")
    primary = result["primary_inverse_candidate"]
    references = result["references"]
    assert result["coverage"]["candidates"] == 432
    assert result["coverage"]["galaxies"] == 131
    assert result["coverage"]["outer_points"] == 968
    assert primary["outer_RMSE_km_s"] == pytest.approx(70.92603164566839)
    assert references["Newtonian_same_nuisance"]["outer_RMSE_km_s"] == pytest.approx(
        72.39921475798786
    )
    assert references["fixed_RAR_same_nuisance"]["outer_RMSE_km_s"] == pytest.approx(
        10.348465773189677
    )
    assert primary["galaxy_improved_fraction"] > 0.83
    assert result["primary_to_RAR_RMSE_ratio"] > 6.8


def test_p0580_conserves_the_budget_and_rejects_route_only_as_the_galaxy_solution():
    result = load_report("p0580_conservative_return_sparc")
    assert result["maximum_total_mass_conservation_error"] <= 1e-10
    gates = result["gates"]
    assert gates["primary_improves_Newtonian_outer_RMSE"]
    assert gates["primary_improves_at_least_60_percent_of_galaxies_vs_Newtonian"]
    assert gates["mass_conservation_pass"]
    assert gates["solar_point_collapse_pass"]
    assert not gates["primary_within_50_percent_of_fixed_RAR_outer_RMSE"]
    assert not gates["primary_conservative_return_supported"]


def test_p0580_endpoint_residence_is_the_dominant_parameter():
    result = load_report("p0580_conservative_return_sparc")
    impacts = result["parameter_impacts"]
    assert impacts[0]["parameter"] == "route_mode"
    assert impacts[0]["best_level"] == "endpoint"
    assert impacts[0]["outer_RMSE_impact_span_km_s"] > 1.8
    assert impacts[-1]["parameter"] == "route_fraction_multiplier"
    assert impacts[-1]["outer_RMSE_impact_span_km_s"] < 0.03
