import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0554_route_softness_interaction_protocol.json"
RESULTS = ROOT / "results" / "p0554_route_softness_interaction"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_combined_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_combined_score" in protocol["status"]
    assert len(protocol["variants"]) == 18
    assert len(protocol["impact_pairs"]) == 7
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["geometry_parameters_refit_per_cluster"] == 6
    assert protocol["evaluation"]["optimization_starts_per_variant_system"] == 8


def test_result_coverage_is_complete():
    report = load_report()
    assert report["status"] == "complete"
    assert report["coverage"] == {
        "variants": 18,
        "SPARC_galaxies": 131,
        "SPARC_outer_points": 968,
        "CLASH_systems": 20,
        "CLASH_points": 84,
        "raw_clusters": 5,
        "raw_heldout_images": 18,
        "exact_geometry_fits": 90,
        "route_fields": 80,
    }
    assert len(pd.read_csv(RESULTS / "raw_system_scores.csv")) == 90
    assert len(pd.read_csv(RESULTS / "geometry.csv")) == 90
    assert len(pd.read_csv(RESULTS / "field_audits.csv")) == 80
    assert len(pd.read_csv(RESULTS / "matched_comparisons.csv")) == 72
    assert len(pd.read_csv(RESULTS / "parameter_impacts.csv")) == 7


def test_angular_route_preserves_galaxy_clash_and_solar_monopole_controls():
    scores = pd.read_csv(RESULTS / "variant_scores.csv").set_index("variant_id")
    baseline = scores.loc["baseline"]
    route = scores.loc["route_parent"]
    assert route.galaxy_outer_RMSE_km_s == baseline.galaxy_outer_RMSE_km_s
    assert route.cluster_RMSE_dex == baseline.cluster_RMSE_dex
    assert (
        route.Mercury_precession_mas_per_century
        == baseline.Mercury_precession_mas_per_century
    )
    assert bool(route.all_solar_proxies_pass)

    lens = scores.loc["lensing_softness_098"]
    combined = scores.loc["combined_parent"]
    assert combined.galaxy_outer_RMSE_km_s == lens.galaxy_outer_RMSE_km_s
    assert combined.cluster_RMSE_dex == lens.cluster_RMSE_dex
    assert (
        combined.Mercury_precession_mas_per_century
        == lens.Mercury_precession_mas_per_century
    )
    assert np.isclose(lens.cluster_RMSE_dex, 0.19641371129844437)
    assert lens.cluster_RMSE_dex < baseline.cluster_RMSE_dex


def test_combination_recovers_topology_but_only_small_matched_accuracy():
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    totals = raw.groupby("variant_id").heldout_roots_converged.sum()
    complete = raw.groupby("variant_id").heldout_all_roots.sum()
    assert totals["baseline"] == 17
    assert totals["lensing_softness_098"] == 17
    assert totals["route_parent"] == 18
    assert totals["combined_parent"] == 18
    assert complete["combined_parent"] == 5

    comparisons = pd.read_csv(RESULTS / "matched_comparisons.csv")
    all_five = comparisons[comparisons.scope.eq("all_five")].set_index("variant_id")
    combined = all_five.loc["combined_parent"]
    assert combined.recovered_systems == "MACS1931"
    assert int(combined.candidate_complete_systems) == 5
    assert int(combined.candidate_total_roots) == 18
    assert np.isclose(combined.matched_improvement_fraction, 0.0031035907254171047)
    assert np.isclose(combined.candidate_complete_RMS_arcsec, 16.839147430843536)

    route = all_five.loc["route_parent"]
    lens = all_five.loc["lensing_softness_098"]
    assert route.matched_improvement_fraction < 0.0
    assert lens.matched_improvement_fraction > 0.02
    assert int(lens.candidate_complete_systems) == 4


def test_small_route_changes_are_mostly_topology_switches_not_rms_levers():
    impacts = pd.read_csv(RESULTS / "parameter_impacts.csv").set_index("parameter")
    assert impacts.index[0] == "lensing_addition_softness"
    assert np.isclose(
        impacts.loc["lensing_addition_softness", "raw_log_elasticity"],
        0.835401836225762,
    )
    assert int(impacts.loc["lensing_addition_softness", "root_count_span"]) == 1

    assert int(impacts.loc["base_width_kpc", "common_complete_systems"]) == 5
    assert int(impacts.loc["base_width_kpc", "root_count_span"]) == 0
    assert np.isclose(
        impacts.loc["base_width_kpc", "raw_RMS_span_arcsec"],
        0.0488588447947504,
    )
    assert int(impacts.loc["extent_slope", "common_complete_systems"]) == 5
    assert np.isclose(
        impacts.loc["extent_slope", "raw_RMS_span_arcsec"],
        0.0063090218112836,
    )

    switching = {
        "base_fraction",
        "route_power",
        "source_weight_power",
        "base_length_kpc",
    }
    assert set(impacts.loc[list(switching), "root_count_span"].astype(int)) == {1}
    assert (impacts.loc[list(switching), "raw_RMS_span_arcsec"] < 0.05).all()


def test_route_field_is_conservative_and_curl_free_to_numerical_precision():
    report = load_report()
    assert report["maximum_route_curl_RMS"] < 1.0e-12
    assert report["maximum_annular_convergence_error"] < 1.0e-12
    assert report["geometry_boundary_fits"] == 72


def test_radial_and_route_effects_are_not_additive_after_refitting():
    interaction = load_report()["radial_route_interaction"]
    assert interaction["common_complete_systems"] == 4
    values = interaction["equal_system_RMS_arcsec"]
    assert values["lensing_softness_098"] < values["baseline"]
    assert values["route_parent"] > values["baseline"]
    assert values["combined_parent"] < values["baseline"]
    assert np.isclose(interaction["RMS_interaction_arcsec"], 0.38289919186910026)
    assert interaction["RMS_interaction_arcsec"] > 0.0


def test_best_complete_formula_is_descriptive_not_promoted():
    report = load_report()
    ranked = report["complete_solar_safe_ranked"]
    assert ranked[0]["variant_id"] == "combined_power_240"
    assert np.isclose(ranked[0]["candidate_complete_RMS_arcsec"], 16.808423480125597)
    assert report["verdict"] == {
        "any_complete_solar_safe_formula": True,
        "combined_parent_complete": True,
        "no_formula_promoted": True,
    }
