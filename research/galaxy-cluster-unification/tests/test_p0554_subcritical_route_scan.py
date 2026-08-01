import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0554_subcritical_route_scan_protocol.json"
RESULTS = ROOT / "results" / "p0554_subcritical_route_scan"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_continuation_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_route_continuation_score" in protocol["status"]
    assert protocol["formula"]["eta_values"] == [value / 10 for value in range(11)]
    assert protocol["formula"]["eta_parameters_fit"] == 0
    assert protocol["formula"]["new_per_cluster_gravity_parameters"] == 0
    assert protocol["evaluation"]["ordinary_geometry_parameters_refit"] == 6
    assert protocol["evaluation"]["optimization_starts_per_eta"] == 8


def test_result_coverage_and_root_closure_are_complete():
    report = load_report()
    assert report["report_version"] == "P0554-SUBCRITICAL-ROUTE-SCAN-RESULTS-0.2.0"
    assert report["coverage"] == {
        "eta_values": 11,
        "geometry_fits": 11,
        "optimization_starts": 88,
        "formula_family_searches": 77,
        "accepted_global_roots": 244,
        "published_images": 19,
        "source_families": 7,
    }
    roots = pd.read_csv(RESULTS / "global_roots.csv")
    assert len(roots) == 244
    assert roots.closure_arcsec.max() < 1.0e-6
    assert len(pd.read_csv(RESULTS / "family_summary.csv")) == 77
    assert len(pd.read_csv(RESULTS / "assignments.csv")) == 209
    assert len(pd.read_csv(RESULTS / "heldout_predictions.csv")) == 44


def test_subcritical_window_and_best_descriptive_eta_are_frozen():
    summary = pd.read_csv(RESULTS / "eta_summary.csv").set_index("eta")
    assert summary.loc[0.0:0.5, "subcritical"].astype(bool).all()
    assert not summary.loc[0.6:1.0, "subcritical"].astype(bool).any()
    best = summary[summary.subcritical.astype(bool)].equal_family_assignment_RMS_arcsec.idxmin()
    assert np.isclose(best, 0.3)
    assert np.isclose(summary.loc[0.0, "equal_family_assignment_RMS_arcsec"], 21.661530260755512)
    assert np.isclose(summary.loc[0.3, "equal_family_assignment_RMS_arcsec"], 21.56006254816065)
    assert np.isclose(
        summary.loc[0.3, "assignment_improvement_fraction_vs_eta0"],
        0.004684235664490055,
    )


def test_caustic_transitions_are_separate_for_families_two_and_three():
    summary = pd.read_csv(RESULTS / "eta_summary.csv").set_index("eta")
    assert summary.family_2_roots.astype(int).tolist() == [3] * 10 + [5]
    assert summary.family_3_roots.astype(int).tolist() == [5] * 6 + [7, 7, 8, 7, 7]
    assert summary.potentially_observable_surplus_roots.astype(int).tolist() == [
        2, 2, 2, 2, 2, 2, 4, 4, 5, 4, 6
    ]
    report = load_report()
    assert report["first_any_topology_change_eta"] == 0.6
    assert report["first_family_3_topology_change_eta"] == 0.6
    assert report["first_family_2_topology_change_eta"] == 1.0


def test_partial_heldout_diagnostic_does_not_hide_missing_root():
    summary = pd.read_csv(RESULTS / "eta_summary.csv").set_index("eta")
    expected = [
        8.12879491613258, 8.109408966458739, 8.06019257447695,
        8.078090959669677, 8.106701127131839, 8.084960234177853,
        8.085132639489066, 8.045889305528112, 8.047895005265563,
        7.978652102072531, 7.070010288342311,
    ]
    assert np.allclose(summary.heldout_converged_only_RMS_arcsec, expected)
    assert summary.loc[0.0:0.9, "heldout_observed_seed_roots"].astype(int).eq(3).all()
    assert int(summary.loc[1.0, "heldout_observed_seed_roots"]) == 4
    assert np.isinf(summary.loc[0.3, "heldout_observed_seed_RMS_arcsec"])
    assert np.isclose(summary.loc[1.0, "heldout_observed_seed_RMS_arcsec"], 7.070010288342311)


def test_large_full_strength_gain_is_after_topology_change():
    summary = pd.read_csv(RESULTS / "eta_summary.csv").set_index("eta")
    assert np.isclose(
        summary.loc[1.0, "assignment_improvement_fraction_vs_eta0"],
        0.24854757067269784,
    )
    assert int(summary.loc[1.0, "potentially_observable_surplus_roots"]) == 6
    assert pd.read_csv(RESULTS / "geometry.csv").geometry_at_boundary.astype(bool).all()
    verdict = load_report()["verdict"]
    assert verdict["topology_and_position_effects_partially_separable"]
    assert verdict["most_full_strength_gain_occurs_after_topology_change"]
    assert verdict["no_formula_promoted"]


def test_angular_continuation_preserves_nonangular_cross_domain_controls():
    parent = pd.read_csv(
        ROOT / "results" / "p0554_route_softness_interaction" / "variant_scores.csv"
    ).set_index("variant_id")
    radial = parent.loc["lensing_softness_098"]
    combined = parent.loc["combined_parent"]
    assert combined.galaxy_outer_RMSE_km_s == radial.galaxy_outer_RMSE_km_s
    assert combined.cluster_RMSE_dex == radial.cluster_RMSE_dex
    assert combined.Mercury_precession_mas_per_century == radial.Mercury_precession_mas_per_century
    assert bool(combined.all_solar_proxies_pass)
    assert load_report()["route"]["maximum_route_curl_RMS"] < 1.0e-12
