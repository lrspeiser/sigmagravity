import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0554_local_neighbor_parameter_screen_protocol.json"
RESULTS = ROOT / "results/p0554_local_neighbor_parameter_screen"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_parameter_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_any_distance_or_weight_profile_score" in protocol["status"]
    assert protocol["parent"] == {
        "variant_id": "local_s200_p2_w1",
        "eta": 0.30,
        "local_mix": 1.0,
        "softening_kpc": 200.0,
        "distance_power": 2.0,
        "neighbor_weight_power": 1.0,
        "symmetric_bend_degrees": 0.0,
        "center_mode": "light_centroid",
    }


def test_complete_screen_coverage_and_conservation():
    report = load_report()
    assert report["coverage"] == {
        "variants": 20,
        "systems": 5,
        "variant_system_scores": 100,
        "route_fields": 100,
        "coordinate_profiles": 3,
    }
    assert len(pd.read_csv(RESULTS / "scores.csv")) == 20
    assert len(pd.read_csv(RESULTS / "system_scores.csv")) == 100
    assert len(pd.read_csv(RESULTS / "field_audits.csv")) == 100
    invariants = report["field_invariants"]
    assert invariants["maximum_route_map_normalization_error"] < 1e-14
    assert invariants["maximum_annular_convergence_error"] < 1e-12
    assert invariants["maximum_normalized_curl_RMS"] < 1e-12


def test_softening_is_largest_impact_and_parent_brackets_best_value():
    summary = pd.read_csv(RESULTS / "coordinate_summary.csv").set_index("coordinate")
    assert summary.profile_RMS_span_arcsec.idxmax() == "softening_kpc"
    softening = summary.loc["softening_kpc"]
    assert np.isclose(softening.parent_value, 200.0)
    assert np.isclose(softening.best_value, 200.0)
    assert not bool(softening.best_at_tested_boundary)


def test_inverse_radius_is_worse_and_falloff_power_is_weakest_lever():
    scores = pd.read_csv(RESULTS / "scores.csv").set_index("variant_id")
    assert scores.loc["local_p0100", "primary_improvement_fraction_vs_local_parent"] < 0
    assert np.isclose(
        scores.loc["local_p0100", "primary_improvement_fraction_vs_local_parent"],
        -0.0010276938409805592,
    )
    summary = pd.read_csv(RESULTS / "coordinate_summary.csv").set_index("coordinate")
    assert summary.loc["distance_power", "profile_RMS_span_arcsec"] == summary.profile_RMS_span_arcsec.min()


def test_stronger_light_weighting_has_conflicting_cluster_signs():
    scores = pd.read_csv(RESULTS / "scores.csv").set_index("variant_id")
    weighted = scores.loc["local_w0200"]
    assert weighted.primary_improvement_fraction_vs_local_parent > 0.003
    assert int(weighted.primary_systems_improved) == 2
    assert weighted.minimum_primary_system_improvement_fraction < 0
    impact = next(
        row
        for row in load_report()["coordinate_impact_ranked"]
        if row["coordinate"] == "neighbor_weight_power"
    )
    assert impact["best_at_tested_boundary"]


def test_no_variant_passes_shortlist_or_changes_cross_domain_controls():
    report = load_report()
    assert report["shortlist"] == []
    assert report["verdict"] == {
        "any_variant_meets_exact_followup_shortlist_rule": False,
        "most_impactful_coordinate": "softening_kpc",
        "no_formula_promoted": True,
    }
    controls = report["cross_domain_preservation"]
    assert np.isclose(controls["galaxy_outer_RMSE_km_s"], 12.57091168672948)
    assert np.isclose(controls["CLASH_radial_RMSE_dex"], 0.19641371129844437)
    assert controls["all_solar_proxies_pass"]
