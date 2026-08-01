import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0554_local_cross_domain_sensitivity"


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_protocol_was_frozen_and_does_not_refit_formula_or_geometry():
    protocol = load_json("configs/p0554_local_cross_domain_sensitivity_protocol.json")
    assert protocol["status"] == "frozen_before_any_new_local_variant_score"
    assert len(protocol["perturbations"]) == 11
    assert "do not refit any formula parameter" in protocol["evaluation"]["universal_q_rule"]
    assert "no geometry or gravity refit" in protocol["evaluation"]["raw_RXJ2129"]
    assert protocol["impact_rules"]["root_change_is_separate_from_RMS"] is True


def test_local_scan_has_complete_declared_coverage():
    report = load_json("results/p0554_local_cross_domain_sensitivity/report.json")
    assert report["coverage"] == {
        "variants": 23,
        "parameters": 11,
        "SPARC_galaxies": 131,
        "SPARC_outer_points": 968,
        "CLASH_systems": 20,
        "CLASH_points": 84,
        "raw_clusters": 5,
        "raw_heldout_images": 18,
    }
    scores = pd.read_csv(RESULTS / "variant_scores.csv")
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    impacts = pd.read_csv(RESULTS / "parameter_impacts.csv")
    assert scores.variant_id.nunique() == 23
    assert len(raw) == 23 * 5
    assert impacts.parameter.nunique() == 11


def test_baseline_and_domain_levers_are_frozen():
    report = load_json("results/p0554_local_cross_domain_sensitivity/report.json")
    baseline = report["baseline"]
    assert np.isclose(baseline["galaxy_outer_RMSE_km_s"], 12.57091168672948)
    assert np.isclose(baseline["cluster_RMSE_dex"], 0.19907732958364824)
    assert np.isclose(baseline["Mercury_precession_mas_per_century"], -1.729829450133022)
    assert baseline["all_solar_proxies_pass"] is True
    assert report["top_parameter_by_domain"] == {
        "galaxy": {"parameter": "alpha", "normalized_span": 0.162467769375058},
        "derived_cluster": {
            "parameter": "mass_radius_delta",
            "normalized_span": 0.3141644773663164,
        },
        "raw_RXJ2129": {
            "parameter": "extent_leak",
            "normalized_span": 1.367141486750286,
        },
        "raw_four_cluster": {
            "parameter": "mass_radius_delta",
            "normalized_span": 0.24246597510450418,
        },
        "solar": {
            "parameter": "screen_exponent",
            "normalized_span": 24.073246804734197,
        },
    }


def test_solar_failures_and_raw_topology_are_not_hidden_by_rms():
    report = load_json("results/p0554_local_cross_domain_sensitivity/report.json")
    scores = pd.read_csv(RESULTS / "variant_scores.csv")
    failed = set(scores.loc[~scores.all_solar_proxies_pass, "variant_id"])
    assert failed == {"alpha_low", "screen_exponent_low", "mass_radius_delta_high"}
    assert report["root_topology"] == {
        "minimum_total_heldout_roots": 13,
        "maximum_total_heldout_roots": 18,
        "total_images": 18,
    }
    impacts = pd.read_csv(RESULTS / "parameter_impacts.csv").set_index("parameter")
    mass_radius = impacts.loc["mass_radius_delta"]
    assert mass_radius.four_cluster_common_complete_systems == 2
    assert mass_radius.four_cluster_low_roots != mass_radius.four_cluster_high_roots
    assert set(report["parameters_with_material_same_direction_across_domains"]) == {
        "extent_leak",
        "screen_scale",
        "invariant_power",
    }
