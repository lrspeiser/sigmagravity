import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0554_all_baryon_route_screen_protocol.json"
RESULTS = ROOT / "results/p0554_all_baryon_route_screen"


def load_report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_screen_was_frozen_before_map_construction_or_scores():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert protocol["status"].startswith("frozen_")
    assert "before_map_construction_or_any_route_score" in protocol["status"]
    assert len(protocol["variants"]) == 10
    assert protocol["parent"]["new_per_cluster_gravity_parameters"] == 0
    assert protocol["parent"]["direction_softening_kpc"] == 200.0


def test_complete_screen_and_conservative_fields():
    report = load_report()
    assert report["coverage"] == {
        "variants": 10,
        "systems": 5,
        "variant_system_scores": 50,
        "route_fields": 50,
        "registered_proxy_maps": 10,
    }
    assert len(pd.read_csv(RESULTS / "system_scores.csv")) == 50
    assert len(pd.read_csv(RESULTS / "map_audits.csv")) == 10
    invariants = report["field_invariants"]
    assert invariants["maximum_route_map_normalization_error"] < 1e-14
    assert invariants["maximum_annular_convergence_error"] < 1e-12
    assert invariants["maximum_normalized_curl_RMS"] < 1e-12


def test_member_direction_beats_every_registered_map_direction():
    scores = pd.read_csv(RESULTS / "scores.csv").set_index("variant_id")
    assert scores.primary_equal_system_RMS_arcsec.idxmin() == "member_parent"
    alternatives = scores.drop(index=["eta_000", "member_parent"])
    assert (alternatives.primary_improvement_fraction_vs_member_parent < 0.0).all()
    assert np.isclose(
        scores.loc["star_continuous", "primary_improvement_fraction_vs_member_parent"],
        -0.00195430019712739,
    )
    assert np.isclose(
        scores.loc["gas_masked", "primary_improvement_fraction_vs_member_parent"],
        -0.0036664370757815146,
    )


def test_maps_still_beat_no_route_but_do_not_transfer_consistently():
    scores = pd.read_csv(RESULTS / "scores.csv").set_index("variant_id")
    map_variants = scores.drop(index=["eta_000", "member_parent"])
    assert (map_variants.primary_improvement_fraction_vs_eta0 > 0.0).all()
    assert (map_variants.primary_systems_improved_vs_member_parent <= 1).all()
    assert load_report()["shortlist"] == []


def test_direction_alignment_exposes_cluster_structure_differences():
    directions = pd.read_csv(RESULTS / "direction_audits.csv")
    star = directions[directions.direction_kind.eq("star")].set_index("system_label")
    assert star.loc["RXJ2129", "mean_alignment_with_member"] > 0.98
    assert star.loc["MACS0329", "mean_alignment_with_member"] > 0.96
    assert star.loc["MACS1931", "mean_alignment_with_member"] < 0.60
    gas = directions[directions.direction_kind.eq("gas_masked")].set_index("system_label")
    assert gas.loc["MACS0429", "mean_alignment_with_member"] < 0.42


def test_cross_domain_controls_remain_fixed_and_nothing_is_promoted():
    report = load_report()
    controls = report["cross_domain_preservation"]
    assert np.isclose(controls["galaxy_outer_RMSE_km_s"], 12.57091168672948)
    assert np.isclose(controls["CLASH_radial_RMSE_dex"], 0.19641371129844437)
    assert controls["all_solar_proxies_pass"]
    assert report["verdict"] == {
        "any_map_direction_beats_member_parent": False,
        "any_variant_meets_exact_followup_rule": False,
        "no_formula_promoted": True,
    }
