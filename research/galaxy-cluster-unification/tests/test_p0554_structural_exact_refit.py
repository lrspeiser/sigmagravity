import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0554_structural_exact_refit"


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_protocol_is_frozen_and_refits_only_ordinary_geometry():
    protocol = load_json("configs/p0554_structural_exact_refit_protocol.json")
    assert protocol["status"] == (
        "frozen_after_structural_screen_before_any_selected_exact_refit_score"
    )
    assert len(protocol["variants"]) == 8
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["geometry_parameters_refit_per_cluster"] == 6
    assert protocol["evaluation"]["optimization_starts_per_variant_system"] == 8
    assert protocol["interpretation_rules"]["all_raw_roots_required_for_accuracy_claim"] is True


def test_exact_refit_outputs_have_complete_fit_coverage():
    report = load_json("results/p0554_structural_exact_refit/report.json")
    assert report["coverage"] == {
        "variants": 8,
        "SPARC_galaxies": 131,
        "CLASH_systems": 20,
        "raw_clusters": 5,
        "raw_heldout_images": 18,
        "geometry_refit_starts": 8,
        "exact_geometry_fits": 40,
    }
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    geometry = pd.read_csv(RESULTS / "geometry.csv")
    assert len(raw) == 8 * 5
    assert len(geometry) == 8 * 5
    assert raw.groupby("variant_id").heldout_images.sum().eq(18).all()


def test_no_fixed_geometry_root_recovery_survives_exact_refitting():
    report = load_json("results/p0554_structural_exact_refit/report.json")
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    aggregate = raw.groupby("variant_id").agg(
        roots=("heldout_roots_converged", "sum"),
        complete=("heldout_all_roots", "sum"),
    )
    assert aggregate.roots.eq(17).all()
    assert aggregate.complete.eq(4).all()
    assert report["complete_solar_safe_ranked_by_all_five_RMS"] == []
    assert report["verdict"] == {
        "any_complete_solar_safe_variant": False,
        "no_variant_promoted": True,
    }


def test_lensing_softness_is_coherent_on_complete_lensing_systems_but_not_validation():
    scores = pd.read_csv(RESULTS / "variant_scores.csv").set_index("variant_id")
    assert scores.loc["lensing_softness_098", "galaxy_outer_RMSE_km_s"] == scores.loc[
        "baseline", "galaxy_outer_RMSE_km_s"
    ]
    assert scores.loc["lensing_softness_098", "cluster_RMSE_dex"] < scores.loc[
        "baseline", "cluster_RMSE_dex"
    ]
    assert scores.loc[
        "lensing_softness_098", "Mercury_precession_mas_per_century"
    ] == scores.loc["baseline", "Mercury_precession_mas_per_century"]

    matched = pd.read_csv(RESULTS / "matched_comparisons.csv")
    rows = matched[matched.variant_id.eq("lensing_softness_098")].set_index("scope")
    assert np.isclose(rows.loc["RXJ2129", "matched_improvement_fraction"], 0.1133908)
    assert np.isclose(rows.loc["four_cluster", "matched_improvement_fraction"], 0.021045)
    assert rows.loc["historical_validation", "matched_improvement_fraction"] < 0.0
    assert rows.loc["all_five", "candidate_total_roots"] == 17


def test_dynamics_softness_low_has_largest_matched_gain_and_a_galaxy_tradeoff():
    matched = pd.read_csv(RESULTS / "matched_comparisons.csv")
    all_five = matched[matched.scope.eq("all_five")].set_index("variant_id")
    candidates = all_five.drop(index="baseline")
    assert candidates.matched_improvement_fraction.idxmax() == "dynamics_softness_098"
    assert np.isclose(
        all_five.loc["dynamics_softness_098", "matched_improvement_fraction"],
        0.024972,
    )
    scores = pd.read_csv(RESULTS / "variant_scores.csv").set_index("variant_id")
    assert scores.loc["dynamics_softness_098", "galaxy_outer_RMSE_km_s"] > scores.loc[
        "baseline", "galaxy_outer_RMSE_km_s"
    ]
    assert scores.loc["dynamics_softness_098", "cluster_RMSE_dex"] < scores.loc[
        "baseline", "cluster_RMSE_dex"
    ]
    assert scores.loc["dynamics_softness_098", "all_solar_proxies_pass"] == np.True_


def test_screen_and_potential_fixed_directions_are_absorbed_or_reversed():
    joined = pd.read_csv(RESULTS / "fixed_vs_refit.csv")
    rx = joined[joined.scope.eq("RXJ2129")].set_index("variant_id")
    assert rx.loc["screen_softness_098", "fixed_direction"] == "improves"
    assert rx.loc["screen_softness_098", "refit_direction"] == "worsens"
    assert rx.loc["potential_scale_plus_001", "fixed_direction"] == "improves"
    assert rx.loc["potential_scale_plus_001", "refit_direction"] == "worsens"
    four = joined[joined.scope.eq("four_cluster")].set_index("variant_id")
    assert four.loc["lensing_softness_098", "fixed_direction"] == "worsens"
    assert four.loc["lensing_softness_098", "refit_direction"] == "improves"


def test_geometry_and_comparator_claim_boundaries_remain_explicit():
    report = load_json("results/p0554_structural_exact_refit/report.json")
    assert report["geometry_boundary_fits"] == 30
    assert np.isclose(report["historical_validation_compact_halo_RMS_arcsec"], 9.989136027113078)
    assert any("spent" in item for item in report["claim_limits"])
