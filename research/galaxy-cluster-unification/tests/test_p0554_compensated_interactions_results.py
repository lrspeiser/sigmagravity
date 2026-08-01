import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0554_compensated_interactions"


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_interaction_protocol_refits_only_ordinary_geometry():
    protocol = load_json("configs/p0554_compensated_interactions_protocol.json")
    assert protocol["status"] == (
        "frozen_after_local_fixed_geometry_screen_before_any_compensated_interaction_refit"
    )
    assert len(protocol["variants"]) == 12
    assert protocol["evaluation"]["formula_parameters_fit"] == 0
    assert protocol["evaluation"]["geometry_parameters_refit_per_cluster"] == 6
    assert protocol["evaluation"]["optimization_starts_per_variant_system"] == 8
    assert protocol["interpretation_rules"]["all_raw_roots_required_for_accuracy_claim"] is True


def test_exact_refit_outputs_cover_every_variant_and_cluster():
    report = load_json("results/p0554_compensated_interactions/report.json")
    raw = pd.read_csv(RESULTS / "raw_system_scores.csv")
    geometry = pd.read_csv(RESULTS / "geometry.csv")
    assert report["coverage"] == {
        "variants": 12,
        "SPARC_galaxies": 131,
        "CLASH_systems": 20,
        "raw_clusters": 5,
        "raw_heldout_images": 18,
        "geometry_refit_starts": 8,
    }
    assert len(raw) == 12 * 5
    assert len(geometry) == 12 * 5
    assert raw.groupby("variant_id").heldout_images.sum().eq(18).all()


def test_sharper_screen_restores_solar_safety_but_not_galaxy_accuracy():
    report = load_json("results/p0554_compensated_interactions/report.json")
    check = report["Solar_compensation_check"]
    assert check["alpha_low_pass"] is False
    assert check["compensated_pass"] is True
    assert np.isclose(check["alpha_low_Mercury_mas_per_century"], -3.50999045541729)
    assert np.isclose(check["compensated_Mercury_mas_per_century"], -0.07998365989054894)
    scores = pd.read_csv(RESULTS / "variant_scores.csv").set_index("variant_id")
    assert (
        scores.loc["alpha_screen_compensated", "galaxy_outer_RMSE_km_s"]
        > scores.loc["baseline", "galaxy_outer_RMSE_km_s"]
    )


def test_best_complete_variant_recovers_root_and_only_modestly_improves_common_systems():
    report = load_json("results/p0554_compensated_interactions/report.json")
    ranked = report["complete_solar_safe_variants_ranked_by_all_five_RMS"]
    assert [row["variant_id"] for row in ranked] == [
        "extent_screen_scale",
        "invariant_power_low",
        "screen_scale_low",
        "extent_apogee",
    ]
    best = ranked[0]
    assert best["candidate_complete_systems"] == 5
    assert best["candidate_total_roots"] == 18
    assert best["recovered_systems"] == "MACS1931"
    assert np.isclose(best["candidate_finite_only_RMS_arcsec"], 16.382711057024135)

    matched = pd.read_csv(RESULTS / "matched_comparisons.csv")
    all_five = matched[
        matched.scope.eq("all_five") & matched.variant_id.eq("extent_screen_scale")
    ].iloc[0]
    assert all_five.matched_complete_systems == 4
    assert "MACS1931" not in all_five.matched_labels
    assert np.isclose(all_five.matched_improvement_fraction, 0.03228469597131367)
    assert all_five.recovered_systems == "MACS1931"


def test_best_descriptive_variant_remains_worse_than_compact_halo_and_is_not_promoted():
    report = load_json("results/p0554_compensated_interactions/report.json")
    best = report["complete_solar_safe_variants_ranked_by_all_five_RMS"][0]
    compact = report["historical_validation_compact_halo_RMS_arcsec"]
    comparisons = pd.read_csv(RESULTS / "matched_comparisons.csv")
    validation = comparisons[
        comparisons.scope.eq("historical_validation")
        & comparisons.variant_id.eq("extent_screen_scale")
    ].iloc[0]
    assert validation.matched_complete_systems == 1
    assert validation.matched_improvement_fraction < 0.0
    assert validation.candidate_finite_only_RMS_arcsec > compact
    assert best["galaxy_outer_RMSE_km_s"] > report["baseline"]["galaxy_outer_RMSE_km_s"]
    assert report["verdict"] == {
        "any_complete_solar_safe_variant": True,
        "no_variant_promoted": True,
    }
