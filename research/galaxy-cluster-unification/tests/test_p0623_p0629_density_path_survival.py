from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def test_p0623_broad_screen_coverage_and_winner():
    report = read_json("results/p0623_density_path_survival/report.json")
    assert report["counts"] == {
        "galaxies": 131,
        "outer_points": 968,
        "features": 44,
        "candidate_formulas": 485,
        "candidate_fold_fits": 1940,
    }
    assert report["selection"]["best_inverse_candidate"] == (
        "inverse_hill0_m1__potential_depth"
    )
    assert report["selection"]["inverse_development_gate_pass"] is True


def test_p0623_feature_provenance_is_baryonic_and_potential_is_local():
    features = read_csv("results/p0623_density_path_survival/feature_catalog.csv")
    assert len(features) == 44
    assert not features.uses_observed_velocity.astype(bool).any()
    potential = features[features.feature.eq("potential_depth")].iloc[0]
    assert bool(potential.varies_with_radius)
    assert potential.kind == "local_field_invariant"


def test_p0623_true_pair_and_path_features_survive_multiple_folds():
    scores = read_csv("results/p0623_density_path_survival/cv_candidate_scores.csv")
    pair = scores[scores.candidate_id.eq("inverse_loglinear__pair_count_L30p0kpc")].iloc[0]
    path = scores[
        scores.candidate_id.eq("inverse_hillfloor_m2__outward_radial_column")
    ].iloc[0]
    assert pair.fold_wins == 4
    assert pair.improvement_vs_constant_fraction > 0.08
    assert path.fold_wins == 4
    assert path.improvement_vs_constant_fraction > 0.08


def test_p0624_potential_only_fails_deep_cluster_transfer():
    scores = read_csv("results/p0624_deep_porous_cross_domain/derived_cluster_scores.csv")
    constant = scores[scores.candidate_id.eq("constant")].iloc[0]
    potential = scores[
        scores.candidate_id.eq("inverse_hill0_m1__potential_depth")
    ].iloc[0]
    assert potential.cluster_equal_system_RMSE_dex > 0.32
    assert potential.cluster_equal_system_RMSE_dex > 1.6 * constant.cluster_equal_system_RMSE_dex


def test_p0624_unbounded_pair_extrapolation_fails_solar_and_raw_topology():
    solar = read_csv("results/p0624_deep_porous_cross_domain/solar_scores.csv")
    pair = solar[
        solar.candidate_id.eq("inverse_loglinear__pair_surface_L30p0kpc")
    ].iloc[0]
    assert pair.q_mercury_max == 6.0
    assert not bool(pair.all_solar_proxies_pass)
    raw = read_csv("results/p0624_deep_porous_cross_domain/raw_cluster_scores.csv")
    block = raw[raw.candidate_id.eq("inverse_loglinear__pair_surface_L30p0kpc")]
    assert int(block.heldout_roots_converged.sum()) == 12


def test_p0625_or_rule_is_the_first_cross_domain_survivor_but_not_raw_winner():
    report = read_json("results/p0625_bounded_porosity_survival/report.json")
    rows = {row["candidate_id"]: row for row in report["candidates"]}
    survivor = rows["OR_max_potential_surface"]
    assert survivor["cross_domain_diagnostic_gate"] is True
    assert survivor["galaxy_cv_improvement_fraction"] > 0.10
    raw = read_csv("results/p0625_bounded_porosity_survival/raw_cluster_scores.csv")
    control = raw[raw.candidate_id.eq("constant") & raw.heldout_all_roots.astype(bool)]
    candidate = raw[
        raw.candidate_id.eq("OR_max_potential_surface") & raw.heldout_all_roots.astype(bool)
    ]
    assert np.sqrt(np.mean(candidate.heldout_RMS_arcsec**2)) > np.sqrt(
        np.mean(control.heldout_RMS_arcsec**2)
    )


def test_p0627_has_only_two_opened_data_rule_passers():
    report = read_json("results/p0627_or_strength_phase_atlas/report.json")
    assert report["cross_domain_rule_passers"] == [
        {"beta": 0.5, "phase_degrees": -67.5},
        {"beta": 0.5, "phase_degrees": -45.0},
    ]
    atlas = read_csv("results/p0627_or_strength_phase_atlas/atlas_summary.csv")
    selected = atlas[np.isclose(atlas.beta, 0.5) & np.isclose(atlas.phase_degrees, -67.5)].iloc[0]
    assert selected.roots == 18
    assert selected.improvement_vs_original_P0618_scalar_fraction > 0.0


def test_p0628_oof_gain_is_repeatable_but_does_not_beat_rar_or_mond():
    scores = read_csv("results/p0628_selected_density_route_synthesis/galaxy_oof_scores.csv")
    indexed = scores.set_index("model")
    selected = indexed.loc["P0628_selected_OOF", "equal_galaxy_RMSE_km_s"]
    constant = indexed.loc["fold_refit_constant_OOF", "equal_galaxy_RMSE_km_s"]
    rar = indexed.loc["fixed_RAR_same_nuisance", "equal_galaxy_RMSE_km_s"]
    mond = indexed.loc["simple_MOND_inner_refit", "equal_galaxy_RMSE_km_s"]
    assert selected < constant
    assert selected > rar
    assert selected > mond
    bootstrap = read_csv("results/p0628_selected_density_route_synthesis/galaxy_bootstrap.csv")
    low, high = np.quantile(
        bootstrap.selected_improvement_vs_constant_fraction, [0.025, 0.975]
    )
    assert low > 0.049
    assert high < 0.105


def test_p0628_density_split_is_only_partly_repaired():
    regimes = read_csv("results/p0628_selected_density_route_synthesis/galaxy_regime_scores.csv")
    mass = regimes[regimes.dimension.eq("mass_regime")].set_index("regime")
    dwarf = mass.loc["dwarf_below_1e9"]
    giant = mass.loc["giant_above_1e10"]
    assert abs(dwarf.selected_mean_galaxy_residual_km_s) < abs(
        dwarf.constant_mean_galaxy_residual_km_s
    )
    assert abs(giant.selected_mean_galaxy_residual_km_s) > abs(
        giant.constant_mean_galaxy_residual_km_s
    )
    assert dwarf.selected_improvement_vs_constant_fraction > 0.17


def test_p0628_raw_candidate_remains_far_from_limited_compact_halo():
    scores = read_csv("results/p0628_selected_density_route_synthesis/cross_domain_scorecard.csv")
    halo = scores[scores.comparator.eq("limited compact halo historical validation")].iloc[0]
    assert halo.ratio > 2.0


def test_p0629_hierarchy_does_not_clear_all_rules():
    report = read_json("results/p0629_hierarchical_density_survival/report.json")
    assert report["diagnostic_passers"] == []
    best = report["best_row"]
    assert best["raw_roots"] == 18
    assert best["dwarf_bias_improves"] is True
    assert best["giant_bias_improves"] is False
    assert best["all_diagnostic_rules_pass"] is False
