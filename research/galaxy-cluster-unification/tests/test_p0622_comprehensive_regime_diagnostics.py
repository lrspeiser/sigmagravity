import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0622_comprehensive_regime_diagnostics"


def load_report():
    return json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))


def test_protocol_is_frozen_and_formula_has_no_object_gravity_parameters():
    protocol = json.loads(
        (ROOT / "configs/p0622_comprehensive_regime_diagnostics_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    report = load_report()
    assert protocol["status"] == "frozen_before_P0622_regime_aggregation"
    assert protocol["formula"]["per_object_gravity_parameters"] == 0
    assert report["coverage"]["new_fitted_gravity_parameters"] == 0
    assert report["coverage"]["per_object_gravity_parameters"] == 0
    assert report["decision"]["per_object_phase_selection_used"] is False


def test_suite_reproduces_full_outer_sparc_coverage_and_baseline_score():
    report = load_report()
    points = pd.read_csv(OUTPUT / "galaxy_outer_predictions.csv")
    scores = pd.read_csv(OUTPUT / "galaxy_regime_scores.csv")
    overall = scores[(scores.dimension == "all") & (scores.bin == "all")].iloc[0]
    assert report["coverage"]["SPARC_galaxies"] == 131
    assert report["coverage"]["SPARC_outer_points"] == 968
    assert len(points) == 968
    assert points.galaxy.nunique() == 131
    assert np.isclose(overall.P0554_RMSE_km_s, 12.57091168672948)
    assert np.isclose(overall.fixed_RAR_RMSE_km_s, 10.348465773189677)


def test_galaxy_regimes_expose_bias_reversal_and_deep_acceleration_failure():
    scores = pd.read_csv(OUTPUT / "galaxy_regime_scores.csv")
    mass = scores[scores.dimension.eq("baryonic_mass_family")].set_index("bin")
    acceleration = scores[scores.dimension.eq("outer_acceleration_family")].set_index("bin")
    assert mass.loc["dwarf_mass", "P0554_mean_residual_km_s"] < 0.0
    assert mass.loc["giant_mass", "P0554_mean_residual_km_s"] > 0.0
    assert acceleration.loc["very_deep_below_0p03_a0", "P0554_to_RAR_ratio"] > 1.5
    assert acceleration.loc["very_deep_below_0p03_a0", "P0554_mean_residual_km_s"] < -10.0


def test_interaction_scan_keeps_sample_sizes_and_uncertainty_visible():
    protocol = json.loads(
        (ROOT / "configs/p0622_comprehensive_regime_diagnostics_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    interactions = pd.read_csv(OUTPUT / "galaxy_interaction_scores.csv")
    assert len(interactions) >= 25
    assert interactions.galaxies.min() >= protocol["regimes"]["minimum_interaction_galaxies"]
    assert (interactions.ratio_bootstrap_2p5 <= interactions.P0554_to_RAR_ratio).all()
    assert (interactions.ratio_bootstrap_97p5 >= interactions.P0554_to_RAR_ratio).all()
    assert interactions.P0554_to_RAR_ratio.max() > 1.75


def test_gas_bias_is_mass_confounded_while_inclination_error_signal_survives():
    correlations = pd.read_csv(OUTPUT / "galaxy_residual_correlations.csv")
    gas = correlations[
        correlations.feature.eq("gas_fraction")
        & correlations.target.eq("P0554_mean_velocity_residual")
    ].iloc[0]
    inclination = correlations[
        correlations.feature.eq("inclination")
        & correlations.target.eq("log_P0554_to_RAR_error_ratio")
    ].iloc[0]
    assert gas.p_value < 0.01
    assert gas.partial_p_value_controlling_log_mass > 0.1
    assert inclination.partial_BH_q_value < 0.05
    assert bool(inclination.input_safe_for_blind_prediction)


def test_cluster_phase_mean_is_rxj2129_dominated_and_sign_split():
    clusters = pd.read_csv(OUTPUT / "cluster_regime_scores.csv")
    loo = pd.read_csv(OUTPUT / "cluster_leave_one_out.csv")
    spent = clusters[clusters.evidence_class.eq("spent_diagnostic_fixed_geometry")]
    without_rxj = loo[loo.omitted_system.eq("RXJ2129")].iloc[0]
    assert len(spent) == 5
    assert int((spent.phase90_improvement_fraction >= 0.0).sum()) == 3
    assert int((spent.phase90_improvement_fraction < 0.0).sum()) == 2
    assert spent.phase90_improvement_fraction.mean() > 0.015
    assert abs(without_rxj.mean_phase90_improvement_fraction) < 0.001
    assert clusters.loc[clusters.system_label.eq("MS2137"), "phase90_outcome"].iloc[0] == "root_incomplete"


def test_evidence_provenance_prevents_nulls_and_derived_targets_from_becoming_validation():
    scenarios = pd.read_csv(OUTPUT / "scenario_matrix.csv")
    assert {
        "raw_observation",
        "derived_observation",
        "analytic_proxy",
        "synthetic_invariant",
        "inherited_result",
        "spent_diagnostic",
    }.issubset(set(scenarios.evidence_class))
    nulls = scenarios[scenarios.result.eq("compatible but untested")]
    assert len(nulls) == 2
    assert nulls.diagnostic_lesson.str.contains("cannot validate|nothing about").all()
    ms2137 = scenarios[scenarios.condition.str.contains("MS2137")].iloc[0]
    assert ms2137.result == "root incomplete"


def test_parameter_matrix_and_promotion_gates_record_cross_domain_conflict():
    parameters = pd.read_csv(OUTPUT / "parameter_domain_matrix.csv")
    report = load_report()
    assert len(parameters) >= 19
    assert {
        "P0554_scalar",
        "bounded_route_factorial",
        "self_coupled_route_support",
        "angular_route_phase",
    }.issubset(set(parameters.parameter_family))
    scalar = parameters[parameters.parameter_family.eq("P0554_scalar")]
    assert int(scalar.cross_domain_direction_conflict.astype(bool).sum()) >= 7
    assert report["decision"]["galaxy_parity_gate_pass"] is False
    assert report["decision"]["raw_cluster_halo_gate_pass"] is False
    assert report["decision"]["Solar_proxy_gate_pass"] is True
    assert report["decision"]["universal_phase_sign_gate_pass"] is False
    assert report["decision"]["formula_promoted"] is False
