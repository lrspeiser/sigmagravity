import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0571 = ROOT / "results" / "p0571_apparent_peak_baryon_invariant"
P0571B = ROOT / "results" / "p0571b_null_safe_tidal_cancellation"
P0572 = ROOT / "results" / "p0572_tidal_cancellation_arrival_forward"
P0572B = ROOT / "results" / "p0572b_pilot_arrival_transfer"


def report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_p0571_has_the_frozen_search_and_peak_coverage():
    result = report(P0571)
    assert result["coverage"]["systems"] == 10
    assert result["coverage"]["primary_peaks"] == 25
    assert result["coverage"]["method_control_peaks"] == 30
    assert result["coverage"]["candidates"] == 480
    assert result["coverage"]["same_radius_controls_per_peak"] == 71
    assert result["coverage"]["search_null_trials"] == 256


def test_p0571_tidal_balance_clue_is_not_promoted():
    result = report(P0571)
    selected = result["selected"]
    assert selected["feature"] == "tidal_balance"
    assert selected["development_absolute_effect"] > 0.20
    assert selected["validation_signed_effect"] > 0.15
    assert selected["validation_systems_same_direction"] == 3
    assert result["search_control"]["empirical_max_search_p"] > 0.10
    assert not selected["null_safe_feature"]
    assert not result["gates"]["forward_activation_authorized"]


def test_p0571b_null_safe_interaction_transfers_across_peak_controls():
    result = report(P0571B)
    selected = result["selected"]
    assert selected["candidate_id"] == "tidal_cancellation__a0.5__b1"
    assert selected["development_absolute_effect"] > 0.25
    assert selected["pilot_validation_signed_effect"] > 0.40
    assert selected["pilot_systems_same_direction"] == 3
    assert result["search_control"]["empirical_max_search_p"] < 0.01
    assert result["gates"]["forward_activation_authorized"]
    assert result["cross_domain"]["solar_fractional_change"] == 0.0


def test_p0572_prospective_forward_selection_fails_every_transfer_gate():
    result = report(P0572)
    selected = result["selected"]
    assert selected["candidate_id"] == "field_weighted__w100__f1"
    assert selected["holdout_improvement_vs_local_fraction"] < 0.0
    assert selected["holdout_systems_improved"] == 0
    assert selected["holdout_uncertainty_improved_fraction"] == 0.0
    assert selected["glafic_holdout_improvement_vs_local_fraction"] < 0.0
    assert not result["gates"]["raw_lensing_followup_authorized"]


def test_p0572_conserves_axisymmetric_and_solar_nulls():
    result = report(P0572)
    assert result["numerical"]["axisymmetric_activation_RMS"] == 0.0
    assert result["numerical"]["axisymmetric_activation_maximum"] == 0.0
    assert result["cross_domain"]["SPARC_rotation_change_km_s"] == 0.0
    assert result["cross_domain"]["solar_fractional_change"] == 0.0
    assert result["gates"]["axisymmetric_null_pass"]


def test_p0572b_locked_posthoc_candidate_replication_is_explicit():
    result = report(P0572B)
    locked = result["locked_formula"]
    assert locked["carrier"].startswith("tidal_weighted")
    assert locked["arrival_smoothing_kpc"] == 50.0
    assert locked["route_fraction_f"] == 0.8
    outcome = result["result"]
    assert outcome["improvement_vs_local_fraction"] > 0.20
    assert outcome["systems_improved"] == 3
    assert outcome["realizations_improved_fraction"] == 1.0
    assert outcome["selected_mean_Pearson"] > outcome["local_mean_Pearson"]
    assert result["gates"]["fresh_sample_followup_authorized"]


def test_p0572_reports_extent_as_the_largest_forward_coordinate():
    impacts = pd.read_csv(P0572 / "parameter_impacts.csv")
    assert impacts.iloc[0].coordinate == "arrival_smoothing_kpc"
    assert impacts.iloc[0].relative_span > impacts.iloc[1].relative_span
