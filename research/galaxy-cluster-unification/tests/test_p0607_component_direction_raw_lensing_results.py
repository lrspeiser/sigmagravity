import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0607_component_direction_raw_lensing"


def test_component_direction_field_is_conservative_and_mass_claim_is_bounded():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(OUTPUT / "component_audits.csv")
    assert report["coverage"]["components"] == 5
    assert report["coverage"]["screen_variants_including_no_route"] == 41
    assert report["map_readiness"]["absolute_component_mass_ready"] is False
    assert report["map_readiness"]["use_in_this_test"] == "direction only"
    assert audit.route_map_normalization_error.max() < 1.0e-12
    assert audit.maximum_annular_convergence_mean_fraction.max() < 1.0e-12
    assert audit.normalized_curl_RMS.max() < 1.0e-12
    assert report["cross_domain_controls"]["Solar_change"] == 0.0


def test_training_only_selection_does_not_masquerade_as_heldout_success():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    selected = report["fixed_geometry_training_selection"]
    positive = selected["best_positive_route"]
    signed = selected["best_opposite_sign_control"]
    assert positive["component_id"] == "gas"
    assert positive["angular_strength"] == 0.0025
    assert signed["component_id"] == "stars"
    assert signed["angular_strength"] == -0.005
    assert report["interpretation"]["positive_route_beats_no_route_training"] is True
    assert report["interpretation"]["positive_route_beats_no_route_spent_heldout"] is False
    assert report["interpretation"]["opposite_sign_beats_positive_training"] is True
    assert report["interpretation"]["opposite_sign_beats_positive_spent_heldout"] is False


def test_component_and_amplitude_response_is_small_near_zero():
    refits = pd.read_csv(OUTPUT / "refit_scores.csv").set_index("role")
    baseline = refits.loc["P0599_no_route_16_start_reference"]
    positive = refits.loc["positive_route"]
    opposite = refits.loc["opposite_sign_control"]
    assert int(positive.heldout_roots_converged) == 7
    assert int(opposite.heldout_roots_converged) == 7
    assert abs(positive.training_RMS_arcsec / baseline.training_RMS_arcsec - 1.0) < 0.001
    assert positive.heldout_RMS_arcsec > baseline.heldout_RMS_arcsec
    assert opposite.heldout_RMS_arcsec > baseline.heldout_RMS_arcsec
