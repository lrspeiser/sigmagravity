import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0564 = ROOT / "results/p0564_baryon_morphology_sign_audit"
P0565 = ROOT / "results/p0565_rxj2129_morphology_gate_transfer"


def load_report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_morphology_audit_is_explicitly_posthoc_and_nonpromotional():
    protocol = json.loads(
        (ROOT / "configs/p0564_baryon_morphology_sign_audit_protocol.json").read_text()
    )
    assert protocol["status"].startswith("frozen_after_p0563_")
    assert "post-hoc" in protocol["interpretation"]
    result = load_report(P0564)
    assert result["primary"]["candidate_gate_nominated"]
    assert not result["primary"]["candidate_gate_validated"]
    assert not result["verdict"]["formula_promoted"]


def test_macs0429_has_a_coherent_core_and_twisted_outer_quadrupole():
    descriptors = pd.read_csv(P0564 / "descriptors.csv")
    values = descriptors.set_index(
        ["system_label", "aperture_arcsec", "component", "descriptor"]
    ).value
    inner = values.loc[
        "MACS0429", 30.0, "joint", "star_gas_normalized_correlation"
    ]
    outer = values.loc[
        "MACS0429", 120.0, "joint", "star_gas_quadrupole_cos2_alignment"
    ]
    assert np.isclose(inner, 0.505048, atol=1e-6)
    assert np.isclose(outer, -0.491632, atol=1e-6)
    positive_inner = [
        values.loc[label, 30.0, "joint", "star_gas_normalized_correlation"]
        for label in ["MACS0329", "MACS1115", "MACS1931"]
    ]
    positive_outer = [
        values.loc[label, 120.0, "joint", "star_gas_quadrupole_cos2_alignment"]
        for label in ["MACS0329", "MACS1115", "MACS1931"]
    ]
    assert inner > max(positive_inner)
    assert outer < min(positive_outer)


def test_rxj2129_morphology_rule_was_frozen_before_its_scores():
    protocol = json.loads(
        (ROOT / "configs/p0565_rxj2129_morphology_gate_transfer_protocol.json").read_text()
    )
    assert protocol["status"].startswith("frozen_before_rxj2129_")
    gate = protocol["sign_gate"]
    assert gate["inner_star_gas_correlation_threshold"] == 0.3278835
    assert gate["outer_quadrupole_cos2_threshold"] == 0.0744005
    assert gate["universal_magnitude"] == 0.3
    assert gate["per_cluster_fitted_gravity_parameters"] == 0


def test_rxj2129_morphology_predicts_its_near_zero_direction():
    result = load_report(P0565)
    morphology = result["morphology"]
    assert np.isclose(morphology["inner_star_gas_correlation"], 0.3932701055)
    assert np.isclose(morphology["outer_quadrupole_cos2_alignment"], -0.1340456088)
    assert morphology["inner_negative_trigger"]
    assert morphology["outer_negative_trigger"]
    assert morphology["predicted_sign"] == "negative"
    assert morphology["predicted_coupling"] == -0.3
    assert {
        row["near_zero_preferred_sign"] for row in result["response_signs"]
    } == {"negative"}
    assert result["gate_audit"][
        "morphology_sign_matches_near_zero_sign_in_both_ensembles"
    ]


def test_rxj2129_exact_effect_is_basin_mixed_so_gate_is_not_validated():
    scores = pd.read_csv(P0565 / "exact_scores.csv")
    candidate = scores[scores.model_id.eq("morphology_gated_t")].set_index("ensemble")
    assert candidate.all_heldout_roots.all()
    assert np.isclose(
        candidate.loc["seed_1", "improvement_fraction_vs_ensemble_zero"],
        -0.012019,
        atol=1e-6,
    )
    assert np.isclose(
        candidate.loc["seed_2", "improvement_fraction_vs_ensemble_zero"],
        0.004761,
        atol=1e-6,
    )
    result = load_report(P0565)
    assert result["gate_audit"]["exact_candidate_all_roots_in_both_ensembles"]
    assert not result["gate_audit"][
        "exact_candidate_improves_zero_in_both_ensembles"
    ]
    assert not result["primary"]["candidate_gate_validated"]
    assert not result["primary"]["formula_promoted"]
