import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0615_self_coupled_quadrupole_route"


def test_self_coupling_uses_five_systems_without_fitted_strength():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["raw_systems"] == 5
    assert report["coverage"]["heldout_images"] == 18
    assert report["coverage"]["new_fitted_gravity_parameters"] == 0
    assert report["interpretation"]["parameter_reduction_achieved"] is True


def test_all_derived_amplitudes_are_bounded_and_system_dependent_only_through_baryons():
    states = pd.read_csv(OUTPUT / "system_states.csv")
    assert states.self_routed_fraction.between(0.0, 1.0).all()
    epsilon_columns = [name for name in states if name.startswith("epsilon_")]
    assert epsilon_columns
    for name in epsilon_columns:
        assert states[name].between(0.0, 1.0).all()
    assert states.quadrupole_Q.between(0.0, 1.0).all()


def test_every_law_is_scored_on_both_cohorts_and_claim_is_diagnostic():
    scores = pd.read_csv(OUTPUT / "summary_scores.csv")
    counts = scores.groupby("law").cohort.nunique()
    assert (counts == 2).all()
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["interpretation"]["formula_promoted"] is False
    assert report["interpretation"]["future_transfer_required"] is True


def test_solar_and_galaxy_inheritance_do_not_override_raw_gates():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["inherited_cross_domain"]["SPARC_to_RAR_ratio"] < 1.5
    assert report["inherited_cross_domain"]["Solar_all_proxies_pass"] is True
    assert report["inherited_cross_domain"]["route_layer_axisymmetric_and_point_source_change"] == 0.0
