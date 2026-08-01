import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0616_frozen_self_coupled_transfer"


def test_transfer_is_chronologically_prospective_and_parameter_free():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["chronology"]["prospective_relative_to_P0615_formula"] is True
    assert report["chronology"]["pristine_project_holdout"] is False
    assert report["locked_formula"]["new_fitted_gravity_parameters"] == 0
    assert report["coverage"]["systems"] == 2


def test_derived_amplitudes_are_finite_and_not_object_fits():
    states = pd.read_csv(OUTPUT / "system_states.csv")
    assert states.selected_epsilon.between(0.0, 1.0).all()
    assert states.self_routed_fraction.between(0.0, 1.0).all()
    assert states.Delta80.gt(0.0).all()


def test_score_pairs_and_gate_accounting_are_complete():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(OUTPUT / "scores.csv")
    assert len(scores) == 4
    assert set(scores.variant_id) == {
        "P0554_scalar_control",
        "P0615_quadratic_self_route",
    }
    assert "all_transfer_gates_pass" in report["gates"]
    assert report["interpretation"]["per_object_gravity_parameters"] == 0
    assert report["interpretation"]["formula_promoted"] is False
    assert report["coverage"]["complete_matched_systems"] == 1
    assert report["responses"][0]["heldout_improvement_fraction"] < 0.0
