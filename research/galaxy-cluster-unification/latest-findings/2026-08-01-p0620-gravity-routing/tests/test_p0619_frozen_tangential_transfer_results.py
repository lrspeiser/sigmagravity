import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0619_frozen_tangential_transfer"


def test_transfer_is_formula_prospective_and_parameter_free():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["chronology"]["prospective_relative_to_P0618_phase_selection"] is True
    assert report["chronology"]["pristine_project_holdout"] is False
    assert report["locked_formula"]["universal_phase_degrees"] == 90.0
    assert report["locked_formula"]["new_fitted_gravity_parameters"] == 0


def test_baseline_and_candidate_are_scored_for_both_systems():
    scores = pd.read_csv(OUTPUT / "scores.csv")
    assert len(scores) == 4
    assert scores.system_label.nunique() == 2
    assert set(scores.variant_id) == {
        "P0554_scalar_control",
        "P0619_tangential_self_route",
    }


def test_transfer_gate_accounting_never_promotes_this_nonpristine_pair():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["interpretation"]["formula_promoted"] is False
    assert report["interpretation"]["per_object_gravity_parameters"] == 0
    assert "all_transfer_gates_pass" in report["gates"]
