import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0618_universal_route_phase"


def test_frozen_universal_phase_design_has_no_object_gravity_fits():
    protocol = json.loads(
        (ROOT / "configs/p0618_universal_route_phase_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_after_P0617_before_universal_phase_scores"
    assert report["coverage"]["universal_phases"] == 9
    assert report["coverage"]["new_fitted_gravity_parameters"] == 0
    assert report["interpretation"]["per_system_phase_selection_allowed"] is False


def test_every_phase_scores_every_system():
    scores = pd.read_csv(OUTPUT / "scores.csv")
    responses = pd.read_csv(OUTPUT / "universal_phase_responses.csv")
    assert len(scores) == 50
    assert len(responses) == 9
    assert scores.system_label.nunique() == 5
    assert scores.groupby("system_label").size().eq(10).all()


def test_preference_dispersion_and_gate_accounting_are_recorded():
    preferences = pd.read_csv(OUTPUT / "system_phase_preferences.csv")
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert len(preferences) == 5
    assert 0.0 <= report["preferred_phase_spin2_resultant"] <= 1.0
    assert report["interpretation"]["formula_promoted"] is False
    assert "all_diagnostic_gates_pass" in report["gates"]
