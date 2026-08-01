import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/p0620_parameter_impact_synthesis"


def test_synthesis_identifies_distinct_kinds_of_impact():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    findings = report["most_impactful_findings"]
    assert findings["most_recurrent"] == "spatial width/support"
    assert findings["largest_new_lens_response"] == "universal angular phase"
    assert findings["most_explosive_but_destructive"] == "routed fraction/strength"


def test_cross_domain_scorecard_and_parameter_table_are_complete():
    parameters = pd.read_csv(OUTPUT / "parameter_findings.csv")
    scorecard = pd.read_csv(OUTPUT / "cross_domain_scorecard.csv")
    assert len(parameters) == 8
    assert len(scorecard) == 6
    assert {"galaxy rotation", "Solar Mercury", "A383 chronological formula transfer"}.issubset(
        set(scorecard.domain)
    )


def test_stage_is_synthesized_without_promoting_the_formula():
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["decision"]["stage_objective_met"] is True
    assert report["decision"]["formula_promoted"] is False
    assert report["current_formula"]["phase"] == "+90 degrees shared by all systems"
