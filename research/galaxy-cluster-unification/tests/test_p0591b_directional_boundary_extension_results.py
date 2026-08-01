import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0591b_count_and_conclusion():
    candidates = pd.read_csv(ROOT / "results/p0591b_directional_boundary_extension/candidate_scores.csv")
    assert len(candidates) == 200
    report = json.loads((ROOT / "results/p0591b_directional_boundary_extension/report.json").read_text())
    gates = report["gates"]
    passed = gates["development_pass"] and gates["holdout_pass"] and gates["both_holdouts_better_than_gaussian"]
    assert (report["conclusion"] == "boundary_extension_beats_smoothing") == passed
