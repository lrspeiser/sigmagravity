import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0590b_has_all_frozen_candidates_and_families():
    candidates = pd.read_csv(ROOT / "results/p0590b_ring_necessity_audit/candidate_scores.csv")
    families = pd.read_csv(ROOT / "results/p0590b_ring_necessity_audit/family_scores.csv")
    assert len(candidates) == 252
    assert set(families.family) == {"general_return", "gaussian_smoothing_null", "strict_arc_return"}
    assert np.all(np.isfinite(candidates.select_dtypes(include=["number"])))


def test_p0590b_conclusion_matches_gates():
    report = json.loads((ROOT / "results/p0590b_ring_necessity_audit/report.json").read_text())
    gates = report["gates"]
    all_pass = (
        gates["general_nonzero_return_radius"]
        and gates["general_development_improvement_pass"]
        and gates["general_holdout_improvement_pass"]
        and gates["strict_arc_both_holdouts_better_than_gaussian"]
    )
    assert (report["conclusion"] == "nonzero_return_radius_is_morphologically_required") == all_pass
