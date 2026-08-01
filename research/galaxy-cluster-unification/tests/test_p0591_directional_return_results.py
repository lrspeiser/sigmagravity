import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0591_candidate_count_and_controls():
    candidates = pd.read_csv(ROOT / "results/p0591_directional_return/candidate_scores.csv")
    scores = pd.read_csv(ROOT / "results/p0591_directional_return/map_scores.csv")
    assert len(candidates) == 315
    assert scores.model_id.nunique() == 4
    assert set(scores.prediction) == {
        "directional_return",
        "gaussian_smoothing",
        "strict_isotropic_ring",
        "axis_rotated_45deg",
    }
    assert np.all(np.isfinite(candidates.select_dtypes(include=["number"])))


def test_p0591_conclusion_matches_all_gates():
    report = json.loads((ROOT / "results/p0591_directional_return/report.json").read_text())
    gates = report["gates"]
    all_pass = (
        gates["development_improvement_pass"]
        and gates["holdout_improvement_pass"]
        and gates["both_holdouts_better_than_gaussian"]
        and gates["baryon_axis_better_than_45deg_rotated_on_holdout"]
    )
    assert (report["conclusion"] == "directional_arc_survives_smoothing_null") == all_pass
