import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0590_locked_one_candidate_before_holdouts():
    report = json.loads((ROOT / "results/p0590_gravity_return_backtrack/report.json").read_text())
    assert report["status"] == "complete_descriptive_single_cluster_method_transfer"
    candidates = pd.read_csv(ROOT / "results/p0590_gravity_return_backtrack/candidate_scores.csv")
    assert len(candidates) == 140
    best = candidates.sort_values(["development_mean_jsd", "lambda_return_radius", "eta_width_fraction", "routed_fraction"]).iloc[0]
    assert np.isclose(best.development_mean_jsd, report["locked_candidate"]["development_mean_jsd"])
    assert report["method_holdout"]["maps"] == 2


def test_p0590_backtracks_are_finite_probabilities():
    table = pd.read_csv(ROOT / "results/p0590_gravity_return_backtrack/backtracked_peaks.csv")
    assert len(table) >= 8
    assert table.model_id.nunique() == 4
    assert np.all(np.isfinite(table.select_dtypes(include=["number"])))
    assert table.top_origin_group_probability.between(0.0, 1.0).all()
    assert table.top_source_probability.between(0.0, 1.0).all()
    assert (table.maximum_hidden_height_kpc >= 0.0).all()
