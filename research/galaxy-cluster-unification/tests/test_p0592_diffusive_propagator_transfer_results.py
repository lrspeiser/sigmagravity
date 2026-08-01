import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0592_coverage_and_universal_candidates():
    report = json.loads((ROOT / "results/p0592_diffusive_propagator_transfer/report.json").read_text())
    assert report["coverage"] == {"clusters": 10, "development": 7, "holdout": 3, "candidates": 32, "lenstool_realizations": 1000}
    table = pd.read_csv(ROOT / "results/p0592_diffusive_propagator_transfer/candidate_scores.csv")
    assert len(table) == 33
    assert np.all(np.isfinite(table.select_dtypes(include=["number"])))


def test_p0592_backtracking_probabilities_are_valid():
    table = pd.read_csv(ROOT / "results/p0592_diffusive_propagator_transfer/backtracked_peaks.csv")
    assert len(table) > 0
    assert table.top_origin_probability.between(0.0, 1.0).all()
