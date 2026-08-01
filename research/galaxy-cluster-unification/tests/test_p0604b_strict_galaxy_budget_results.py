import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0604b_strict_budget_is_honest_and_complete():
    base = ROOT / "results/p0604b_strict_galaxy_budget"
    report = json.loads((base / "report.json").read_text())
    folds = pd.read_csv(base / "fold_selections.csv")
    oof = pd.read_csv(base / "oof_scores.csv")
    assert report["status"] == "complete_posthoc_strict_budget_whole_cluster_CV"
    assert report["strict_interpretation"]["budgets_predeclared_before_parent_frontier"] is False
    assert report["strict_interpretation"]["fresh_confirmation"] is False
    assert len(folds) == 15
    assert len(oof) == 300
    primary = folds[folds.galaxy_RMSE_ratio_budget.eq(1.0)]
    assert len(primary) == 5
    assert np.all(primary.selected_galaxy_RMSE_ratio <= 1.0)
    metrics = ["jensen_shannon", "pearson", "normalized_RMSE", "centroid_offset_kpc"]
    assert np.all(np.isfinite(oof[metrics]))
