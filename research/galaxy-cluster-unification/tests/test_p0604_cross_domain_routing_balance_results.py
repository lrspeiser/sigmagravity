import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0604_coverage_and_budget_selection():
    base = ROOT / "results/p0604_cross_domain_routing_balance"
    report = json.loads((base / "report.json").read_text())
    assert report["coverage"] == {
        "candidates": 720,
        "clusters": 10,
        "cluster_targets": 2,
        "galaxies": 131,
        "galaxy_outer_points": 968,
        "galaxy_budgets": 3,
        "folds_per_budget": 5,
    }
    assert report["primary_budget"] == 1.05
    assert len(report["primary_fold_selections"]) == 5
    assert all(row["selected_galaxy_RMSE_ratio"] <= 1.05 for row in report["primary_fold_selections"])


def test_p0604_outputs_are_finite_and_complete():
    base = ROOT / "results/p0604_cross_domain_routing_balance"
    candidates = pd.read_csv(base / "candidate_scores.csv")
    folds = pd.read_csv(base / "fold_selections.csv")
    oof = pd.read_csv(base / "oof_scores.csv")
    impacts = pd.read_csv(base / "parameter_impacts.csv")
    assert len(candidates) == 720
    assert len(folds) == 15
    assert len(oof) == 15 * 2 * 5 * 2
    assert len(impacts) == 4
    for frame in (candidates, folds, impacts):
        assert np.all(np.isfinite(frame.select_dtypes(include=["number"])))
    metrics = ["jensen_shannon", "pearson", "normalized_RMSE", "centroid_offset_kpc"]
    assert np.all(np.isfinite(oof[metrics]))
