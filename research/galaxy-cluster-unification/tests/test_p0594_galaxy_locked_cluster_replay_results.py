import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0594_cluster_replay_coverage_and_locked_candidate():
    report = json.loads((ROOT / "results/p0594_galaxy_locked_cluster_replay/report.json").read_text())
    assert report["coverage"] == {
        "clusters": 10,
        "development": 7,
        "holdout": 3,
        "candidates": 3,
        "lenstool_realizations": 1000,
    }
    assert report["galaxy_locked_candidate"]["q_R80"] == 0.75
    assert report["galaxy_locked_candidate"]["routed_fraction"] == 0.25


def test_p0594_result_tables_are_finite():
    system = pd.read_csv(ROOT / "results/p0594_galaxy_locked_cluster_replay/system_scores.csv")
    uncertainty = pd.read_csv(ROOT / "results/p0594_galaxy_locked_cluster_replay/uncertainty.csv")
    glafic = pd.read_csv(ROOT / "results/p0594_galaxy_locked_cluster_replay/glafic_scores.csv")
    assert len(system) == 30
    assert len(uncertainty) == 1000
    assert len(glafic) == 30
    assert np.all(np.isfinite(system.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(uncertainty.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(glafic.select_dtypes(include=["number"])))
