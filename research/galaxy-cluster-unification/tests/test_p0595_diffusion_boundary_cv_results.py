import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0595_whole_galaxy_cv_coverage():
    report = json.loads((ROOT / "results/p0595_diffusion_boundary_cv/report.json").read_text())
    assert report["coverage"] == {"galaxies": 131, "outer_points": 968, "candidates": 216, "folds": 5}
    selections = pd.read_csv(ROOT / "results/p0595_diffusion_boundary_cv/fold_selections.csv")
    assert len(selections) == 5
    assert set(selections.fold) == set(range(5))


def test_p0595_result_tables_are_finite():
    candidates = pd.read_csv(ROOT / "results/p0595_diffusion_boundary_cv/candidate_fold_scores.csv")
    galaxies = pd.read_csv(ROOT / "results/p0595_diffusion_boundary_cv/galaxy_scores.csv")
    associations = pd.read_csv(ROOT / "results/p0595_diffusion_boundary_cv/morphology_associations.csv")
    assert len(candidates) == 216 * 5
    assert len(galaxies) == 131
    assert len(associations) == 8
    assert np.all(np.isfinite(candidates.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(galaxies[["oof_RMSE_km_s", "fixed_RAR_RMSE_km_s", "delta_MSE_km_s2"]]))
    assert np.all(np.isfinite(associations.select_dtypes(include=["number"])))
