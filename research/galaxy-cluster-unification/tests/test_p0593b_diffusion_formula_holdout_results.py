import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0593b_formula_partition_and_selection_are_complete():
    report = json.loads((ROOT / "results/p0593b_diffusion_formula_holdout/report.json").read_text())
    assert report["coverage"]["candidates"] == 160
    assert report["coverage"]["discovery_galaxies"] == 91
    assert report["coverage"]["holdout_galaxies"] == 40
    assert report["selected_candidate"]["candidate_id"]


def test_p0593b_result_tables_are_finite_and_holdout_only():
    scores = pd.read_csv(ROOT / "results/p0593b_diffusion_formula_holdout/candidate_scores.csv")
    galaxies = pd.read_csv(ROOT / "results/p0593b_diffusion_formula_holdout/holdout_galaxy_scores.csv")
    predictions = pd.read_csv(ROOT / "results/p0593b_diffusion_formula_holdout/selected_predictions.csv")
    assert len(scores) == 160
    assert len(galaxies) == 40
    assert predictions.galaxy.nunique() == 131
    assert set(predictions.formula_partition) == {"discovery", "formula_holdout"}
    assert np.all(np.isfinite(scores.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(galaxies.select_dtypes(include=["number"])))
