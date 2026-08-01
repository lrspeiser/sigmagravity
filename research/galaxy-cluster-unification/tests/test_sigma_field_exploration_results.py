import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sigma_field_exploration"


def test_sigma_field_report_preserves_exploratory_scores_and_scope():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    best = report["best_descriptive_grid_row"]

    assert report["grid_rows"] == 125
    assert report["grid_convergence"] == {
        "both_spherical_solutions_converged": 120,
        "not_both_converged": 5,
    }
    assert best["eta"] == pytest.approx(0.8)
    assert best["log10_rho_s_g_cm3"] == pytest.approx(-23.5)
    assert best["L_Sigma_kpc"] == pytest.approx(3.0)
    assert best["joint_descriptive_score_dex"] == pytest.approx(0.2132291754)
    assert report["scope"]["relativistic_completion"] is False
    assert report["scope"]["RXJ2129_independent_validation"] is False


def test_sigma_field_artifacts_contain_complete_grid_and_raw_lens_images():
    grid = pd.read_csv(RESULTS / "parameter_grid.csv")
    lens = pd.read_csv(RESULTS / "raw_lensing_predictions.csv")

    assert len(grid) == 125
    assert grid[["eta", "log10_rho_s_g_cm3", "L_Sigma_kpc"]].drop_duplicates().shape[0] == 125
    assert (grid["galaxy_sigma_converged"] & grid["cluster_sigma_converged"]).sum() == 120
    assert set(lens["stage"]) == {"training", "heldout"}
    assert (lens["stage"] == "training").sum() == 15
    assert (lens["stage"] == "heldout").sum() == 7
