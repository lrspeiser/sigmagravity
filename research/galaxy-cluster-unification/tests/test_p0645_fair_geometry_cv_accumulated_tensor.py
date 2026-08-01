from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0645_fair_geometry_cv_accumulated_tensor"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_fair_refit_rejects_nonzero_tensor_without_unsealing_targets():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["selection"]["selected_lambda"] == 0.0
    assert result["gate_results"]["CV_improvement"] is False
    assert result["gate_results"]["nonzero_lambda"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False


def test_every_source_family_remains_represented_in_each_fit():
    definitions = report()["fold_definition"]
    assert len(definitions) == 5
    all_validation = []
    for row in definitions:
        assert len(row["fit_images"]) + len(row["validation_images"]) == 15
        assert 2 <= len(row["validation_images"]) <= 4
        all_validation.extend(row["validation_images"])
    assert len(all_validation) == 15
    assert len(set(all_validation)) == 15


def test_lambda_zero_has_best_root_complete_cv_score():
    scores = pd.read_csv(RESULTS / "lambda_scores.csv")
    complete = scores[scores.all_CV_roots]
    selected = complete.sort_values(["pooled_CV_RMS_arcsec", "lambda"]).iloc[0]
    assert selected["lambda"] == 0.0
    assert np.isclose(
        selected.pooled_CV_RMS_arcsec,
        report()["selection"]["lambda0_CV_RMS_arcsec"],
    )
    assert scores.loc[scores["lambda"].eq(0.5), "CV_roots"].iloc[0] == 13


def test_all_conventional_geometry_was_refit_for_each_fold():
    coverage = report()["coverage"]
    assert coverage["CV_folds"] == 5
    assert coverage["lambda_rows"] == 6
    assert coverage["lambda_fold_refits"] == 30
    assert coverage["ordinary_geometry_parameters_refit_per_run"] == 6
    assert coverage["per_object_spatial_gravity_parameters"] == 0


def test_final_zero_tensor_control_is_numerically_safe():
    final = report()["full_refit"]
    assert final["training_roots"] == 15
    assert final["spent_heldout_roots"] == 7
    assert not any(final["near_bound"].values())
    assert report()["gate_results"]["spent_heldout_not_worse"] is True
    assert report()["gate_results"]["geometry_interior"] is True


def test_machine_readable_outputs_are_complete():
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    predictions = pd.read_csv(RESULTS / "cv_predictions.csv")
    assert len(folds) == 30
    assert set(folds["lambda"]) == {0.0, 0.5, 1.0, 2.0, 3.5, 5.0}
    assert predictions["lambda"].nunique() == 6
    assert (RESULTS / "full_refit_predictions.csv").stat().st_size > 1000
    assert (RESULTS / "fair_geometry_cv.png").stat().st_size > 20000
