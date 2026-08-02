from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0650_amplitude_free_spent_lensing"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_predictive_failure_is_the_only_rejection_reason():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_robustness"] is False
    assert sum(result["gate_results"].values()) == 10
    assert result["gate_results"]["CV_improvement"] is False
    assert result["gate_results"]["beats_matched_multipole"] is False
    assert all(
        value
        for name, value in result["gate_results"].items()
        if name not in {"CV_improvement", "beats_matched_multipole"}
    )


def test_amplitude_is_formula_defined_not_fit_or_searched():
    result = report()
    assert result["coverage"]["candidate_fields"] == 1
    assert result["coverage"]["amplitude_rows"] == 1
    assert result["coverage"]["fitted_field_amplitude_parameters"] == 0
    assert result["coverage"]["per_object_spatial_gravity_parameters"] == 0
    assert result["gate_results"]["amplitude_is_one"] is True


def test_natural_magnitude_is_large_but_spatial_cv_is_poor():
    result = report()
    assert result["field_audit"]["mismatch_mode"] == "linear_chord_mix"
    assert result["field_audit"]["unit_deflection_RMS_arcsec"] > 0.78
    comparison = result["comparison"]
    assert comparison["CV_improvement_fraction_vs_lambda0"] < 0.0
    assert comparison["CV_improvement_fraction_vs_best_matched_multipole"] < -0.09
    assert comparison["P0601_spent_heldout_used_for_selection"] is False


def test_all_roots_converge_but_folds_are_heterogeneous():
    folds = pd.read_csv(RESULTS / "fold_scores.csv").set_index("fold")
    assert int(folds.validation_roots.sum()) == 15
    assert folds.loc[[0, 1], "validation_RMS_arcsec"].min() > 3.8
    assert folds.loc[[2, 3, 4], "validation_RMS_arcsec"].max() < 1.5
    assert report()["full_refit"]["training_roots"] == 15
    assert report()["full_refit"]["spent_heldout_roots"] == 7


def test_descriptive_spent_holdout_does_not_override_cv():
    full = report()["full_refit"]
    assert full["spent_heldout_worsening_fraction_vs_P0599"] < -0.10
    assert np.isclose(full["spent_heldout_RMS_arcsec"], 1.6214739823437934)
    assert report()["candidate_advanced_to_robustness"] is False


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0650_amplitude_free_spent_lensing.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0650_amplitude_free_spent_lensing.py"
    )
    assert (RESULTS / "amplitude_free_spent_lensing.png").stat().st_size > 20000
