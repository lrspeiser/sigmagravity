from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0644_spent_rxj2129_accumulated_tensor"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_preregistered_failure_and_blindness_are_preserved():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["gate_results"]["training_improvement"] is False
    assert result["gate_results"]["fixed_screen_root_safety"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False


def test_training_only_selection_does_not_leak_spent_holdout():
    selection = report()["selection"]
    assert selection["selected_lambda"] == 0.0
    assert selection["spent_heldout_used_for_selection"] is False
    assert selection["selected_on"] == "training exact-root RMS only"


def test_tensor_field_is_conservative_and_finite():
    audit = report()["field_audit"]
    assert audit["normalized_curl_RMS"] <= 1e-10
    assert audit["source_integral_fraction"] <= 1e-4
    assert audit["unit_deflection_RMS_arcsec"] > 0.0
    assert audit["unit_deflection_maximum_arcsec"] > audit["unit_deflection_RMS_arcsec"]
    assert audit["proxy_mass_changes_deflection_normalization"] is False


def test_lambda_screen_records_root_failure_boundary():
    screen = pd.read_csv(RESULTS / "lambda_screen.csv")
    assert screen["lambda"].tolist() == [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    complete = screen[(screen.training_roots_converged.eq(15)) & (screen.heldout_roots_converged.eq(7))]
    assert complete["lambda"].max() == 5.0
    assert screen.loc[screen["lambda"].eq(20.0), "training_roots_converged"].iloc[0] < 15


def test_spent_holdout_trend_is_recorded_but_not_selected():
    screen = pd.read_csv(RESULTS / "lambda_screen.csv")
    complete = screen[screen["lambda"].le(5.0)]
    assert np.all(np.diff(complete.training_RMS_arcsec) > 0.0)
    assert np.all(np.diff(complete.heldout_RMS_arcsec) < 0.0)
    assert report()["selection"]["selected_lambda"] == 0.0


def test_outputs_are_complete():
    assert (RESULTS / "fixed_geometry_predictions.csv").stat().st_size > 1000
    assert (RESULTS / "selected_refit_predictions.csv").stat().st_size > 1000
    assert (RESULTS / "spent_tensor_screen.png").stat().st_size > 20000
