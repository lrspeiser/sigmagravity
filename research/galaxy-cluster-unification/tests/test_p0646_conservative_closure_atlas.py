from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0646_conservative_closure_atlas"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_boundary_limited_result_and_blindness_are_preserved():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert sum(result["gate_results"].values()) == 9
    assert result["gate_results"]["lambda_not_endpoint"] is False
    assert all(
        value for name, value in result["gate_results"].items() if name != "lambda_not_endpoint"
    )
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False


def test_gas_minus_star_flux_is_selected_and_beats_controls():
    selection = report()["selection"]
    assert selection["closure"] == "gas_minus_star_flux"
    assert selection["lambda"] == 5.0
    assert selection["improvement_fraction_vs_lambda0"] >= 0.01
    assert selection["improvement_fraction_vs_isotropic"] >= 0.005
    assert selection["P0601_spent_heldout_used_for_selection"] is False


def test_corrected_baseline_is_finite_and_matches_p0645():
    selection = report()["selection"]
    p0645 = json.loads(
        (ROOT / "results/p0645_fair_geometry_cv_accumulated_tensor/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert np.isfinite(selection["lambda0_CV_RMS_arcsec"])
    assert np.isclose(
        selection["lambda0_CV_RMS_arcsec"],
        p0645["selection"]["lambda0_CV_RMS_arcsec"],
    )


def test_sign_and_isotropic_controls_are_informative():
    stage1 = pd.read_csv(RESULTS / "stage1_scores.csv")
    gas = stage1[stage1.closure.eq("gas_minus_star_flux")].sort_values("lambda")
    assert np.all(np.diff(gas.pooled_CV_RMS_arcsec) < 0.0)
    opposite = stage1[stage1.closure.eq("star_minus_gas_flux") & stage1["lambda"].ge(2.0)]
    assert (~opposite.all_CV_roots).all()
    stage2 = pd.read_csv(RESULTS / "stage2_scores.csv")
    assert set(stage2.closure) == {
        "lambda0_baseline",
        "gas_minus_star_flux",
        "perpendicular_cw_flux",
        "isotropic_control",
    }


def test_selected_field_is_conservative_and_root_safe():
    result = report()
    audit = result["field_audits"]["gas_minus_star_flux"]
    assert audit["normalized_curl_RMS"] <= 1e-10
    assert audit["source_integral_fraction"] <= 1e-4
    assert result["full_refit"]["training_roots"] == 15
    assert result["full_refit"]["spent_heldout_roots"] == 7


def test_machine_readable_atlas_is_complete():
    stage1 = pd.read_csv(RESULTS / "stage1_scores.csv")
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    assert len(stage1) == 25
    assert len(stage1[stage1.closure.ne("lambda0_baseline")]) == 24
    assert report()["coverage"]["closures"] == 8
    assert report()["coverage"]["stage1_fold_refits"] == 125
    assert len(folds) >= 125
    assert (RESULTS / "closure_atlas.png").stat().st_size > 20000
