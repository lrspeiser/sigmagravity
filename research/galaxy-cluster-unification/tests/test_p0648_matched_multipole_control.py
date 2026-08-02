from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0648_matched_multipole_control"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_multipoles_do_not_explain_the_full_candidate_gain():
    comparison = report()["comparison"]
    assert comparison["best_multipole_order"] == 3
    assert comparison["best_multipole_amplitude"] == -12.5
    assert comparison["candidate_fractional_improvement_vs_best_multipole"] > 0.11
    assert comparison["candidate_specificity_survives"] is True
    assert comparison["generic_angular_control_explains_gain"] is False
    assert comparison["P0647_candidate_was_already_rejected"] is True


def test_neither_attractive_amplitude_is_promoted_as_identified():
    comparison = report()["comparison"]
    assert comparison["multipole_amplitude_identified"] is False
    assert comparison["P0601_spent_heldout_used_for_selection"] is False
    stage2 = pd.read_csv(RESULTS / "stage2_scores.csv").set_index("order")
    assert np.isclose(stage2.loc[3, "pooled_CV_RMS_arcsec"], 2.599360329775749)
    assert np.isclose(stage2.loc[4, "pooled_CV_RMS_arcsec"], 2.695273811460096)
    assert stage2.all_CV_roots.all()


def test_screen_to_exact_change_is_preserved():
    stage1 = pd.read_csv(RESULTS / "stage1_scores.csv")
    screened = stage1[stage1.order.eq(3) & stage1["lambda"].eq(-12.5)].iloc[0]
    exact = pd.read_csv(RESULTS / "stage2_scores.csv").query("order == 3").iloc[0]
    assert screened.pooled_CV_RMS_arcsec < 2.1
    assert exact.pooled_CV_RMS_arcsec > 2.5
    assert int(screened.CV_roots) == 15
    assert int(exact.CV_roots) == 15


def test_field_quality_failure_is_not_hidden():
    result = report()
    gates = result["field_gate_results"]
    assert gates["m3_curl"] is True
    assert gates["m3_source"] is True
    assert gates["m3_normalization"] is True
    assert gates["m4_curl"] is True
    assert gates["m4_source"] is False
    assert gates["m4_normalization"] is True
    assert result["field_audits"]["m4"]["source_integral_fraction"] > 1e-4


def test_full_refit_is_descriptive_and_root_complete():
    full = report()["full_refit"]
    assert full["training_roots"] == 15
    assert full["spent_heldout_roots"] == 7
    assert np.isclose(full["spent_heldout_RMS_arcsec"], 1.0441394877004757)


def test_blindness_coverage_and_hashes_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["coverage"]["multipole_amplitude_parameters"] == 1
    assert result["coverage"]["per_object_spatial_gravity_parameters"] == 0
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0648_matched_multipole_control.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0648_matched_multipole_control.py"
    )
    assert (RESULTS / "matched_multipole_control.png").stat().st_size > 20000
