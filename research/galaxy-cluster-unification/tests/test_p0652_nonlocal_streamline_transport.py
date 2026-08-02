from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0652_nonlocal_streamline_transport"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nonlocal_primary_has_large_root_complete_cv_gain():
    result = report()
    comparison = result["comparison"]
    assert comparison["CV_improvement_fraction_vs_lambda0"] > 0.24
    assert comparison["CV_improvement_fraction_vs_best_matched_multipole"] > 0.20
    assert result["closure_scores"][0]["CV_roots"] == 15
    assert comparison["P0601_spent_heldout_used_for_selection"] is False


def test_two_frozen_safety_gates_prevent_advancement():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_robustness"] is False
    assert sum(result["gate_results"].values()) == 10
    assert result["gate_results"]["spent_heldout_not_worse"] is False
    assert result["gate_results"]["field_source_integral"] is False
    assert all(
        value
        for name, value in result["gate_results"].items()
        if name not in {"spent_heldout_not_worse", "field_source_integral"}
    )


def test_path_transport_is_nontrivial_and_source_imbalanced():
    audit = report()["field_audits"]["streamline_averaged_gas_minus_star_flux"]
    assert audit["streamline_steps"] == 12
    assert audit["transport_relative_change_RMS"] > 0.75
    assert 0.68 < audit["transport_flux_RMS_ratio"] < 0.70
    assert audit["source_integral_fraction"] > 0.049
    assert audit["normalized_curl_RMS"] < 1e-10


def test_diagnostic_residual_cannot_replace_primary():
    scores = pd.read_csv(RESULTS / "closure_scores.csv").set_index("closure")
    assert np.isclose(
        scores.loc["streamline_averaged_gas_minus_star_flux", "pooled_CV_RMS_arcsec"],
        2.0751483644294844,
    )
    assert scores.loc[
        "streamline_balanced_residual_flux", "pooled_CV_RMS_arcsec"
    ] > 2.78
    assert report()["diagnostic_closure_cannot_advance"] is True


def test_formula_adds_no_amplitude_or_physical_length_fit():
    coverage = report()["coverage"]
    assert coverage["advancing_candidates"] == 1
    assert coverage["amplitude_rows_per_closure"] == 1
    assert coverage["fitted_field_amplitude_parameters"] == 0
    assert coverage["new_physical_length_constants"] == 0
    assert coverage["per_object_spatial_gravity_parameters"] == 0


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0652_nonlocal_streamline_transport.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0652_nonlocal_streamline_transport.py"
    )
    assert (RESULTS / "nonlocal_streamline_transport.png").stat().st_size > 20000
