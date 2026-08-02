from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0655_conservative_streamline_deposit"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deposition_is_exactly_conservative_and_nontrivial():
    result = report()
    audit = result["field_audit"]
    assert audit["transport_flux_sum_relative_error"] == 0.0
    assert audit["transport_is_source_conservative"] is True
    assert audit["source_integral_fraction"] < 1e-10
    assert audit["maximum_flux_edge_fraction_of_RMS"] == 0.0
    assert audit["transport_relative_change_RMS"] > 0.8
    for name in (
        "transport_flux_conserved",
        "transport_declared_conservative",
        "field_source_integral",
        "edge_flux_closed",
        "transport_nontrivial",
    ):
        assert result["gate_results"][name] is True


def test_one_cross_validation_image_and_fit_root_are_lost():
    result = report()
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    predictions = pd.read_csv(RESULTS / "cv_predictions.csv")
    assert result["CV_summary"]["CV_roots"] == 14
    assert result["CV_summary"]["CV_images"] == 15
    failed = predictions[~predictions.root_converged]
    assert failed.image_id.tolist() == ["3b"]
    assert failed.fold.tolist() == [3]
    fold_three = folds[folds.fold == 3].iloc[0]
    assert fold_three.fit_roots == 12
    assert fold_three.fit_images == 13
    assert fold_three.validation_roots == 1
    assert fold_three.validation_images == 2


def test_predictive_and_heldout_gates_reject_candidate():
    result = report()
    assert result["status"] == "fail"
    assert result["candidate_advanced_to_robustness"] is False
    failed = {name for name, value in result["gate_results"].items() if not value}
    assert failed == {
        "CV_roots",
        "CV_improvement",
        "beats_matched_multipole",
        "spent_heldout_not_worse",
    }
    assert sum(result["gate_results"].values()) == 13
    assert result["full_refit"]["spent_heldout_worsening_fraction_vs_P0599"] > 0.70


def test_formula_adds_no_fit_or_physical_scale():
    coverage = report()["coverage"]
    assert coverage["candidate_fields"] == 1
    assert coverage["amplitude_rows"] == 1
    assert coverage["fitted_field_amplitude_parameters"] == 0
    assert coverage["new_physical_length_constants"] == 0
    assert coverage["per_object_spatial_gravity_parameters"] == 0


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0655_conservative_streamline_deposit.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0655_conservative_streamline_deposit.py"
    )
    assert (RESULTS / "conservative_streamline_deposit.png").stat().st_size > 20000
