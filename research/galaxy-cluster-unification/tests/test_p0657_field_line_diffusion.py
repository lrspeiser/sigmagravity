from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0657_field_line_diffusion"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diffusion_passes_every_mathematical_gate():
    result = report()
    audit = result["field_audit"]
    assert audit["transport_operator"] == "symmetric_field_line_graph_diffusion"
    assert audit["transport_flux_sum_relative_error"] < 1e-8
    assert audit["transport_component_overshoot_fraction"] == 0.0
    assert audit["source_integral_fraction"] < 1e-10
    assert audit["maximum_flux_edge_fraction_of_RMS"] == 0.0
    assert audit["transport_is_self_adjoint_diffusion"] is True
    for name in (
        "field_curl",
        "field_source_integral",
        "edge_flux_closed",
        "transport_nontrivial",
        "transport_flux_conserved",
        "transport_no_overshoot",
        "transport_declared_conservative",
        "transport_self_adjoint",
    ):
        assert result["gate_results"][name] is True


def test_fold_one_loses_two_validation_roots():
    result = report()
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    predictions = pd.read_csv(RESULTS / "cv_predictions.csv")
    assert result["CV_summary"]["CV_roots"] == 13
    assert result["CV_summary"]["CV_images"] == 15
    failed = predictions[~predictions.root_converged]
    assert failed.image_id.tolist() == ["1b", "6b"]
    assert failed.fold.tolist() == [1, 1]
    fold_one = folds[folds.fold == 1].iloc[0]
    assert fold_one.validation_roots == 2
    assert fold_one.validation_images == 4


def test_only_cross_validation_score_gates_fail():
    result = report()
    assert result["status"] == "fail"
    assert result["candidate_advanced_to_robustness"] is False
    failed = {name for name, value in result["gate_results"].items() if not value}
    assert failed == {"CV_roots", "CV_improvement", "beats_matched_multipole"}
    assert sum(result["gate_results"].values()) == 16


def test_spent_heldout_improves_with_all_roots():
    result = report()
    full = result["full_refit"]
    assert full["training_roots"] == 15
    assert full["spent_heldout_roots"] == 7
    assert full["spent_heldout_worsening_fraction_vs_P0599"] < -0.02
    assert result["gate_results"]["spent_heldout_not_worse"] is True


def test_formula_adds_no_fitted_gravity_parameter():
    coverage = report()["coverage"]
    assert coverage["fitted_field_amplitude_parameters"] == 0
    assert coverage["fitted_diffusion_parameters"] == 0
    assert coverage["new_physical_length_constants"] == 0
    assert coverage["per_object_spatial_gravity_parameters"] == 0


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0657_field_line_diffusion.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0657_field_line_diffusion.py"
    )
    assert (RESULTS / "field_line_diffusion.png").stat().st_size > 20000
