from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0654_zero_padded_nonlocal_transport"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_padding_repairs_source_and_edge_conservation():
    result = report()
    audit = result["field_audit"]
    assert audit["source_integral_fraction"] < 1e-10
    assert audit["maximum_flux_edge_fraction_of_RMS"] == 0.0
    assert audit["normalized_curl_RMS"] < 1e-10
    assert result["gate_results"]["field_source_integral"] is True
    assert result["gate_results"]["edge_flux_closed"] is True


def test_padding_is_exactly_inherited_numerical_trace_cap():
    result = report()
    domain = result["numerical_domain"]
    assert domain["actual_padding_arcsec"] == 48.0
    assert domain["expected_padding_arcsec"] == 48.0
    assert domain["trace_length_cap_cells"] == 48
    assert domain["physical_support_radius_arcsec"] == 58.0
    assert domain["zero_mass_added"] is True
    assert result["coverage"]["field_grid_cells_per_axis"] == 217
    assert result["field_audit"]["computational_padding_changes_physical_support"] is False


def test_one_missing_cv_root_rejects_candidate():
    result = report()
    folds = pd.read_csv(RESULTS / "fold_scores.csv")
    predictions = pd.read_csv(RESULTS / "cv_predictions.csv")
    assert result["status"] == "fail"
    assert result["candidate_advanced_to_robustness"] is False
    assert result["CV_summary"]["CV_roots"] == 14
    assert result["CV_summary"]["CV_images"] == 15
    assert result["CV_summary"]["pooled_CV_RMS_arcsec"] == float("inf")
    assert int(folds.validation_roots.sum()) == 14
    failed = predictions[~predictions.root_converged]
    assert failed.image_id.tolist() == ["6b"]
    assert failed.fold.tolist() == [1]


def test_only_cross_validation_score_gates_fail():
    result = report()
    failed = {name for name, value in result["gate_results"].items() if not value}
    assert failed == {"CV_roots", "CV_improvement", "beats_matched_multipole"}
    assert sum(result["gate_results"].values()) == 12
    assert result["gate_results"]["spent_heldout_not_worse"] is True
    assert result["full_refit"]["spent_heldout_worsening_fraction_vs_P0599"] < 0.10


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
        ROOT / "configs/p0654_zero_padded_nonlocal_transport.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0654_zero_padded_nonlocal_transport.py"
    )
    assert (RESULTS / "zero_padded_nonlocal_transport.png").stat().st_size > 20000
