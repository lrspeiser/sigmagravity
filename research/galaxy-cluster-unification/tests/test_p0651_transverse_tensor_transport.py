from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0651_transverse_tensor_transport"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_map_stage_passes_before_spent_lens_runs():
    result = report()
    assert result["spent_lens_stage_run"] is True
    assert len(result["map_gate_results"]) == 11
    assert all(result["map_gate_results"].values())
    assert result["map_metrics"]["registered_cluster_to_galaxy_ratio"] > 8.6
    assert min(
        result["map_metrics"]["mass_sensitivity_cluster_to_galaxy_ratios"].values()
    ) > 7.3


def test_spent_lens_predictive_gates_reject_candidate():
    result = report()
    assert result["status"] == "fail_spent_lens_stage"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_robustness"] is False
    assert sum(result["lens_gate_results"].values()) == 8
    assert result["lens_gate_results"]["CV_improvement"] is False
    assert result["lens_gate_results"]["beats_matched_multipole"] is False
    assert all(
        value
        for name, value in result["lens_gate_results"].items()
        if name not in {"CV_improvement", "beats_matched_multipole"}
    )


def test_unit_magnitude_is_natural_but_accuracy_is_worse():
    result = report()
    assert result["field_audit"]["mismatch_mode"] == "transverse_tensor_mix"
    assert result["field_audit"]["unit_deflection_RMS_arcsec"] > 0.80
    comparison = result["comparison"]
    assert comparison["CV_improvement_fraction_vs_lambda0"] < -0.15
    assert comparison["CV_improvement_fraction_vs_best_matched_multipole"] < -0.22
    assert comparison["P0601_spent_heldout_used_for_selection"] is False


def test_fold_failure_is_spatial_not_root_topology():
    folds = pd.read_csv(RESULTS / "fold_scores.csv").set_index("fold")
    assert int(folds.validation_roots.sum()) == 15
    assert folds.loc[0, "validation_RMS_arcsec"] > 5.5
    assert folds.loc[[2, 3], "validation_RMS_arcsec"].max() < 1.3
    assert report()["full_refit"]["training_roots"] == 15
    assert report()["full_refit"]["spent_heldout_roots"] == 7


def test_formula_has_no_fitted_or_per_object_field_strength():
    coverage = report()["coverage"]
    assert coverage["candidate_fields"] == 1
    assert coverage["amplitude_rows"] == 1
    assert coverage["fitted_field_amplitude_parameters"] == 0
    assert coverage["per_object_spatial_gravity_parameters"] == 0


def test_blindness_hashes_and_figure_are_preserved():
    result = report()
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0651_transverse_tensor_transport.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0651_transverse_tensor_transport.py"
    )
    assert (RESULTS / "transverse_tensor_transport.png").stat().st_size > 20000
