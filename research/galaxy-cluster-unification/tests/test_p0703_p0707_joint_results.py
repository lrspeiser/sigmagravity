from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(stage: str) -> dict:
    return json.loads((ROOT / "results" / stage / "report.json").read_text(encoding="utf-8"))


def test_planarity_branch_failures_are_retained_without_unsealing() -> None:
    expected = {
        "p0703_spent_planarity_blended_joint_screen": {"no_missing_multiplicity", "nuisance_bounds"},
        "p0704_spent_planarity_endpoint_joint_screen": {"no_missing_multiplicity"},
        "p0705_spent_planarity_intensity_joint_screen": {"training_roots", "training_RMS", "no_missing_multiplicity"},
    }
    for stage, failed in expected.items():
        report = load(stage)
        assert report["status"] == "fail"
        assert set(report["failed_gates"]) == failed
        assert report["sealed_P0633_kinematics_opened"] is False
        assert report["sealed_P0640_lensing_constraints_opened"] is False


def test_two_potential_metric_passes_math_solar_and_spent_joint_gates() -> None:
    audit = load("p0706_two_potential_rar_metric_audit")
    joint = load("p0707_spent_two_potential_rar_metric_joint_screen")
    assert audit["all_math_and_solar_gates_pass"] is True
    assert all(audit["gate_results"].values())
    assert joint["status"] == "pass"
    assert joint["all_progression_gates_pass"] is True
    assert all(joint["gate_results"].values())
    assert joint["failed_gates"] == []
    assert joint["candidate_advanced_to_external_lock_robustness"] is True
    assert joint["candidate_advanced_to_sealed_outcomes"] is False
    assert joint["sealed_P0633_kinematics_opened"] is False
    assert joint["sealed_P0640_lensing_constraints_opened"] is False


def test_p0707_is_competitive_without_per_object_gravity_settings() -> None:
    joint = load("p0707_spent_two_potential_rar_metric_joint_screen")
    galaxy = joint["spent_DDO154"]
    cluster = joint["spent_RXJ2129"]
    assert galaxy["comparisons"]["candidate_RMSE_to_algebraic_MOND_ratio"] < 1.0
    assert galaxy["comparisons"]["candidate_weighted_RMSE_to_algebraic_MOND_ratio"] < 1.2
    assert cluster["candidate_to_compact_halo_heldout_RMS_ratio"] < 1.25
    assert cluster["topology"]["missing_multiplicity_families"] == 0
    assert cluster["topology"]["parity_diverse_families"] == 7
    assert cluster["topology"]["critical_curve_present_families"] == 7
    assert joint["gate_results"]["accounting_no_per_object_gravity"] is True
