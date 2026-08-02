from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0677_spent_rxj2129_dual_transverse_survival_field"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dual_survival_fails_strength_gates_and_retires_branch():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_spent_raw_lens_topology_audit"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {"response_ratio_lower", "dual_change_nonperturbative"}


def test_dual_survival_converges_but_remains_far_below_target():
    metrics = report()["metrics"]
    assert metrics["dual_normalized_residual_RMS"] < 1e-5
    assert metrics["minimum_dual_constitutive_eigenvalue"] > 0.0
    assert 1.30 < metrics["dual_to_scalar_strong_lens_deflection_RMS_ratio"] < 1.31
    assert 0.31 < metrics["dual_minus_scalar_strong_lens_relative_RMS"] < 0.32
    assert metrics["dual_normalized_deflection_curl_RMS"] < 1e-8


def test_sources_and_no_raw_or_sealed_score_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0677_spent_rxj2129_dual_transverse_survival_field.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0677_spent_rxj2129_dual_transverse_survival_field.py"
    )
    field_path = RESULTS / "rxj2129_absolute_scalar_dual_transverse_fields.npz"
    assert result["field_sha256"] == digest(field_path)
    assert result["raw_lens_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
