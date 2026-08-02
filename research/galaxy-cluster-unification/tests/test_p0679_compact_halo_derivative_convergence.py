from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0679_compact_halo_derivative_convergence"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_derivative_audit_fails_only_threshold_count():
    result = report()
    assert result["status"] == "fail"
    assert result["all_integrity_gates_pass"] is False
    assert result["P0678_decomposition_numerically_qualified"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {"direct_threshold_count"}


def test_grid_and_direct_curl_converge_strongly():
    result = report()
    metrics = result["metrics"]
    grid = pd.read_csv(RESULTS / "nested_grid_curl.csv")
    steps = pd.read_csv(RESULTS / "direct_step_derivatives.csv")
    assert grid.normalized_curl_RMS.is_monotonic_decreasing
    assert metrics["grid_257_curl_improvement_factor_vs_33"] > 49.0
    assert steps.iloc[-1].normalized_curl_RMS < 2e-7
    assert metrics["direct_steps_below_original_P0678_threshold"] == 3
    assert metrics["smallest_two_step_kappa_relative_RMS_difference"] < 5e-7
    assert (
        metrics["smallest_two_step_jacobian_determinant_relative_RMS_difference"]
        < 8e-7
    )


def test_protocol_and_no_score_seals_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0679_compact_halo_derivative_convergence.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0679_compact_halo_derivative_convergence.py"
    )
    assert result["new_candidate_formula_fit"] is False
    assert result["new_raw_image_root_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
