from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0681_compact_halo_hessian_richardson"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_derivative_audit_fails_declared_gates():
    result = report()
    assert result["status"] == "fail"
    assert result["all_integrity_gates_pass"] is False
    assert result["P0678_decomposition_numerically_qualified"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {
        "kappa_error_decreases",
        "determinant_error_decreases",
        "smallest_kappa_agreement",
        "smallest_determinant_agreement",
    }


def test_curl_converges_but_exact_differences_plateau():
    table = pd.read_csv(RESULTS / "exact_hessian_step_convergence.csv")
    assert table.normalized_curl_RMS.is_monotonic_decreasing
    assert table.iloc[-1].normalized_curl_RMS < 2e-9
    assert table.kappa_exact_relative_RMS.between(1.6e-5, 1.8e-5).all()
    assert table.determinant_exact_relative_RMS.between(2.0e-5, 2.2e-5).all()
    assert table.negative_jacobian_points.eq(6).all()


def test_protocol_hash_and_no_score_seals_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0681_compact_halo_hessian_richardson.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0681_compact_halo_hessian_richardson.py"
    )
    assert result["new_candidate_formula_fit"] is False
    assert result["new_raw_image_root_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
