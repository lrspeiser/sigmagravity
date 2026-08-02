from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0680_compact_halo_exact_hessian"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_hessian_audit_fails_only_direct_agreement():
    result = report()
    assert result["status"] == "fail"
    assert result["all_integrity_gates_pass"] is False
    assert result["P0678_decomposition_numerically_qualified"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {"kappa_agreement", "determinant_agreement"}


def test_exact_hessian_is_symmetric_and_topology_sign_agrees():
    metrics = report()["metrics"]
    assert metrics["exact_hessian_normalized_curl_RMS"] == 0.0
    assert metrics["direct_0p01_normalized_curl_RMS"] < 2e-7
    assert 1e-5 < metrics["exact_vs_direct_0p01_kappa_relative_RMS_difference"] < 2e-5
    assert (
        2e-5
        < metrics[
            "exact_vs_direct_0p01_jacobian_determinant_relative_RMS_difference"
        ]
        < 3e-5
    )
    assert metrics["exact_negative_jacobian_points"] == 6
    assert metrics["direct_negative_jacobian_points"] == 6


def test_protocol_hash_and_seals_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0680_compact_halo_exact_hessian.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0680_compact_halo_exact_hessian.py"
    )
    assert result["new_candidate_formula_fit"] is False
    assert result["new_raw_image_root_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
