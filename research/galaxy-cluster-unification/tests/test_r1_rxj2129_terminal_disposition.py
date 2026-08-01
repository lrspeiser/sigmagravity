from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/r1_rxj2129_terminal_observable_disposition/report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_disposition_is_internally_consistent_and_binding() -> None:
    if not REPORT.is_file():
        pytest.skip("terminal H2/X4 component pair is still running")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["report_version"] == (
        "R1B3-RXJ2129-terminal-observable-disposition-0.2-integrity-bound"
    )
    for item in report["inputs"].values():
        path = ROOT / item["path"]
        assert path.is_file()
        assert _sha256(path) == item["sha256"]

    h2 = report["component_outcomes"]["H2"]
    x4 = report["component_outcomes"]["X4"]
    assert h2["status"] in {"pass", "fail"}
    assert x4["status"] in {"pass", "fail"}
    expected_branch = {
        ("pass", "pass"): "both_observable_production_gates_pass_global_ceiling_binding",
        ("fail", "fail"): "both_observable_production_gates_fail_global_ceiling_binding",
        ("fail", "pass"): "H2_fails_X4_passes_global_ceiling_binding",
        ("pass", "fail"): "H2_passes_X4_fails_global_ceiling_binding",
    }[(h2["status"], x4["status"])]
    assert report["branch"] == expected_branch
    assert all(report["status_consistency_checks"].values())

    integrity = report["artifact_integrity"]
    assert integrity["H2"]["artifact_count"] == 4
    assert integrity["H2"]["all_reported_artifacts_rehashed"]
    assert integrity["H2"]["immutable_input_artifact_count"] == 11
    assert integrity["H2"]["all_immutable_inputs_rehashed"]
    assert all(
        artifact["integrity_passed"]
        for artifact in integrity["H2"]["artifacts"].values()
    )
    assert integrity["X4"]["manifest_product_count"] == 116
    assert integrity["X4"]["response_product_count"] == 108
    assert integrity["X4"]["detector_map_count"] == 8
    assert integrity["X4"]["input_artifact_count"] == 4
    assert integrity["X4"]["all_implementation_inputs_rehashed"]
    assert integrity["X4"]["all_manifest_products_rehashed"]

    disposition = report["global_disposition"]
    assert disposition["ten_system_hard_shortfall_changes"] is False
    assert disposition["minimum_strict_system_deficit_even_if_RXJ2129_passed"] == 9
    assert disposition["population_R2_identifiable"] is False
    assert disposition["unification_claim"] == (
        "withheld_due_public_data_identifiability_ceiling"
    )
    assert not any(
        report["authorization"][key]
        for key in (
            "assemble_H3_covariance",
            "construct_X5_joint_likelihood",
            "select_another_system",
            "reconstruct_dynamical_or_Weyl_response",
            "cross_validate_latent_response",
            "fit_new_force_or_action",
        )
    )
