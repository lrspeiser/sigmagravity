from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19ck_single_rmf_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ck", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19ck_report_preserves_fail_closed_authorization() -> None:
    report = json.loads((ROOT / "results" / "sigma_v19ck_single_rmf_diagnostic" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_single_rmf_diagnostic"
    assert report["recovery_unchanged"]
    assert all(report["preflight_gate_results"].values())
    boundary = report["authorization_boundary"]
    assert not boundary["final_retry_authorized"]
    assert not boundary["cell_drop_or_response_change_authorized"]
    assert not boundary["v19br_resume_authorized"]
    assert not boundary["target_action_or_gravity_accessed"]


def test_v19ck_captures_verbose_and_direct_disposition() -> None:
    report = json.loads((ROOT / "results" / "sigma_v19ck_single_rmf_diagnostic" / "report.json").read_text(encoding="utf-8"))
    assert report["specextract"]["log_bytes"] > 0
    assert report["decision"] in {
        "verbose_diagnostic_did_not_reproduce_final_retry_still_not_authorized",
        "specextract_orchestration_failure_isolated_remediation_preregistration_required",
        "direct_mkacisrmf_failure_captured_remediation_preregistration_required",
        "rmf_failure_reproduced_without_direct_diagnostic_remediation_not_authorized",
    }
    if report["direct_mkacisrmf"] is not None:
        assert report["direct_mkacisrmf"]["log_bytes"] > 0
