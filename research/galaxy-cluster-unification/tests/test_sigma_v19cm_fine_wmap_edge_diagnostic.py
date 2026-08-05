from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def report() -> dict:
    return json.loads((ROOT / "results" / "sigma_v19cm_fine_wmap_edge_diagnostic" / "report.json").read_text(encoding="utf-8"))


def test_v19cm_fine_wmap_passes_all_registered_gates() -> None:
    payload = report()
    assert payload["status"] == "completed_fine_wmap_edge_diagnostic"
    assert payload["decision"] == "fine_wmap_physically_equivalent_recovery_candidate_passed_separate_final_protocol_required"
    assert all(payload["preflight"].values())
    assert all(payload["gate_results"].values())


def test_v19cm_retains_weighting_and_only_changes_wmap_resolution() -> None:
    command = report()["command"]
    assert "weight=yes" in command
    assert "weight_rmf=yes" in command
    assert "binwmap=det=1" in command
    assert "energy_wmap=500:7000" in command
    assert "binarfwmap=1" in command


def test_v19cm_does_not_admit_or_authorize_final_response() -> None:
    boundary = report()["authorization_boundary"]
    assert not boundary["final_retry_authorized"]
    assert not boundary["diagnostic_products_admitted"]
    assert not boundary["target_or_gravity_accessed"]
