from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report() -> dict:
    return json.loads((ROOT / "results" / "sigma_v19cp_v19x2_runtime_alias_remediation" / "report.json").read_text(encoding="utf-8"))


def test_v19cp_repairs_only_the_runtime_alias() -> None:
    payload = report()
    assert payload["status"] == "v19x2_runtime_alias_remediation_completed"
    assert payload["changed_paths"] == [{"path": "runtime_authorization.required_completed_cells", "before": None, "after": 5082}]
    assert payload["scientific_sections_unchanged"]
    assert all(payload["gate_results"].values())


def test_v19cp_reaches_a_registered_v19x2_disposition() -> None:
    assert report()["v19x2_report"]["status"] in {
        "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized",
        "unified_spectral_combination_commissioning_gate_failed",
    }


def test_v19cp_preserves_physics_boundary() -> None:
    boundary = report()["authorization_boundary"]
    assert not boundary["v19bs_run"] and not boundary["action_derived"]
    assert not boundary["target_or_gravity_opened"] and not boundary["solar_optimized"]
