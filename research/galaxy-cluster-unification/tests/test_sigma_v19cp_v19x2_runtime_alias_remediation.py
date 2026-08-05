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
    assert payload["gate_results"]["only_required_completed_cells_alias_added_and_equals_5082"]
    assert payload["gate_results"]["no_lensing_halo_action_gravity_holdout_or_solar_access"]


def test_v19cp_records_the_next_precombination_path_failure() -> None:
    payload = report()
    assert payload["decision"] == "v19x2_or_source_chain_execution_incomplete"
    assert not payload["gate_results"]["v19x2_byte_identical_runner_reaches_registered_scientific_disposition"]
    assert payload["v19x2_report"]["status"] == "unified_spectral_combination_commissioning_execution_failed"
    assert payload["v19x2_report"]["execution_exception"] == (
        "RuntimeError: V19X2 cell path is outside frozen archive roots: ('BULLET', 45, 554, 3)"
    )
    assert not payload["v19x2_report"]["gates"]["execution_completed"]


def test_v19cp_preserves_physics_boundary() -> None:
    boundary = report()["authorization_boundary"]
    assert not boundary["v19bs_run"] and not boundary["action_derived"]
    assert not boundary["target_or_gravity_opened"] and not boundary["solar_optimized"]
