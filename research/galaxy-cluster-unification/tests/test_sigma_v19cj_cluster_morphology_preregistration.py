from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preregister_sigma_v19cj_cluster_morphology.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19cj", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19cj_preregistration_passes_before_value_access() -> None:
    report = MODULE.build_report()
    assert report["decision"] == (
        "registered_morphology_tables_may_be_acquired_under_frozen_rule"
    )
    assert all(report["gate_results"].values())


def test_v19cj_freezes_uncertainty_aware_three_state_rule() -> None:
    rule = MODULE.build_report()["primary_state_rule"]
    assert rule["secure_relaxed"] == "delta+e_delta < 0"
    assert rule["secure_disturbed"] == "delta-e_delta > 0"
    assert rule["boundary_intermediate"] == "delta-e_delta <= 0 <= delta+e_delta"
    assert rule["no_threshold_may_be_moved_after_values_are_read"]


def test_v19cj_forbids_lensing_or_sigma_selection() -> None:
    report = MODULE.build_report()
    access = report["access_boundary_audit"]
    assert not access["downloaded_registered_table_before_freeze"]
    assert not access["read_candidate_morphology_value_before_freeze"]
    assert not access["selected_cluster"]
    assert not access["opened_lensing_halo_or_gravity_target"]
    assert not access["changed_action_formula_or_constants"]
    assert not access["performed_detailed_solar_optimization"]


def test_v19cj_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = (
        ROOT
        / "results"
        / "sigma_v19cj_cluster_morphology_preregistration"
        / "report.json"
    )
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
