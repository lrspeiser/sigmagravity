from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19bu_wallaby_source_only_metadata.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19bu_passes_every_source_only_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == (
        "passed_source_only_candidate_universe_not_holdout_selection"
    )
    assert all(report["gate_results"].values())


def test_v19bu_recovers_the_published_candidate_universe_without_deduplicating() -> None:
    audit = MODULE.build_report()["source_payload_audit"]
    assert audit["row_count"] == 711
    assert audit["unique_source_names"] == 592
    assert audit["duplicate_name_rows"] == 119
    assert audit["maximum_rows_per_name"] == 2
    assert audit["kflag_unique_name_counts"]["2"] == 109


def test_v19bu_contains_only_the_frozen_source_column_whitelist() -> None:
    report = MODULE.build_report()
    boundary = report["source_target_boundary"]
    assert set(boundary["allowed_source_columns"]).isdisjoint(
        boundary["sealed_target_columns"]
    )
    assert report["source_payload_audit"]["columns"] == boundary[
        "allowed_source_columns"
    ]


def test_v19bu_keeps_every_kinematic_and_gravity_target_sealed() -> None:
    report = MODULE.build_report()
    boundary = report["access_boundary_audit"]
    assert not boundary["kinematic_model_table_rows_read"]
    assert not boundary["rotation_speed_values_read"]
    assert not boundary["velocity_field_or_cube_opened"]
    assert not boundary["final_holdout_sample_selected"]
    assert not boundary["gravity_formula_or_constant_changed"]
    assert not boundary["solar_system_optimization_performed"]


def test_v19bu_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19bu_wallaby_source_only_metadata" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
