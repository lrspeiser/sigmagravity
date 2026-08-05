from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts" / "audit_sigma_v19cg_whole_repo_cluster_holdout_contamination.py"
)
SPEC = importlib.util.spec_from_file_location("sigma_v19cg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19cg_retires_the_original_shortlist_fail_closed() -> None:
    report = MODULE.build_report()
    assert report["decision"] == (
        "original_cluster_shortlist_retired_from_prospective_holdout_role"
    )
    assert all(report["gate_results"].values())


def test_v19cg_records_prior_use_and_coordinate_exposure_separately() -> None:
    summary = MODULE.build_report()["summary"]
    assert summary["original_shortlist_systems"] == 8
    assert summary["prior_sigma_used_systems"] == 6
    assert summary["raw_coordinate_exposed_shortlist_systems"] == 3
    assert summary["disqualified_unique_systems"] == 7
    assert summary["remaining_source_incomplete_reserves"] == 1
    assert summary["admitted_prospective_holdouts"] == 0


def test_v19cg_does_not_store_or_use_exposed_coordinates() -> None:
    report = MODULE.build_report()
    incident = report["coordinate_exposure_incident"]
    assert not incident["coordinate_values_copied_into_repository"]
    assert not incident["coordinate_values_used_for_a_score_or_selection"]
    access = report["access_boundary_audit"]
    assert access["recorded_incident_without_coordinate_values"]
    assert not access["opened_another_raw_coordinate_or_map_after_incident"]
    assert not access["selected_replacement_cluster"]


def test_v19cg_preserves_galaxy_readiness_but_supersedes_cluster_readiness() -> None:
    supersession = MODULE.build_report()["supersession"]
    assert supersession["v19cf_galaxy_readiness_conclusion_unchanged"]
    assert not supersession[
        "v19cf_cluster_source_universe_ready_for_prospective_core"
    ]
    assert not supersession[
        "v19bh_shortlist_may_supply_final_whole_object_holdouts"
    ]
    assert not supersession[
        "v19bt_six_source_imaging_preflights_may_supply_final_whole_object_holdouts"
    ]


def test_v19cg_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = (
        ROOT
        / "results"
        / "sigma_v19cg_whole_repo_cluster_holdout_contamination"
        / "report.json"
    )
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
