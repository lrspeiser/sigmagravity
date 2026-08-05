from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_sigma_v19ch_jwst_slice_clean_source_frame.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19ch_establishes_a_large_clean_source_frame() -> None:
    report = MODULE.build_report()
    assert report["decision"] == (
        "clean_cluster_source_frame_established_metadata_stratification_required"
    )
    assert all(report["gate_results"].values())


def test_v19ch_counts_quarantine_and_exposure_without_double_counting() -> None:
    summary = MODULE.build_report()["summary"]
    assert summary["external_source_frame_targets"] == 182
    assert summary["repository_identity_hit_targets"] == 12
    assert summary["raw_exposed_current_source_targets"] == 12
    assert summary["hit_and_exposure_overlap"] == 4
    assert summary["quarantined_or_spent_unique_targets"] == 20
    assert summary["zero_hit_unexposed_candidates"] == 162
    assert summary["replacement_clusters_selected"] == 0
    assert summary["clusters_admitted"] == 0


def test_v19ch_fails_closed_on_the_entire_paper_sample() -> None:
    report = MODULE.build_report()
    incident = report["coordinate_exposure_incident"]
    assert len(incident["paper_sample_aliases"]) == 14
    assert len(incident["current_source_frame_ids_failed_closed"]) == 12
    assert incident["raw_coordinate_values_entered_ephemeral_filter_process"]
    assert incident[
        "raw_coordinate_values_returned_visibly_for_at_least_one_system"
    ]
    assert not incident["coordinate_values_copied_into_repository"]
    assert not incident["coordinate_values_used_for_score_selection_or_physics"]
    clean = set(report["clean_source_frame_candidate_ids"])
    assert not clean.intersection(incident["current_source_frame_ids_failed_closed"])


def test_v19ch_keeps_selection_and_physics_frozen() -> None:
    report = MODULE.build_report()
    access = report["access_boundary_audit"]
    assert access["used_only_external_program_identity_for_source_frame"]
    assert access["recorded_pdf_exposure_fail_closed"]
    assert not access["opened_raw_target_payload_after_freeze"]
    assert not access["selected_replacement_cluster"]
    assert not access["changed_action_formula_or_constants"]
    assert not access["performed_detailed_solar_optimization"]


def test_v19ch_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = (
        ROOT
        / "results"
        / "sigma_v19ch_jwst_slice_clean_source_frame"
        / "report.json"
    )
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
