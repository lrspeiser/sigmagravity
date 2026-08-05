from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_sigma_v19cf_cross_scale_data_readiness.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19cf", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19cf_passes_every_role_and_integrity_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == (
        "source_universes_ready_but_prospective_core_not_execution_ready"
    )
    assert all(report["gate_results"].values())


def test_v19cf_records_real_wallaby_breadth_without_opening_targets() -> None:
    galaxy = MODULE.build_report()["galaxy_inventory"]
    assert galaxy["canonical_source_rows"] == 592
    assert galaxy["unique_canonical_ids"] == 592
    assert galaxy["kinematic_available_any_policy"] == 109
    assert galaxy["release_rows_with_counterpart_mixtures"] == 711
    assert galaxy["candidate_rows_in_mixture"] == 18550
    assert galaxy["source_universe_ready"]
    assert not galaxy["prospective_sample_admitted"]
    assert not galaxy["execution_ready"]


def test_v19cf_records_cluster_gap_without_relabeling_spent_data() -> None:
    cluster = MODULE.build_report()["cluster_inventory"]
    assert cluster["metadata_shortlist"] == 8
    assert cluster["relaxed_side"] == 4
    assert cluster["disturbed_side"] == 4
    assert cluster["admitted_prospective_holdouts"] == 0
    assert cluster["spent_registry_clusters"] == 4
    assert cluster["spent_raw_forward_score_ready"] == 0
    assert cluster["source_universe_ready"]
    assert not cluster["prospective_sample_admitted"]
    assert not cluster["execution_ready"]


def test_v19cf_access_boundary_remains_target_blind() -> None:
    access = MODULE.build_report()["access_boundary_audit"]
    assert access["read_source_and_metadata_reports"]
    assert access["read_wallaby_source_only_variety_rows"]
    assert not access["read_new_kinematic_values"]
    assert not access["read_new_lensing_coordinates_or_maps"]
    assert not access["selected_final_sample"]
    assert not access["changed_action_formula_or_constants"]
    assert not access["performed_detailed_solar_optimization"]


def test_v19cf_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19cf_cross_scale_data_readiness" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
