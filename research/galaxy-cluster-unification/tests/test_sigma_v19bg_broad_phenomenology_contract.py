from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19bg_broad_phenomenology_contract.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bg", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19bg_contract_passes_every_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == "passed_broad_phenomenology_contract"
    assert all(report["gate_results"].values())


def test_v19bg_records_broad_core_coverage_without_overclaiming() -> None:
    report = MODULE.build_report()
    coverage = report["current_coverage"]
    assert coverage["SPARC_full_systems"] == 131
    assert coverage["SPARC_full_points"] == 3034
    assert coverage["BCG_systems"] == 34
    assert coverage["CLASH_model_derived_systems"] == 20
    assert coverage["spent_raw_lensing_systems"] == 5
    assert coverage["RELICS_resolved_baryon_maps"] == 4
    assert coverage["RELICS_raw_score_ready_systems"] == 0
    assert coverage["merging_cluster_directional_source_systems"] == 2


def test_v19bg_does_not_select_theory_or_open_holdout() -> None:
    report = MODULE.build_report()
    assert report["mechanism_decision"]["linear_isotropic_identity_long_wave_control"] == "closed"
    assert not report["mechanism_decision"]["action_selected"]
    assert not report["mechanism_decision"]["constant_selected"]
    assert not report["long_wave_scale_state"]["measured_interval"]
    assert not report["long_wave_scale_state"]["selected_constant"]
    authorization = report["authorization_audit"]
    assert not authorization["open_new_holdout"]
    assert not authorization["change_gravity_formula"]
    assert not authorization["perform_detailed_solar_optimization"]


def test_v19bg_requires_diverse_blind_cluster_sample_and_other_phenomena() -> None:
    report = MODULE.build_report()
    blind = report["future_blind_core_gate"]
    assert blind["minimum_new_clusters"] >= 6
    assert blind["minimum_relaxed_clusters"] >= 2
    assert blind["minimum_disturbed_or_merging_clusters"] >= 2
    assert blind["per_image_position_uncertainties_required"]
    assert len(report["registered_galaxy_strata"]) == 8
    assert len(report["registered_cluster_strata"]) == 6
    assert len(report["other_phenomena"]) == 7


def test_v19bg_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19bg_broad_phenomenology_contract" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
