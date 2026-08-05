from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19cc_cross_scale_prediction_gates.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19cc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19cc_freezes_every_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == "passed_cross_scale_prediction_gate_freeze"
    assert all(report["gate_results"].values())


def test_v19cc_keeps_broad_galaxy_and_cluster_tests_first() -> None:
    report = MODULE.build_report()
    assert report["priority_order"][0] == "broad_galaxy_and_raw_cluster_core"
    assert report["priority_order"][1] == "same_metric_weak_lensing_and_merger_geometry"
    assert report["priority_order"][-1] == "local_relativistic_and_mathematical_veto"
    assert report["core_breadth"]["galaxy_holdout_minimum"] >= 48
    assert report["core_breadth"]["galaxy_morphology_strata"] == 8
    assert report["core_breadth"]["cluster_holdout_minimum"] >= 6
    assert report["core_breadth"]["cluster_relaxed_minimum"] >= 2
    assert report["core_breadth"]["cluster_disturbed_minimum"] >= 2


def test_v19cc_quantifies_near_term_same_metric_tests() -> None:
    report = MODULE.build_report()
    assert set(report["near_term_gate_ids"]) == {
        "N1_RESOLVED_CLUSTER_WEAK_LENSING",
        "N2_GALAXY_GALAXY_WEAK_LENSING",
        "N3_JOINT_DYNAMICS_AND_LENSING",
        "N4_COLLIDING_CLUSTER_DIRECTION_AND_OFFSETS",
    }


def test_v19cc_does_not_overclaim_broader_dark_matter_phenomena() -> None:
    report = MODULE.build_report()
    assert len(report["broader_prediction_states"]) == 5
    assert set(report["broader_prediction_states"].values()) == {"not_earned"}
    assert not any(report["theory_state"].values())


def test_v19cc_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19cc_cross_scale_prediction_gates" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
