from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19bh_blind_cluster_admission.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bh", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19bh_passes_every_admission_protocol_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == "passed_metadata_only_holdout_admission_protocol"
    assert all(report["gate_results"].values())


def test_v19bh_shortlist_is_fresh_balanced_and_not_admitted() -> None:
    report = MODULE.build_report()
    state = report["admission_state"]
    assert state["metadata_shortlist_count"] >= 8
    assert state["relaxed_side_count"] >= 4
    assert state["disturbed_side_count"] >= 4
    assert state["admitted_holdouts"] == 0
    assert not state["raw_target_payload_opened"]
    assert not state["final_six_selected"]
    assert all(
        not audit["hits"]
        for audit in report["alias_audit"]["systems"].values()
    )


def test_v19bh_requires_cluster_state_mass_and_projection_breadth() -> None:
    final = MODULE.build_report()["final_sample_requirements"]
    assert final["clusters"] >= 6
    assert final["relaxed_side_minimum"] >= 2
    assert final["disturbed_side_minimum"] >= 2
    assert final["cool_core_relaxed_minimum"] >= 1
    assert final["non_cool_core_relaxed_minimum"] >= 1
    assert final["plane_of_sky_merger_minimum"] >= 1
    assert final["projection_challenging_or_line_of_sight_merger_minimum"] >= 1
    assert final["lower_mass_half_minimum"] >= 2
    assert final["higher_mass_half_minimum"] >= 2


def test_v19bh_orders_every_registered_non_solar_phenomenon() -> None:
    report = MODULE.build_report()
    ladder = report["cross_domain_prediction_ladder"]
    assert [row["tier"] for row in ladder] == [1, 2, 3]
    assert sum(len(row["phenomena"]) for row in ladder) == 7
    assert not report["priority"]["detailed_solar_optimization_now"]


def test_v19bh_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19bh_blind_cluster_admission" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
