from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bj_source_invariant_action_preselection.json"
SCRIPT = ROOT / "scripts" / "check_sigma_v19bj_source_invariant_action_preselection.py"
REPORT = (
    ROOT / "results" / "sigma_v19bj_source_invariant_action_preselection" / "report.json"
)
SPEC = importlib.util.spec_from_file_location("sigma_v19bj", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19bj_protocol_freeze_passes_every_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == "passed_target_blind_source_preselection_freeze"
    assert all(report["gate_results"].values())


def test_density_is_control_and_multiple_directional_features_are_registered() -> None:
    report = MODULE.build_report()
    summary = report["source_library_summary"]
    assert "D0_DENSITY_ONLY_CONTROL" in summary["ineligible_controls_or_future_only"]
    assert len(summary["eligible_after_gas"]) == 5
    assert len(summary["directional_after_gas"]) >= 4


def test_source_gate_requires_transfer_novelty_projection_and_resolution() -> None:
    gates = MODULE.build_report()["identifiability_gates"]
    assert gates["minimum_clusters_passing_each_advanced_feature"] == 2
    assert gates["minimum_projection_draw_pass_fraction"] >= 0.90
    assert gates["minimum_leave_one_region_out_pass_fraction"] >= 0.90
    assert gates["maximum_resolution_change_in_normalized_amplitude_fraction"] <= 0.10
    assert (
        gates["minimum_cross_validated_variance_not_predicted_by_total_density_fraction"]
        >= 0.20
    )
    assert not gates["lensing_or_halo_agreement_used_for_selection"]


def test_three_action_placements_do_not_select_an_action() -> None:
    report = MODULE.build_report()
    assert {row["id"] for row in report["action_placement_classes"]} == {
        "P1_CONSTRAINED_COMPOSITE_RESPONSE",
        "P2_CAUSAL_DYNAMIC_RESPONSE",
        "P3_DEGENERATE_PURE_METRIC_NONLINEAR_VERTEX",
    }
    assert not any(report["theory_state"].values())


def test_execution_waits_for_complete_gas_chain() -> None:
    prerequisites = MODULE.build_report()["execution_prerequisites"]
    assert prerequisites["terminal_v19w4_unified_response_pass"]
    assert prerequisites["terminal_v19x2_commissioning_pass"]
    assert prerequisites["all_494_frozen_regional_temperature_fits_pass"]
    assert prerequisites["common_grid_gas_density_temperature_pressure_entropy_posterior_exists"]
    assert prerequisites["current_state"].startswith("pending")


def test_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    actual = json.loads(REPORT.read_text(encoding="utf-8"))
    assert actual == expected
