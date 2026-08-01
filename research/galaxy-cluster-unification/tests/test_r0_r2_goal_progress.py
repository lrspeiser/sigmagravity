from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_full_premise_goal_progress_does_not_confuse_inventory_with_readiness() -> None:
    report = json.loads((ROOT / "results/r0_r2_goal_progress/report.json").read_text())
    ledger = pd.read_csv(ROOT / report["output"])
    terminal_exists = (
        ROOT / "results/r1_rxj2129_terminal_observable_disposition/report.json"
    ).is_file()

    assert report["report_version"] == "R0-R2-goal-progress-0.4-full-completion-evidence"
    assert report["requirements"] == 8
    assert report["completion_evidence_requirements"] == 11
    assert report["completion_evidence_checks"]["unique_scored_source_files_rehashed"] == 133
    assert report["completion_evidence_checks"]["scored_source_hash_failures"] == []
    assert report["completion_evidence_checks"]["BCG_profile_hash_failures"] == []
    assert report["full_goal_complete"] is terminal_exists
    assert report["terminal_stop_rule_satisfied"] is terminal_exists
    assert report["premise_passed"] is False
    assert report["strict_ready_systems"] == 0
    assert report["target_strict_systems"] == 10
    assert report["requirements_incomplete_or_not_authorized"] == 4
    assert report["requirements_closed_by_hard_public_data_ceiling"] == 4
    assert report["hard_public_data_shortfall_established"] is True
    assert report["current_universe_structural_ceiling"] == 3
    assert report["minimum_strict_system_deficit_even_if_RXJ2129_passes"] == 9
    assert set(ledger["requirement_id"]) == {
        "R0_PROVENANCE",
        "R0_CLASH_ACQUISITION",
        "R0_BCG_ACQUISITION",
        "R1_TEN_SYSTEM_FREEZE",
        "R2_DYNAMICAL_RESPONSE",
        "R2_WEYL_RESPONSE",
        "R2_LATENT_CROSS_VALIDATION",
        "THEORY_STOP_RULE",
    }
    assert ledger.set_index("requirement_id").loc["R0_PROVENANCE", "status"] == "pass"
    assert ledger.set_index("requirement_id").loc[
        "R1_TEN_SYSTEM_FREEZE", "status"
    ] == "hard_public_data_shortfall"
    assert ledger.set_index("requirement_id").loc[
        "R2_LATENT_CROSS_VALIDATION", "status"
    ] == "empirically_unidentifiable_with_audited_public_data"
    assert report["authorization"]["select_another_system"] is False
    assert report["authorization"]["reconstruct_dynamical_or_Weyl_response"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
    assert report["authorization"][
        "finish_already_running_RXJ2129_observable_gates"
    ] is (not terminal_exists)
