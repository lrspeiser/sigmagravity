from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_current_candidate_universe_cannot_supply_the_ten_system_freeze() -> None:
    report = json.loads(
        (ROOT / "results/r1_ten_system_attainability/report.json").read_text()
    )
    ledger = pd.read_csv(ROOT / report["output"])

    assert report["candidate_systems_evaluated"] == 15
    assert report["target_strict_systems"] == 10
    assert report["current_strict_ready_systems"] == 0
    assert report["current_candidate_universe_structural_ceiling"] == 3
    assert set(report["structural_ceiling_systems"]) == {
        "Abell 1689",
        "MACS J1206",
        "RX J2129",
    }
    assert report["accepted_dynamics_and_rank_three_systems"] == ["RX J2129"]
    assert report["minimum_new_rank_three_systems_required_even_if_every_ceiling_system_is_repaired"] == 7
    assert report["ten_system_freeze_attainable_from_current_candidate_universe"] is False
    assert int(ledger["structural_rank_three_ceiling"].sum()) == 3
    assert report["authorization"]["freeze_ten_system_sample"] is False
    assert report["authorization"]["fit_new_force_or_action"] is False
