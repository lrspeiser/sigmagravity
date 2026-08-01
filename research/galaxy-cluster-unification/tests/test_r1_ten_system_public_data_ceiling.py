import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "r1_ten_system_public_data_ceiling_protocol.json"
REPORT = ROOT / "results" / "r1_ten_system_public_data_ceiling" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ceiling_protocol_was_frozen_before_active_rxj2129_outcomes() -> None:
    protocol = load(PROTOCOL)
    assert protocol["status"] == (
        "frozen_after_third_external_nonpromotion_before_RXJ2129_H2_or_X4_outcome"
    )
    assert protocol["selection_blind"]
    assert not protocol["gravity_residuals_seen"]
    assert protocol["scope"]["strict_same_system_target"] == 10
    assert protocol["authorization"]["finish_RXJ2129_H2_and_X4"]


def test_hard_public_data_shortfall_is_derived_from_every_frozen_check() -> None:
    report = load(REPORT)
    assert report["hard_public_data_shortfall_established"] == all(
        report["checks"].values()
    )
    assert report["audited_public_data_universe"][
        "unique_BCG_hosts_source_screened"
    ] >= 30
    assert report["current_universe_structural_ceiling"] == 3
    assert report["minimum_new_rank_three_systems_required"] >= 7
    assert report["audited_public_data_universe"][
        "external_one_off_candidates_promoted"
    ] == 0


def test_ceiling_is_bound_to_the_current_upstream_artifacts() -> None:
    report = load(REPORT)
    for record in report["inputs"].values():
        path = ROOT / record["path"]
        assert path.is_file()
        assert sha256(path) == record["sha256"]


def test_rxj2129_cannot_change_the_ten_system_conclusion() -> None:
    branch = load(REPORT)["RXJ2129_outcome_independence"]
    assert branch["maximum_strict_ready_if_RXJ2129_passes"] == 1
    assert branch["minimum_remaining_strict_system_deficit_if_RXJ2129_passes"] == 9
    assert not branch["ten_system_shortfall_changes_if_RXJ2129_passes"]


def test_ceiling_forbids_population_response_and_new_force_law() -> None:
    report = load(REPORT)
    assert report["decision"]["R1C_ten_system_freeze"] == (
        "unattainable_with_audited_public_data"
    )
    authorization = report["authorization"]
    assert not authorization["freeze_ten_system_sample"]
    assert not authorization["select_fourth_external_one_off_target"]
    assert not authorization["run_population_R2_cross_validation"]
    assert not authorization["claim_one_or_two_potential_population_identification"]
    assert not authorization["fit_new_force_or_action"]
