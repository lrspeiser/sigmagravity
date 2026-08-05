from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19bq_v19x4b_observed_source_successor_preflight as checker
import freeze_sigma_v19bq_v19x4b_observed_source_invariant_scoring as freezer
import run_sigma_v19bq_v19x4b_observed_source_invariant_scoring as runner

CONFIG = ROOT / "configs" / "sigma_v19bq_v19x4b_observed_source_successor_preflight.json"
REPORT = ROOT / "results" / "sigma_v19bq_v19x4b_observed_source_successor_preflight" / "report.json"
ORIGINAL = ROOT / "configs" / "sigma_v19bp_observed_source_invariant_scoring.json"


def test_frozen_preflight_report_is_current_and_target_sealed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    rebuilt = checker.execute(config, CONFIG)
    assert frozen == rebuilt
    assert all(frozen["gates"].values())
    assert all(frozen["hash_gates"].values())
    assert not frozen["terminal_gas_stellar_or_source_result_opened"]
    assert not frozen["lensing_halo_action_gravity_or_holdout_payload_opened"]


def test_successor_contract_preserves_every_source_decision_section() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    for section in freezer.SCIENCE_SECTIONS:
        assert config["successor_contract"]["science_section_sha256"][section] == (
            freezer.canonical_sha256(original[section])
        )


def test_inherited_decision_requires_i4_direction_and_all_six_branches() -> None:
    def branch(direction: bool, amplitude: bool, scalar: bool) -> dict:
        return {
            "candidates": {
                "I4": {"direction_pass": direction, "amplitude_or_scalar_pass": amplitude},
                "I5": {"direction_pass": False, "amplitude_or_scalar_pass": scalar},
            }
        }

    decision = runner.inherited_v19bp.aggregate_source_decision(
        [branch(True, False, True) for _ in range(6)], 6
    )
    assert decision["action_derivation_authorized"]
    with pytest.raises(ValueError, match="every registered branch"):
        runner.inherited_v19bp.aggregate_source_decision(
            [branch(True, False, True) for _ in range(5)], 6
        )
