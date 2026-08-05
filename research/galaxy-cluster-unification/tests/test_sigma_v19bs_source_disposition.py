from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_sigma_v19bs_source_disposition_preflight as checker
import run_sigma_v19bs_source_disposition as runner

CONFIG = ROOT / "configs" / "sigma_v19bs_source_disposition.json"
REPORT = ROOT / "results" / "sigma_v19bs_source_disposition" / "preflight_report.json"


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def decision(direction: bool, i4_amplitude: bool, i5_scalar: bool) -> dict:
    return {
        "I4_direction_pass": direction,
        "I4_amplitude_pass": i4_amplitude,
        "I5_scalar_pass": i5_scalar,
        "action_derivation_authorized": direction and (i4_amplitude or i5_scalar),
    }


def test_frozen_preflight_is_current_and_result_sealed() -> None:
    frozen = json.loads(REPORT.read_text(encoding="utf-8"))
    rebuilt = checker.execute(config(), CONFIG)
    assert frozen == rebuilt
    assert all(frozen["gates"].values())
    assert not frozen["terminal_v19bq_result_opened"]
    assert not frozen["lensing_halo_galaxy_action_gravity_or_holdout_payload_opened"]


def test_action_placement_classes_are_exactly_inherited_from_v19bj() -> None:
    current = config()
    v19bj = json.loads(
        (ROOT / current["parents"]["v19bj_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert current["action_placement_classes"] == v19bj["action_placement_classes"]


def test_i4_amplitude_pass_authorizes_only_time_even_compatible_classes() -> None:
    result = runner.disposition_from_decision(decision(True, True, False), config())
    assert result["action_derivation_authorized"]
    assert result["source_terms_authorized"] == [
        "I4_THERMODYNAMIC_GRADIENT_STRESS_DIRECTION",
        "I4_THERMODYNAMIC_GRADIENT_STRESS_AMPLITUDE",
    ]
    assert [row["id"] for row in result["compatible_action_placement_classes"]] == [
        "P1_CONSTRAINED_COMPOSITE_RESPONSE",
        "P3_DEGENERATE_PURE_METRIC_NONLINEAR_VERTEX",
    ]
    assert result["excluded_action_placement_classes"] == [
        "P2_CAUSAL_DYNAMIC_RESPONSE"
    ]


def test_i5_can_rescue_strength_but_not_i4_direction() -> None:
    rescued = runner.disposition_from_decision(decision(True, False, True), config())
    assert rescued["action_derivation_authorized"]
    assert "I5_BAROCLINICITY_SCALAR" in rescued["source_terms_authorized"]
    failed = runner.disposition_from_decision(decision(False, False, True), config())
    assert not failed["action_derivation_authorized"]
    assert failed["compatible_action_placement_classes"] == []


def test_inconsistent_terminal_decision_fails_closed() -> None:
    inconsistent = decision(True, True, False)
    inconsistent["action_derivation_authorized"] = False
    with pytest.raises(RuntimeError, match="violates the frozen I4/I5 logic"):
        runner.disposition_from_decision(inconsistent, config())
