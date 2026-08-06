import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19cw_observation_hierarchy_equivalence as runner


CONFIG = ROOT / "configs" / "sigma_v19cw_observation_hierarchy_equivalence.json"
REPORT = ROOT / "results" / "sigma_v19cw_observation_hierarchy_equivalence" / "report.json"


def test_v19cw_is_abell_only_and_seals_science() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = payload["authorization"]
    assert authorization["run_abell2146_equivalence_commissioning"]
    assert not authorization["run_bullet_hierarchy"]
    assert not authorization["fit_temperature_or_abundance"]
    assert not authorization["change_cells_grouping_background_rule_or_response_weights"]
    assert not authorization["change_gravity_formula_parameter_source_state_or_lensing_target"]
    assert not authorization["run_v19bq_v19bs_or_derive_action"]


def test_v19cw_partition_is_outcome_blind_and_complete() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    hierarchy = payload["hierarchy"]
    groups = payload["direct_reference"]["observation_groups"]
    assert hierarchy["partition_key"] == "obsid"
    assert not hierarchy["partition_uses_scientific_outcome"]
    assert len(groups) == 10
    assert sum(groups.values()) == payload["direct_reference"]["cells"] == 1270
    assert hierarchy["intermediate_rmf_threshold"] == 0.0
    assert hierarchy["final_rmf_threshold"] == 1e-6


def test_additive_weighting_is_associative() -> None:
    area = np.array([1.0, 3.0, 5.0, 2.0])
    exposure = np.array([2.0, 7.0, 11.0, 13.0])
    direct = np.sum(area * exposure) / np.sum(exposure)
    groups = ([0, 1], [2, 3])
    group_area = np.array([np.sum(area[g] * exposure[g]) / np.sum(exposure[g]) for g in groups])
    group_exposure = np.array([np.sum(exposure[g]) for g in groups])
    hierarchical = np.sum(group_area * group_exposure) / np.sum(group_exposure)
    assert hierarchical == direct


def test_relative_difference_handles_zero_reference() -> None:
    assert runner.relative_difference(0.0, 0.0) == 0.0
    assert np.isfinite(runner.relative_difference(1.0, 0.0))


def test_v19cw_completed_result_authorizes_only_a_separate_bullet_freeze() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "observation_hierarchy_equivalent_and_bullet_recovery_may_be_frozen"
    assert payload["decision"] == "freeze_bullet_hierarchical_recovery"
    assert all(payload["gates"].values())
    assert payload["bullet_hierarchical_execution_authorized"]
    assert not payload["gravity_formula_or_parameter_changed"]
    assert not payload["source_state_or_lensing_target_opened"]
    assert not payload["v19bq_or_v19bs_run"]
    assert not payload["action_derived"]
    assert payload["comparisons"]["source"]["counts_exact"]
    assert payload["comparisons"]["source"]["grouping_exact"]
    assert payload["comparisons"]["arf"]["relative_l2_difference"] < 1e-7
    assert payload["comparisons"]["rmf"]["dense_relative_frobenius_difference"] < 1e-7
