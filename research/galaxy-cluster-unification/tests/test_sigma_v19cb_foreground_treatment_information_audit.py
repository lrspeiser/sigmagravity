from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sigma_v19cb_foreground_treatment_information_audit.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19cb_foreground_treatment_information_audit.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19cb", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v19cb_weight_branches_have_declared_meanings() -> None:
    branches = {row["id"]: row for row in config()["treatment_branches"]}
    clean = {"foreground_astrometric_evidence": "false", "quality_controlled_foreground_contamination": "false"}
    quality = {"foreground_astrometric_evidence": "true", "quality_controlled_foreground_contamination": "true"}
    weak = {"foreground_astrometric_evidence": "true", "quality_controlled_foreground_contamination": "false"}
    assert MODULE.candidate_weight(clean, branches["retain_all"]) == 1.0
    assert MODULE.candidate_weight(quality, branches["soft_quality_0p1"]) == 0.1
    assert MODULE.candidate_weight(quality, branches["mask_quality_diagnostic"]) == 0.0
    assert MODULE.candidate_weight(weak, branches["mask_quality_diagnostic"]) == 1.0
    assert MODULE.candidate_weight(weak, branches["mask_any_astrometry_diagnostic"]) == 0.0


def test_v19cb_discloses_exploration_and_authorizes_no_mask() -> None:
    cfg = config()
    assert cfg["honesty_boundary"]["complete_v19ca_source_result_inspected_before_freeze"]
    assert not cfg["honesty_boundary"]["gravity_kinematic_or_lensing_target_inspected"]
    assert not cfg["honesty_boundary"]["this_is_a_preregistered_theory_or_holdout_gate"]
    boundary = cfg["access_boundary"]
    for key in ("hard_star_mask_authorized", "treatment_branch_selected", "candidate_or_galaxy_removed", "optical_counterpart_selected", "wallaby_kinematic_table_row_read", "rotation_speed_or_velocity_field_read", "gravity_formula_residual_or_halo_result_read", "development_validation_holdout_split_selected", "gravity_action_or_constant_changed", "lensing_payload_opened", "solar_system_optimization_performed"):
        assert not boundary[key]
