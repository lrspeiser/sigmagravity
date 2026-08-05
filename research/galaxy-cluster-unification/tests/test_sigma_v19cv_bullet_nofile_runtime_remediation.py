from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19cv_bullet_nofile_runtime_remediation.json"


def test_v19cv_descriptor_projection_and_limit_are_sufficient() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pre = config["live_precondition"]
    projected = pre["bullet_integrated_cells"] * pre["descriptors_per_spectrum"] + pre["fixed_process_descriptors"]
    assert projected == pre["projected_bullet_open_descriptors"] == 11440
    assert projected > config["runtime_change"]["required_soft_before"]
    assert projected < config["runtime_change"]["soft_after"]


def test_v19cv_is_runtime_only_and_preserves_hard_limit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    change = config["runtime_change"]
    auth = config["authorization"]
    assert change["required_hard_before"] == change["hard_after"]
    assert auth["change_only_live_v19x2_parent_nofile_soft_limit"] is True
    assert auth["change_hard_limit"] is False
    assert auth["change_spectrum_response_weight_grouping_fit_or_gate"] is False
    assert auth["change_gravity_formula_or_parameter"] is False
    assert auth["open_lensing_halo_action_holdout_or_solar_payload"] is False
