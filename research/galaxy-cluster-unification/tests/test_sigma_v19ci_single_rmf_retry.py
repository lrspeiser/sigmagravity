from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ci_single_rmf_retry.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v19ci_static_parents_and_runner_are_frozen() -> None:
    config = load_config()
    for spec in config["parents"].values():
        assert sha256(ROOT / spec["path"]) == spec["sha256"]
    runner = config["implementation"]["runner"]
    assert sha256(ROOT / runner["path"]) == runner["sha256"]


def test_v19ci_scope_is_one_exact_missing_rmf_checkpoint() -> None:
    config = load_config()
    boundary = config["initial_failure_boundary"]
    retry = config["retry"]
    assert boundary["missing_cells_at_launch"] == 384
    assert boundary["completed_recovery_cells"] == 383
    assert boundary["failed_cells"] == 1
    assert boundary["failed_cell"] == "BULLET_bin384_obs5358_ccd0"
    assert boundary["failed_token"] == "c3432"
    assert boundary["failed_stage"] == "specextract RMF creation"
    assert retry["maximum_additional_failed_cell_attempts"] == 1
    assert retry["reuse_383_completed_checkpoints"]
    assert retry["reexecute_only_missing_completed_checkpoint"]


def test_v19ci_preserves_failure_and_forbids_science_changes() -> None:
    config = load_config()
    authorization = config["authorization"]
    assert authorization["preserve_failed_partial_attempt"]
    assert not authorization[
        "modify_or_delete_383_completed_recovery_checkpoints"
    ]
    assert not authorization["modify_protected_base_archive"]
    assert not authorization["change_manifest_or_drop_failed_cell"]
    assert not authorization["change_response_placement_or_science_setting"]
    assert not authorization["open_lensing_halo_action_gravity_or_holdout"]
    assert not authorization["derive_or_select_action"]
    assert not authorization["change_gravity_formula_or_parameter"]
    assert not authorization["automatic_second_retry_if_this_fails"]
