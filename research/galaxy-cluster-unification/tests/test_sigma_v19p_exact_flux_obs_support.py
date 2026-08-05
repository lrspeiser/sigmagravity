from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19p_exact_flux_obs_support.json"
INVENTORY = (
    ROOT / "results" / "sigma_v19p_exact_flux_obs_support" / "input_inventory.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19p_was_hash_frozen_before_exact_support_was_opened() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert inventory["integrity"]["fits_array_opened"] is False
    assert inventory["integrity"]["event_row_read"] is False
    assert inventory["integrity"]["corrected_count_or_task_outcome_known"] is False
    assert config["integrity"]["per_observation_fits_array_opened_at_freeze"] is False
    assert config["integrity"]["flux_obs_fov_applied_at_freeze"] is False
    assert config["integrity"]["corrected_science_count_known_at_freeze"] is False
    assert config["conservation_gates"]["exact_total_science_count_delta"] == 0
    assert config["sample"]["expected_response_task_count_total"] == 5082
