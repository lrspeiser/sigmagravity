from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19q_positive_exposure_response_workload.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19q_freezes_one_universal_zero_exception_support_rule() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    rule = config["science_support_rule"]
    assert rule["exposure_threshold"] == 0.0
    assert rule["comparison"] == "strictly_greater_than"
    assert rule["same_rule_for_every_cluster_observation_region_and_ccd"] is True
    assert rule["event_identity_obsid_region_ccd_or_position_exception"] is False
    assert config["integrity"]["positive_exposure_event_filter_applied_at_freeze"] is False
    assert config["integrity"]["cross_sample_conservation_outcome_known_at_freeze"] is False
    assert config["sample"]["expected_response_task_count_total"] == 5082
