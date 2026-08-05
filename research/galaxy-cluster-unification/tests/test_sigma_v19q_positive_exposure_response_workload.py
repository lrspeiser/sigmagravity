from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19q_positive_exposure_response_workload.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19q_positive_exposure_response_workload.py"
REPORT = (
    ROOT
    / "results"
    / "sigma_v19q_positive_exposure_response_workload"
    / "report.json"
)


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


def test_v19q_passes_exact_conservation_and_authorizes_responses() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == "positive_exposure_response_workload_conserved_and_authorized"
    assert report["manifest"]["response_task_count"] == 5082
    assert all(report["global_gates"].values())
    clusters = {row["cluster"]: row for row in report["clusters"]}
    bullet = clusters["BULLET"]
    abell = clusters["ABELL2146"]
    assert bullet["response_task_count"] == 3812
    assert abell["response_task_count"] == 1270
    assert bullet["zero_exposure_events_rejected_inside_bins"] == 1
    assert abell["zero_exposure_events_rejected_inside_bins"] == 0
    assert bullet["science_count_delta"] == 0
    assert abell["science_count_delta"] == 0
    assert all(all(row["gates"].values()) for row in clusters.values())
    assert report["response_extraction_authorized"] is True
    assert report["spectrum_or_response_constructed"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
