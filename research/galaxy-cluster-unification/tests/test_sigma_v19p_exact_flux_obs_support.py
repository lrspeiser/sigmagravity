from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19p_exact_flux_obs_support.json"
INVENTORY = (
    ROOT / "results" / "sigma_v19p_exact_flux_obs_support" / "input_inventory.json"
)
RUNNER = ROOT / "scripts" / "run_sigma_v19p_exact_flux_obs_support.py"
REPORT = ROOT / "results" / "sigma_v19p_exact_flux_obs_support" / "report.json"
DIAGNOSTIC = (
    ROOT / "results" / "sigma_v19p_exact_flux_obs_support" / "pixel_diagnostic.json"
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


def test_v19p_fails_closed_on_the_single_zero_exposure_event() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == "exact_flux_obs_support_or_response_workload_gate_failed"
    assert report["manifest"]["response_task_count"] == 5082
    clusters = {row["cluster"]: row for row in report["clusters"]}
    bullet = clusters["BULLET"]
    abell = clusters["ABELL2146"]
    assert bullet["science_count_delta"] == 1
    assert bullet["per_observation_image_sum_changed_pixel_count"] == 0
    assert bullet["gates"]["task_key_set_equals_v19n"] is True
    assert bullet["gates"]["total_science_count_conservation_exact"] is False
    assert all(abell["gates"].values())
    bad_observations = [
        row for row in bullet["observations"] if row["science_count_delta"]
    ]
    assert len(bad_observations) == 1
    assert bad_observations[0]["obsid"] == 554
    assert diagnostic["v19p_report_sha256"] == sha256(REPORT)
    assert diagnostic["delta_pixel_count_inside_regions"] == 1
    delta = diagnostic["deltas"][0]
    assert delta["cluster"] == "BULLET"
    assert delta["obsid"] == 554
    assert delta["bin_id"] == 24
    assert delta["event_minus_image"] == 1
    assert delta["flux_obs_exposure"] == 0.0
    assert delta["events_in_pixel"][0]["ccd_id"] == 3
    assert report["response_extraction_authorized"] is False
    assert report["spectrum_or_response_constructed"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
