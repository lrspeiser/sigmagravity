from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19r_response_commissioning.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19r_response_commissioning.py"
REPORT = ROOT / "results" / "sigma_v19r_response_commissioning" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19r_freezes_a_unique_manifest_selected_cell_before_responses() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    selected = config["selection"]
    assert selected["selection_uses_response_or_temperature_outcome"] is False
    assert selected["unique_maximum"] is True
    assert (
        selected["cluster"],
        selected["bin_id"],
        selected["obsid"],
        selected["ccd_id"],
    ) == ("BULLET", 390, 5356, 2)
    assert selected["source_band_events"] == 625
    assert selected["background_band_events"] == 232
    assert config["integrity"]["source_or_background_pha_existed_at_freeze"] is False
    assert config["integrity"]["arf_or_rmf_existed_at_freeze"] is False
    assert config["integrity"]["response_output_opened_at_freeze"] is False


def test_v19r_commissioned_response_passes_every_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == (
        "commissioning_response_passed_and_full_response_production_authorized"
    )
    assert all(report["gates"].values())
    assert report["preflight"] == {
        "positive_exposure_task_events": 625,
        "source_band_events": 625,
        "background_band_events": 232,
    }
    assert report["source_pha_channel_audit"]["exact"] is True
    assert report["background_pha_channel_audit"]["exact"] is True
    assert report["response_audit"]["arf_positive_bins"] == 1070
    assert report["response_audit"]["rmf_nonzero_elements"] == 538171
    assert report["full_response_production_authorized"] is True
    assert report["temperature_density_mach_or_speed_fitted"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
    for item in report["frozen_snapshot"].values():
        path = ROOT / item["relative_path"]
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
