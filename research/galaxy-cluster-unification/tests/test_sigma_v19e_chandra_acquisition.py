from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19e_chandra_acquisition.json"
REPORT = ROOT / "results" / "sigma_v19e_chandra_acquisition" / "provenance.json"
SCRIPT = ROOT / "scripts" / "download_sigma_v19e_chandra_inputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19e_download", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_freezes_published_matched_depth_observations() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["clusters"]["BULLET"]["obsids"] == [
        554,
        3184,
        4984,
        4985,
        4986,
        5355,
        5356,
        5357,
        5358,
        5361,
    ]
    assert config["clusters"]["ABELL2146"]["obsids"] == [
        10888,
        10464,
        13020,
        13021,
        13023,
        12247,
        12245,
        13120,
        12246,
        13138,
    ]
    assert config["deferred_robustness_upgrade"]["additional_exposure_ks"] > 1800


def test_wrapper_validates_frozen_ancestry_and_pair() -> None:
    module = load_module()
    config = module.validate(CONFIG)
    for key in ("member_extraction_config", "member_extraction_report"):
        assert config["parents"][f"{key}_sha256"] == digest(
            ROOT / config["parents"][key]
        )
    assert set(config["clusters"]) == {"BULLET", "ABELL2146"}


def test_acquisition_report_is_complete_and_target_blind() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["protocol_version"] == config["protocol_version"]
    assert report["config_sha256"] == digest(CONFIG)
    assert len(report["per_obsid"]) == 20
    assert {row["cluster"] for row in report["per_obsid"]} == {
        "BULLET",
        "ABELL2146",
    }
    for row in report["per_obsid"]:
        for role in config["required_roles_per_obsid"]:
            assert row["role_counts"][role] >= 1
    for record in report["records"]:
        path = ROOT / record["relative_path"]
        assert path.stat().st_size == record["bytes"]
        assert digest(path) == record["sha256"]
    assert report["lensing_target_opened"] is False
    assert report["temperature_map_constructed"] is False
