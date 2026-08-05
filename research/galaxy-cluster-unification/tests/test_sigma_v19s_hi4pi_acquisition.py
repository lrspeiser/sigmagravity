from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19s_hi4pi_acquisition.json"
RUNNER = ROOT / "scripts" / "download_sigma_v19s_hi4pi.py"
REPORT = ROOT / "results" / "sigma_v19s_hi4pi_acquisition" / "provenance.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19s_was_frozen_before_querying_hi4pi() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert config["source"]["map"] == "h1_nh_HI4PI.fits"
    assert config["source"]["cone_radius_deg"] == 0.1
    assert config["source"]["selected_statistic"].startswith("inverse-distance")
    assert set(config["targets"]) == {"BULLET", "ABELL2146"}
    assert config["integrity"]["query_executed_at_freeze"] is False
    assert config["integrity"]["v19s_nh_value_known_at_freeze"] is False
    assert config["integrity"]["temperature_or_abundance_fit_at_freeze"] is False


def test_v19s_acquired_both_frozen_hi4pi_columns() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == "both_frozen_hi4pi_columns_acquired_and_hashed"
    records = {row["cluster"]: row for row in report["records"]}
    assert records["BULLET"]["weighted_average_nh_cm2"] == 4.38e20
    assert records["ABELL2146"]["weighted_average_nh_cm2"] == 2.62e20
    for record in records.values():
        path = ROOT / record["relative_path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
        assert record["map"] == "h1_nh_HI4PI.fits"
    assert report["temperature_fit_authorized"] is True
    assert report["temperature_or_abundance_fit_run"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
