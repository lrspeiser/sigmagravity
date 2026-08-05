from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19s_hi4pi_acquisition.json"


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
