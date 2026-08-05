from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19r_response_commissioning.json"


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
