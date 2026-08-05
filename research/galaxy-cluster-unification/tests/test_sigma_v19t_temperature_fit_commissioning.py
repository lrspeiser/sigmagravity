from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19t_temperature_fit_commissioning.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19t_freezes_model_and_gates_before_loading_spectrum() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    model = config["model"]
    assert model["expression"] == "xstbabs * xsapec"
    assert model["fit_energy_keV"] == [0.5, 7.0]
    assert model["statistic"] == "chi2xspecvar"
    assert model["galactic_nh_cm2_fixed"] == 4.38e20
    assert model["redshift_fixed"] == 0.296
    assert model["abundance_solar_fixed"] == 0.3
    assert model["temperature_keV"] == {
        "primary_initial": 8.0,
        "alternate_initials": [3.0, 15.0],
        "minimum": 1.0,
        "maximum": 30.0,
    }
    assert config["grouping"]["minimum_counts"] == 25
    assert config["gates"]["maximum_multistart_fractional_temperature_spread"] == 0.05
    assert config["gates"]["maximum_fractional_68_percent_half_width"] == 0.5
    assert config["gates"]["maximum_reduced_statistic"] == 1.5
    assert config["integrity"]["spectrum_loaded_in_sherpa_at_freeze"] is False
    assert config["integrity"]["temperature_or_normalization_fit_at_freeze"] is False
    assert config["integrity"]["fit_statistic_or_interval_known_at_freeze"] is False
