from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19t_temperature_fit_commissioning.json"
RUNNER = ROOT / "scripts" / "fit_sigma_v19t_temperature_commissioning.py"
REPORT = ROOT / "results" / "sigma_v19t_temperature_fit_commissioning" / "report.json"


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


def test_v19t_temperature_fit_commissioning_passes_all_gates() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["status"] == (
        "temperature_fit_commissioning_passed_and_full_fit_pipeline_authorized"
    )
    assert all(report["gates"].values())
    assert [row["initial_temperature_keV"] for row in report["fits"]] == [
        8.0,
        3.0,
        15.0,
    ]
    primary = report["fits"][0]
    assert abs(primary["temperature_keV"] - 15.238254838689814) < 1e-9
    assert primary["optimization_attempts"] == ["levmar"]
    assert primary["dof"] == 23
    assert primary["reduced_statistic"] < 0.738
    confidence = primary["confidence_68_percent"]
    assert confidence["lower_keV"] < primary["temperature_keV"] < confidence["upper_keV"]
    assert report["multistart_fractional_temperature_spread"] < 1e-10
    assert report["primary_fractional_68_percent_half_width"] < 0.5
    product = report["frozen_grouped_pha"]
    path = ROOT / product["relative_path"]
    assert path.stat().st_size == product["bytes"]
    assert sha256(path) == product["sha256"]
    assert report["full_response_and_fit_production_authorized"] is True
    assert report["scientific_temperature_map_claimed"] is False
    assert report["thermal_stress_constructed"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
