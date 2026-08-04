from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fit_sigma_v17c_regional_temperatures.py"
CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("sigma_v17c_regional_fit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_regional_fit_requires_matching_authorized_reports(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"protocol_version": "P"}), encoding="utf-8")
    integrated_path = tmp_path / "integrated.json"
    integrated = {
        "status": "both_integrated_temperature_gates_passed",
        "protocol_version": "P",
        "config_sha256": _sha256(config_path),
        "regional_fit_authorized": True,
    }
    integrated_path.write_text(json.dumps(integrated), encoding="utf-8")
    spectra_path = tmp_path / "regional_spectra.json"
    spectra = {
        "status": "both_frozen_regional_spectra_extracted_combined_and_grouped",
        "protocol_version": "P",
        "config_sha256": _sha256(config_path),
        "integrated_temperatures_report_sha256": _sha256(integrated_path),
        "regional_temperature_fit_authorized": True,
    }
    spectra_path.write_text(json.dumps(spectra), encoding="utf-8")
    module.validate_inputs(
        config_path,
        {"protocol_version": "P"},
        spectra_path,
        spectra,
        integrated_path,
        integrated,
    )
    integrated["status"] = "integrated_temperature_gate_failed"
    with pytest.raises(RuntimeError, match="has not passed"):
        module.validate_inputs(
            config_path,
            {"protocol_version": "P"},
            spectra_path,
            spectra,
            integrated_path,
            integrated,
        )


def test_regional_fit_freezes_integrated_abundance_and_profiles_temperature() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'integrated_fit["parameters"]["abundance_solar"]' in source
    assert "ui.freeze(thermal.Abundanc)" in source
    assert "ui.thaw(thermal.kT)" in source
    assert "ui.thaw(thermal.norm)" in source
    assert "ui.conf(thermal.kT)" in source
    assert '"no_fit_selection"' in CONFIG.read_text(encoding="utf-8")


def test_regional_gate_calculation_is_frozen_before_data_exist() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")
    assert "(upper - lower) / (2.0 * temperature)" in source
    assert config["gates"]["regional"] == {
        "finite_temperature_and_interval": True,
        "maximum_fractional_68_percent_half_width": 0.5,
        "maximum_reduced_statistic": 1.5,
        "minimum_passing_regions_per_cluster": 12,
        "failure_rule": (
            "retain and report every failed region; do not merge, split, smooth, "
            "or refit it with a different model"
        ),
    }
    assert '"thermal_stress_construction_authorized": all_passed' in source
    assert '"thermal_stress_constructed": False' in source
    assert '"lensing_target_opened": False' in source


def test_regional_fit_exception_is_retained_and_cannot_authorize_a_map() -> None:
    module = _load_module()
    row = module.failed_region_result(
        "PLCKG287",
        {
            "region_id": 7,
            "source_region": "region_007.reg",
            "source_region_sha256": "abc",
        },
        RuntimeError("optimizer failed"),
    )

    assert row["region_id"] == 7
    assert row["fit_completed"] is False
    assert row["parameters"]["temperature_keV"] is None
    assert row["gates"]["all_passed"] is False
    assert "RuntimeError: optimizer failed" in row["fit_exception"]
