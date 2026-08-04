from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_sigma_v17d_thermal_stress_maps.py"
CONFIG = ROOT / "configs" / "sigma_v17d_thermal_stress_map.json"
SPECTRAL_CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("sigma_v17d_thermal_maps", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_thermal_map_inputs_and_claim_boundary_are_hash_locked() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for hash_key, path_key in (
        ("dynamical_stress_gate_sha256", "dynamical_stress_gate"),
        ("static_spent_baseline_sha256", "static_spent_baseline"),
        ("spectral_protocol_sha256", "spectral_protocol"),
    ):
        assert config["parents"][hash_key] == _sha256(ROOT / config["parents"][path_key])
    for cluster in config["clusters"].values():
        assert cluster["baryon_map_sha256"] == _sha256(ROOT / cluster["baryon_map"])
        assert cluster["temperature_binmap_sha256"] == _sha256(
            ROOT / cluster["temperature_binmap"]
        )
    assert config["integrity"]["v17_lensing_target_opened"] is False
    assert config["integrity"]["per_cluster_normalization"] is False
    assert config["integrity"]["per_cluster_scale_or_orientation"] is False


def test_temperature_assignment_uses_global_fallback_and_retains_every_bin() -> None:
    module = _load_module()
    ids = np.array([[-1, 0, 1], [1, 0, -1]], dtype=int)
    total, contrast, resolved = module.assign_temperature_fields(
        ids,
        {0: 8.0, 1: 12.0},
        10.0,
    )
    assert np.array_equal(total, np.array([[10.0, 8.0, 12.0], [12.0, 8.0, 10.0]]))
    assert np.array_equal(contrast, np.array([[0.0, -2.0, 2.0], [2.0, -2.0, 0.0]]))
    assert np.array_equal(resolved, ids >= 0)
    with pytest.raises(RuntimeError, match="unknown regions"):
        module.assign_temperature_fields(ids, {0: 8.0}, 10.0)


def test_thermal_energy_conversion_and_one_metric_feature_inventory() -> None:
    module = _load_module()
    ratio = module.energy_ratio_per_kev(0.61)
    assert 1.7e-6 < ratio < 1.8e-6
    axis = np.linspace(-8.0, 8.0, 17)
    east, north = np.meshgrid(axis, axis)
    total = np.exp(-(east**2 + north**2) / 8.0) * 1e-6
    contrast = (east / 8.0) * total
    config = {
        "feature_construction": {
            "target_half_width_kpc": 4.0,
            "target_grid_points": 9,
            "gaussian_scales_kpc": [2.0],
            "fourier_padding_factor": 2,
        }
    }
    features = module.feature_inventory(axis, total, contrast, config)
    assert set(features) == {
        "target_axis_kpc",
        "thermal_total_smooth_2kpc_convergence",
        "thermal_total_smooth_2kpc_shear_1",
        "thermal_total_smooth_2kpc_shear_2",
        "thermal_contrast_smooth_2kpc_convergence",
        "thermal_contrast_smooth_2kpc_shear_1",
        "thermal_contrast_smooth_2kpc_shear_2",
    }
    assert all(np.isfinite(values).all() for values in features.values())


def test_map_construction_cannot_bypass_regional_authorization(tmp_path: Path) -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    integrated_path = tmp_path / "integrated.json"
    integrated = {
        "status": "both_integrated_temperature_gates_passed",
        "config_sha256": _sha256(SPECTRAL_CONFIG),
    }
    integrated_path.write_text(json.dumps(integrated), encoding="utf-8")
    regional_path = tmp_path / "regional.json"
    regional = {
        "status": "both_regional_temperature_gates_passed",
        "config_sha256": _sha256(SPECTRAL_CONFIG),
        "integrated_temperatures_report_sha256": _sha256(integrated_path),
        "thermal_stress_construction_authorized": True,
    }
    regional_path.write_text(json.dumps(regional), encoding="utf-8")
    module.validate_authorization(
        CONFIG,
        config,
        SPECTRAL_CONFIG,
        regional_path,
        regional,
        integrated_path,
        integrated,
    )
    regional["thermal_stress_construction_authorized"] = False
    with pytest.raises(RuntimeError, match="not authorized"):
        module.validate_authorization(
            CONFIG,
            config,
            SPECTRAL_CONFIG,
            regional_path,
            regional,
            integrated_path,
            integrated,
        )


def test_source_builder_does_not_import_or_read_a_lensing_target() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "glafic_comparator" not in source
    assert "frozen_sky_field" not in source
    assert '"inverse_coefficients_fit": False' in source
    assert '"lensing_target_opened": False' in source
