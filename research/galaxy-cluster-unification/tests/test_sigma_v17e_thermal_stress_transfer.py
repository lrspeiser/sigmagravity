from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v17e_thermal_stress_transfer.py"
CONFIG = ROOT / "configs" / "sigma_v17e_thermal_stress_transfer.json"
THERMAL_CONFIG = ROOT / "configs" / "sigma_v17d_thermal_stress_map.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("sigma_v17e_transfer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v17e_protocol_hashes_the_frozen_parents_and_forbids_switches() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("dynamical_stress_gate", "dynamical_stress_gate_sha256"),
        ("static_baseline_config", "static_baseline_config_sha256"),
        ("static_incremental_control_report", "static_incremental_control_report_sha256"),
        ("thermal_source_protocol", "thermal_source_protocol_sha256"),
    ):
        assert config["parents"][hash_key] == _sha256(
            ROOT / config["parents"][path_key]
        )
    assert config["sample"]["clusters"] == ["AS295", "PLCKG287"]
    assert config["thermal_features"]["object_label_or_regime_switch"] is False
    assert (
        config["thermal_features"]["per_cluster_normalization_scale_orientation_or_shear"]
        is False
    )
    assert config["integrity"]["galaxy_cluster_formula_switch"] is False
    assert config["integrity"]["per_cluster_gravity_parameters"] == 0


def test_nested_thermal_feature_families_are_exact_and_one_metric() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    total = module.thermal_feature_names(config, "thermal_total")
    component = module.thermal_feature_names(config, "thermal_component")
    assert total == [
        "thermal_total_smooth_25kpc",
        "thermal_total_smooth_75kpc",
        "thermal_total_smooth_150kpc",
    ]
    assert component == [
        *total,
        "thermal_contrast_smooth_25kpc",
        "thermal_contrast_smooth_75kpc",
        "thermal_contrast_smooth_150kpc",
    ]
    assert len(component) == len(set(component)) == 6


def test_target_authorization_requires_immutable_blind_source_products(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    clusters = []
    for name in config["sample"]["clusters"]:
        product = tmp_path / f"{name}.npz"
        product.write_bytes(name.encode("ascii"))
        clusters.append(
            {
                "cluster": name,
                "product": str(product),
                "product_sha256": _sha256(product),
            }
        )
    report = {
        "status": config["authorization"]["required_thermal_source_status"],
        "config_sha256": config["parents"]["thermal_source_protocol_sha256"],
        "source_maps_frozen": True,
        "inverse_coefficients_fit": False,
        "lensing_target_opened": False,
        "clusters": clusters,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    products = module.validate_authorization(CONFIG, config, report_path, report)
    assert set(products) == {"AS295", "PLCKG287"}

    report["lensing_target_opened"] = True
    with pytest.raises(RuntimeError, match="lensing_target_opened"):
        module.validate_authorization(CONFIG, config, report_path, report)


def test_saved_thermal_arrays_load_as_complete_metric_triplets(tmp_path: Path) -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    thermal_config = json.loads(THERMAL_CONFIG.read_text(encoding="utf-8"))
    axis = np.linspace(-2.0, 2.0, 5)
    arrays: dict[str, np.ndarray] = {
        "target_axis_kpc": axis,
        "source_axis_kpc": axis,
        "q_total": np.ones((5, 5)),
        "q_contrast": np.zeros((5, 5)),
    }
    for name in module.thermal_feature_names(config, "thermal_component"):
        for index, channel in enumerate(("convergence", "shear_1", "shear_2"), start=1):
            arrays[f"{name}_{channel}"] = np.full((5, 5), float(index))
    product = tmp_path / "thermal.npz"
    np.savez_compressed(product, **arrays)
    features = module.load_thermal_features(
        product,
        axis,
        config,
        thermal_config,
    )
    assert set(features) == set(
        module.thermal_feature_names(config, "thermal_component")
    )
    for feature in features.values():
        np.testing.assert_allclose(feature.convergence, 1.0)
        np.testing.assert_allclose(feature.shear_1, 2.0)
        np.testing.assert_allclose(feature.shear_2, 3.0)


def test_resolution_stability_metric_includes_error_alignment_and_power() -> None:
    module = _load_module()
    primary = [
        {
            "train_cluster": "A",
            "test_cluster": "B",
            "full_field_NRMSE": 0.5,
            "residual_shear_alignment_cosine": 0.6,
            "residual_power_closed": 0.3,
        },
        {
            "train_cluster": "B",
            "test_cluster": "A",
            "full_field_NRMSE": 0.4,
            "residual_shear_alignment_cosine": 0.7,
            "residual_power_closed": 0.4,
        },
    ]
    doubled = [
        {
            "train_cluster": "A",
            "test_cluster": "B",
            "full_field_NRMSE": 0.505,
            "residual_shear_alignment_cosine": 0.59,
            "residual_power_closed": 0.295,
        },
        {
            "train_cluster": "B",
            "test_cluster": "A",
            "full_field_NRMSE": 0.404,
            "residual_shear_alignment_cosine": 0.69,
            "residual_power_closed": 0.39,
        },
    ]
    result = module.resolution_change(0.45, primary, 0.4545, doubled)
    np.testing.assert_allclose(result["maximum_change"], 0.01, atol=1.0e-12)
    assert len(result["components"]) == 7


def test_source_authorization_precedes_any_target_dataset_construction() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    authorization = source.index("products = validate_authorization(")
    target_construction = source.index("primary_datasets = build_datasets(")
    assert authorization < target_construction
    assert '"source_maps_frozen_before_target_opened": True' in source
    assert '"lensing_target_opened": True' in source
