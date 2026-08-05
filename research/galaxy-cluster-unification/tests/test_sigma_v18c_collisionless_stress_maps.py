from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v18c_collisionless_stress_maps.json"
REPORT = ROOT / "results" / "sigma_v18c_collisionless_stress_maps" / "report.json"
SCRIPT = ROOT / "scripts" / "build_sigma_v18c_collisionless_stress_maps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v18c_maps", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adaptive_bandwidth_has_no_cluster_scale_parameter() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    kernel = config["adaptive_kernel"]
    assert kernel["primary_neighbor_rank"] == 8
    assert kernel["declared_sensitivity_neighbor_ranks"] == [6, 8, 12]
    assert kernel["fixed_physical_length"] is False
    assert kernel["per_cluster_bandwidth"] is False
    assert config["authorization"]["formula_or_kernel_selection_from_target"] is False
    assert config["authorization"]["lensing_target_opened"] is False


def test_bandwidth_uses_declared_neighbor_rank() -> None:
    module = load_module()
    x = np.arange(10, dtype=float)
    y = np.zeros(10, dtype=float)
    bandwidth = module.adaptive_bandwidths(x, y, 2)
    assert np.all(bandwidth > 0.0)
    assert bandwidth[5] == 0.5
    assert bandwidth[0] == 1.0


def test_report_and_products_are_target_blind_and_finite() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "both_target_blind_collisionless_stress_maps_constructed_and_frozen"
    assert report["primary_neighbor_rank"] == 8
    assert report["fixed_physical_length"] is False
    assert report["per_cluster_bandwidth_amplitude_or_orientation"] is False
    assert report["inverse_coefficient_fit"] is False
    assert report["lensing_target_opened"] is False
    assert report["holdout_opened"] is False
    counts = {item["cluster"]: item["selected_members"] for item in report["clusters"]}
    assert counts == {"MACS0416": 231, "PLCKG287": 129}
    for item in report["clusters"]:
        path = ROOT / item["product"]
        assert digest(path) == item["product_sha256"]
        with np.load(path) as product:
            for rank in (6, 8, 12):
                q = product[f"q_member_k{rank}"]
                assert q.shape == (193, 193)
                assert np.all(np.isfinite(q))
                assert np.all(q >= 0.0)
                assert np.max(q) > 0.0
                assert np.all(product[f"bandwidth_kpc_k{rank}"] > 0.0)


def test_report_hashes_frozen_parents() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == digest(CONFIG)
    for key in ("readiness_config", "readiness_report"):
        assert report["input_hashes"][key] == digest(ROOT / config["parents"][key])
