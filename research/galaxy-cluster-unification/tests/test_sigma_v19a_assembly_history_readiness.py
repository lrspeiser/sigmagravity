from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19a_assembly_history_readiness.json"
REPORT = ROOT / "results" / "sigma_v19a_assembly_history_readiness" / "report.json"
SCRIPT = ROOT / "scripts" / "audit_sigma_v19a_assembly_history_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19a_readiness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dressler_shectman_is_large_for_separated_velocity_groups() -> None:
    module = load_module()
    x = np.r_[np.linspace(-100.0, -10.0, 10), np.linspace(10.0, 100.0, 10)]
    y = np.zeros_like(x)
    grouped = np.r_[np.full(10, -500.0), np.full(10, 500.0)]
    mixed = np.resize(np.asarray([-500.0, 500.0]), 20)
    grouped_delta, _ = module.dressler_shectman(x, y, grouped, 4)
    mixed_delta, _ = module.dressler_shectman(x, y, mixed, 4)
    assert grouped_delta > 2.0 * mixed_delta


def test_velocity_gradient_recovers_declared_direction() -> None:
    module = load_module()
    x = np.asarray([-100.0, 0.0, 100.0, -100.0, 0.0, 100.0])
    y = np.asarray([-50.0, -50.0, -50.0, 50.0, 50.0, 50.0])
    velocity = 2.0 * x - y
    result = module.velocity_gradient(x, y, velocity)
    assert math.isclose(result["east_km_s_per_mpc"], 2000.0, rel_tol=1e-12)
    assert math.isclose(result["north_km_s_per_mpc"], -1000.0, rel_tol=1e-12)
    assert math.isclose(result["r_squared"], 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_protocol_forbids_target_and_formula_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["authorization"]["formula_selection_authorized"] is False
    assert config["authorization"]["lensing_or_halo_input_authorized"] is False
    assert config["authorization"]["new_lensing_target_access_authorized"] is False
    assert config["gates"]["instantaneous_phase_space_controls_may_satisfy_history_gate"] is False


def test_report_separates_snapshot_diagnostics_from_history() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    controls = report["instantaneous_phase_space_controls"]
    assert controls["MACS0416"]["selected_members"] == 231
    assert controls["PLCKG287"]["selected_members"] == 129
    assert all(not row["is_genuinely_time_ordered"] for row in controls.values())
    assert report["instantaneous_controls_count_as_history"] is False
    assert report["gravity_parameters_fit"] == 0


def test_report_closes_readiness_without_claiming_physics_falsification() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    gates = report["gate_results"]
    assert gates["member_phase_space_controls_available"] is True
    assert gates["one_common_time_ordered_observable"] is False
    assert gates["unique_causal_origin_both_clusters"] is False
    assert gates["five_sigma_history_statistic_both_clusters"] is False
    assert gates["projection_uncertainty_ensemble_both_clusters"] is False
    assert gates["member_and_temperature_uncertainty_products_both_clusters"] is False
    assert gates["history_source_construction_authorized"] is False
    assert "not a falsification" in report["failure_classification"]
    assert report["new_lensing_target_opened"] is False
    assert report["holdout_opened"] is False


def test_report_hashes_every_frozen_parent_and_member_catalog() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == digest(CONFIG)
    for key in ("observability_gate", "member_readiness_config", "member_readiness_report"):
        assert report["input_hashes"][key] == digest(ROOT / config["parents"][key])
    parent = json.loads(
        (ROOT / config["parents"]["member_readiness_config"]).read_text(encoding="utf-8")
    )
    for name, cluster in parent["clusters"].items():
        assert report["input_hashes"][f"{name}_member_catalog"] == digest(
            ROOT / cluster["member_catalog"]
        )
