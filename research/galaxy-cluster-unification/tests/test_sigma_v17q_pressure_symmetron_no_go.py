import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17q_pressure_symmetron_no_go.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17q_pressure_symmetron_no_go.py"
REPORT = ROOT / "results" / "sigma_v17q_pressure_symmetron_no_go" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17q_symmetron", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17q_is_hash_locked_and_does_not_open_holdouts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = config["parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["protocol"])
    assert parent["report_sha256"] == _sha256(ROOT / parent["report"])
    for entry in config["inputs"].values():
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    assert config["authorization"]["untouched_holdout_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False


def test_model_s_parser_recovers_physical_center_and_surface() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    solar = runner.load_model_s(
        ROOT / config["inputs"]["solar_model_s"]["path"],
        config["physical_constants"],
    )

    assert solar.radius_fraction.iloc[0] == pytest.approx(0.0)
    assert solar.radius_fraction.iloc[-1] <= 1.0
    assert solar.pressure_pa.iloc[0] > solar.pressure_pa.iloc[-1]
    assert runner.solar_pressure_compactness(solar, config["physical_constants"]) > 0.0


def test_pressure_column_ordering_reverses_desired_selectivity() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cluster = report["cluster_spent_diagnostic"]
    theorem = report["pressure_column_no_go"]

    assert cluster["points"] == cluster["points_requiring_order_unity_extra"]
    assert cluster["minimum_cluster_to_solar_column_ratio"] > 1.0
    assert cluster["median_cluster_to_solar_column_ratio"] > 20.0
    assert theorem["all_required_cluster_columns_at_least_solar"] is True
    assert theorem["gamma_lower_bound"] > theorem["cassini_limit"]
    assert theorem["excess_factor"] > 3000.0


def test_optimistic_model_s_profile_barely_screens_the_sun() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    control = report["optimistic_model_s_control"]

    assert control["numerics_gate_pass"] is True
    assert control["screening_fraction"] > 0.9
    assert control["required_screening_fraction"] < 4e-4
    assert control["gamma_excess_factor"] > 3000.0
    assert control["cassini_gate_pass"] is False


def test_v17q_triggers_the_declared_direct_pressure_mechanism_reset() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    selection = report["selection"]

    assert report["status"] == "completed_pressure_symmetron_no_go_audit"
    assert report["holdout_opened"] is False
    assert selection["outcome"] == (
        "retire_standard_pressure_symmetron_and_reset_direct_pressure_metric"
    )
    assert len(selection["same_solar_cluster_gate_failures"]) == 3
    assert selection["mechanism_reset_triggered"] is True
    assert any("not every possible" in item for item in report["claim_boundary"])
