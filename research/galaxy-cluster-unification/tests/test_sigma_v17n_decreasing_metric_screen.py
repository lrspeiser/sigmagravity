import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17n_decreasing_metric_screen_no_go.json"
RUNNER = ROOT / "scripts" / "audit_sigma_v17n_decreasing_metric_screen.py"
REPORT = ROOT / "results" / "sigma_v17n_decreasing_metric_screen_no_go" / "report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17n_no_go", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17n_is_target_blind_and_hash_locked() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["authorization"]["observational_data_opened"] is False
    assert config["authorization"]["empirical_fit_authorized"] is False
    assert config["authorization"]["holdout_authorized"] is False
    assert config["parent"]["sha256"] == _sha256(ROOT / config["parent"]["protocol"])
    assert config["parent"]["report_sha256"] == _sha256(ROOT / config["parent"]["report"])
    shared = config["shared_exact_matter_kernel"]
    assert shared["sha256"] == _sha256(ROOT / shared["path"])


def test_quartic_soft_start_only_moves_the_negative_derivative() -> None:
    runner = _load_runner()

    assert runner.quartic_soft_start(0.0)[1] == pytest.approx(0.0, abs=1e-15)
    assert runner.quartic_soft_start(1.0)[1] < 0.0
    assert runner.quartic_soft_start(10.0)[0] < runner.quartic_soft_start(1.0)[0]


def test_all_representative_screens_decrease_somewhere() -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    for declaration in config["representative_curves"]:
        _, derivative = runner.CURVES[declaration["id"]](declaration["audit_z"])
        assert derivative < 0.0


def test_report_retires_the_curve_class_not_the_halo_goal() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "decreasing_metric_screen_class_retired"
    assert report["observational_data_opened"] is False
    assert report["theorem"]["maximum_normalized_hessian_error"] <= 1e-5
    assert all(row["finite_positive_zero_surface"] for row in report["curve_rows"])
    assert report["gates"]["exact_transverse_hessian_identity_pass"] is True
    assert report["gates"]["decreasing_metric_screen_class_survives"] is False
    assert report["decision"]["outcome"] == (
        "retire_all_decreasing_acceleration_screens_inside_physical_metric"
    )
    assert report["decision"]["holdout_authorized"] is False
