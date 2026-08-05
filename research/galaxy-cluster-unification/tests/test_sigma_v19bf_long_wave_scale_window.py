import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bf_long_wave_scale_window.json"
SCRIPT = ROOT / "scripts" / "check_sigma_v19bf_long_wave_scale_window.py"
REPORT = ROOT / "results" / "sigma_v19bf_long_wave_scale_window" / "report.json"
GRID = ROOT / "results" / "sigma_v19bf_long_wave_scale_window" / "scale_grid.csv"
PLOT = ROOT / "results" / "sigma_v19bf_long_wave_scale_window" / "activation_window.svg"
SPEC = importlib.util.spec_from_file_location("sigma_v19bf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_tiny_baseline_activation_is_stable_and_quadratic():
    x = 1.0e-8
    actual = MODULE.low_pass_activation(x, 1.0)
    assert actual > 0.0
    assert math.isclose(actual, 0.5 * x * x, rel_tol=1.0e-8)


def test_inverse_activation_recovers_declared_targets():
    for target in (1.0e-12, 0.05, 0.45, 0.50, 0.55, 0.90):
        x = MODULE.x_at_activation(target)
        assert math.isclose(MODULE.low_pass_activation(x, 1.0), target, rel_tol=2.0e-12, abs_tol=1.0e-15)


def test_derived_scale_window_is_nonempty_and_contains_six_kpc():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bounds = MODULE.constraint_bounds(config)
    assert bounds["nonempty"]
    assert bounds["effective_lower_kpc"] < 6.0 < bounds["effective_upper_kpc"]
    assert bounds["active_lower_constraint"] == "transition_maximum"
    assert bounds["active_upper_constraint"] == "transition_minimum"


def test_protocol_forbids_action_constant_and_payload_selection():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    assert not authorization["read_v19w_or_v19x_gas_result"]
    assert not authorization["select_candidate_action"]
    assert not authorization["select_universal_length_or_amplitude"]
    assert not authorization["read_lensing_or_halo_payload"]
    assert not authorization["open_holdout"]
    assert not authorization["change_gravity_physics"]


def test_frozen_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["implementation"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["implementation"]["runner_sha256"]


def test_completed_report_and_artifacts_pass_only_the_scale_gate():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == "passed_dimensionless_scale_window"
    assert all(report["gate_results"].values())
    assert report["theory_state"]["dimensionless_scale_window_derived"]
    assert not report["theory_state"]["covariant_action_selected"]
    assert not report["theory_state"]["universal_constants_selected"]
    assert not report["theory_state"]["weak_field_metric_derived"]
    assert GRID.is_file()
    assert PLOT.is_file()
    assert len(GRID.read_text(encoding="utf-8").splitlines()) == 1002
    svg = PLOT.read_text(encoding="utf-8")
    ET.fromstring(svg)
    assert "nan" not in svg.lower()
    assert "Long-wave activation across physical baselines" in svg
    assert MODULE.sha256(GRID) == report["outputs"]["grid_sha256"]
    assert MODULE.sha256(PLOT) == report["outputs"]["plot_sha256"]


def test_report_is_byte_reproducible():
    first = MODULE.run(CONFIG)
    first_bytes = REPORT.read_bytes()
    first_grid = GRID.read_bytes()
    first_plot = PLOT.read_bytes()
    second = MODULE.run(CONFIG)
    assert first == second
    assert REPORT.read_bytes() == first_bytes
    assert GRID.read_bytes() == first_grid
    assert PLOT.read_bytes() == first_plot
