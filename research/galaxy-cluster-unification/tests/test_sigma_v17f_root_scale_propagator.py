import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17f_root_scale_propagator.json"
RUNNER = ROOT / "scripts" / "run_sigma_v17f_root_scale_propagator.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17f_root_scale", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v17f_freeze_is_conditional_and_has_one_length() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17F-ROOT-SCALE-PROPAGATOR-1.0.0"
    assert config["authorization"]["required_v17e_gate"] == "gate_results.advance=true"
    assert config["fit"]["thermal_parameters_per_direction"] == 1
    assert config["fit"]["per_cluster_gravity_parameters"] == 0
    assert config["fit"]["lensing_only_multiplier"] is False
    assert config["propagation"]["object_specific_length_allowed"] is False
    assert config["projected_root_equation"]["source_families"] == [
        "q_total",
        "q_contrast",
    ]
    assert min(config["propagation"]["L_sigma_kpc_grid"]) == 0.0
    assert config["integrity"]["v17e_result_existed_at_freeze"] is False


def test_every_v17f_parent_hash_is_current() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("thermal_source_protocol", "thermal_source_protocol_sha256"),
        ("thermal_transfer_protocol", "thermal_transfer_protocol_sha256"),
        ("static_baseline_protocol", "static_baseline_protocol_sha256"),
    ):
        path = ROOT / config["parents"][path_key]
        assert path.is_file()
        assert config["parents"][hash_key] == _sha256(path)


def test_zero_length_is_exact_source_only_limit() -> None:
    runner = _load_runner()
    rng = np.random.default_rng(20260804)
    source = rng.normal(size=(17, 17))

    propagated = runner.helmholtz_propagate(source, 2.0, 0.0, 2)

    assert np.array_equal(propagated, source)
    assert propagated is not source


def test_positive_length_broadens_a_compact_source_without_sign_flip() -> None:
    runner = _load_runner()
    source = np.zeros((65, 65), dtype=float)
    center = source.shape[0] // 2
    source[center, center] = 1.0

    propagated = runner.helmholtz_propagate(source, 1.0, 4.0, 2)

    assert 0.0 < propagated[center, center] < 1.0
    assert propagated[center, center + 1] > 0.0
    assert np.min(propagated) > -1e-12
    assert abs(float(np.sum(propagated)) - 1.0) < 1e-3


def test_authorization_fails_closed_without_upstream_reports(tmp_path: Path) -> None:
    runner = _load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="upstream report is absent"):
        runner.validate_authorization(
            CONFIG,
            config,
            tmp_path / "missing_v17e.json",
            tmp_path / "missing_thermal.json",
        )
