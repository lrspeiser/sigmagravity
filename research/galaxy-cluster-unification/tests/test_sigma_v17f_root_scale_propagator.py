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

    assert config["protocol_version"] == "SIGMA-V17F-ROOT-SCALE-PROPAGATOR-1.0.1"
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
    assert config["conditional_action_lift"]["action_is_complete_one_metric_theory"] is False


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


def test_propagated_source_builds_registered_one_metric_triplet(tmp_path: Path) -> None:
    runner = _load_runner()
    axis = np.linspace(-8.0, 8.0, 17)
    source = np.zeros((17, 17), dtype=float)
    source[8, 8] = 1.0
    product = tmp_path / "thermal.npz"
    np.savez_compressed(
        product,
        source_axis_kpc=axis,
        q_total=source,
        q_contrast=-source,
    )

    feature = runner.propagated_feature(product, "q_total", 2.0, axis, 2)

    assert feature.family == "scalar_scale"
    assert feature.convergence.shape == source.shape
    assert feature.shear_1.shape == source.shape
    assert feature.shear_2.shape == source.shape
    assert np.isfinite(feature.convergence).all()


def test_cross_transfer_recovers_shared_one_metric_amplitude() -> None:
    runner = _load_runner()
    axis = np.linspace(-4.0, 4.0, 9)
    east, north = np.meshgrid(axis, axis)
    zeros = np.zeros_like(east)
    mask = np.ones_like(east, dtype=bool)
    beta = 2.75

    def make_dataset(name: str, shift: float):
        convergence = np.exp(-((east - shift) ** 2 + north**2) / 4.0)
        shear_1 = convergence * (east**2 - north**2) / 20.0
        shear_2 = convergence * east * north / 10.0
        feature = runner.MetricFeature(
            name="known_root",
            family="scalar_scale",
            convergence=convergence,
            shear_1=shear_1,
            shear_2=shear_2,
        )
        wrong = runner.MetricFeature(
            name="wrong_root",
            family="scalar_scale",
            convergence=np.roll(convergence, 3, axis=0),
            shear_1=np.roll(shear_1, 3, axis=0),
            shear_2=np.roll(shear_2, 3, axis=0),
        )
        return runner.EquivariantDataset(
            name=name,
            mask=mask,
            base=(zeros, zeros, zeros),
            target=(beta * convergence, beta * shear_1, beta * shear_2),
            features={"known_root": feature, "wrong_root": wrong},
        )

    datasets = [make_dataset("A", -0.5), make_dataset("B", 0.75)]
    adjusted = [datasets, datasets]
    known = runner.score_candidate(datasets, adjusted, "known_root", 4.0)
    wrong = runner.score_candidate(datasets, adjusted, "wrong_root", 4.0)

    assert known["symmetric_cross_cluster_full_field_NRMSE"] < 1e-12
    assert known["directional_beta_log10_difference_dex"] < 1e-12
    assert all(abs(row["beta_sigma"] - beta) < 1e-12 for row in known["directions"])
    assert wrong["symmetric_cross_cluster_full_field_NRMSE"] > 0.1


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
