import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_sigma_v19cy_a2319_relative_fe_line_shift as line_shift


def test_frozen_relative_line_scope_and_seals() -> None:
    config, parent = line_shift.validate_inputs()
    assert config["protocol_version"].endswith("1.0.1")
    assert "cannot convert the failed result into a pass" in config[
        "post_execution_gate_correction"
    ]
    assert parent["terminal_gate_passed"]
    assert [item["name"] for item in config["regions"]] == [
        "a",
        "b",
        "d",
        "b_prime",
        "c_prime",
        "d_prime",
        "e_prime",
    ]
    assert config["energy_column"] == "EPI2"
    assert config["uncertainty"]["draws"] == 150
    assert not config["empirical_model"]["published_values_enter_optimizer"]
    assert not config["authorization"]["fit_bapec_or_claim_absolute_velocity"]
    assert not config["authorization"]["access_validation_or_holdout_assets"]


def test_poisson_deviance_is_zero_at_identity() -> None:
    values = np.array([0.0, 1.0, 4.0, 10.0])
    assert abs(line_shift.poisson_deviance(values, values + (values == 0) * 1e-12)) < 1e-9


def test_manufactured_line_shifts_recover_ordering() -> None:
    centers = np.arange(6200.5, 6700.0, 1.0)
    base = (
        250.0 * np.exp(-0.5 * ((centers - 6352.0) / 3.0) ** 2)
        + 80.0 * np.exp(-0.5 * ((centers - 6603.0) / 3.5) ** 2)
        + 8.0
    )
    histograms = {}
    for name, shift in (("blue", 3.0), ("middle", 0.0), ("red", -3.0)):
        histograms[name] = np.interp(centers - shift, centers, base, left=8.0, right=8.0)
    config, _ = line_shift.validate_inputs()
    fits = line_shift.fit_dataset(
        histograms,
        centers,
        config["windows"]["primary_fe_k"],
        config["empirical_model"],
    )
    velocities = [
        fits[name]["velocity_relative_unweighted_mean_kms"]
        for name in ("blue", "middle", "red")
    ]
    assert velocities[0] < velocities[1] < velocities[2]
    assert all(fits[name]["inside_shift_bounds"] for name in fits)


def test_published_benchmark_never_enters_optimizer() -> None:
    source = inspect.getsource(line_shift.fit_one_shift)
    assert "published" not in source.lower()
    assert "benchmark" not in source.lower()
    comparison_source = inspect.getsource(line_shift.comparison_metrics)
    assert "benchmark" in comparison_source
