from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19de_bullet_integrated_redshift_profile.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19de_bullet_integrated_redshift_profile.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19de_profile", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_authorization_is_integrated_commissioning_only() -> None:
    auth = config()["authorization"]
    assert auth["open_integrated_source_pha_and_response_after_committed_preflight"] is True
    assert auth["fit_integrated_temperature_abundance_redshift"] is True
    assert auth["run_posterior_predictive_or_thermal_sobol"] is False
    assert auth["open_any_regional_source_line_or_velocity"] is False
    assert auth["open_obsid554_or_abell2146"] is False
    assert auth["open_lensing_halo_gravity_or_action"] is False


def test_profile_grids_and_evaluation_order_are_exact() -> None:
    runner = load_runner()
    payload = config()["profile"]
    coarse = runner.inclusive_grid(payload["optical_redshift_center"], payload["half_range"], payload["coarse_step"])
    fine = runner.inclusive_grid(payload["optical_redshift_center"], payload["fine_half_width"], payload["fine_step"])
    assert len(coarse) == len(fine) == 101
    assert coarse[0] == 0.246 and coarse[-1] == 0.346
    assert fine[0] == 0.291 and fine[-1] == 0.301
    ordered = runner.evaluation_order(coarse, payload["optical_redshift_center"])
    assert ordered[0] == payload["optical_redshift_center"]
    assert sorted(ordered) == coarse


def test_component_exchange_is_canonicalized_without_changing_pairs() -> None:
    runner = load_runner()
    state = {"T1": 20.0, "T2": 5.0, "Z1": 0.2, "Z2": 0.8, "norm1": 0.01, "norm2": 0.03, "nH": 0.04}
    result = runner.canonical_state(state)
    assert result["T1"] == 5.0 and result["T2"] == 20.0
    assert result["Z1"] == 0.8 and result["Z2"] == 0.2
    assert result["norm1"] == 0.03 and result["norm2"] == 0.01
    assert result["nH"] == state["nH"]


def test_profile_interval_interpolation_and_secondary_minimum_detection() -> None:
    runner = load_runner()
    values = np.arange(0.290, 0.303, 0.001)
    rows = [{"redshift": float(value), "statistic": float(((value - 0.296) / 0.002) ** 2)} for value in values]
    lower = runner.profile_crossing(rows, 0.296, 1.0, "lower")
    upper = runner.profile_crossing(rows, 0.296, 1.0, "upper")
    assert np.isclose(lower, 0.294)
    assert np.isclose(upper, 0.298)
    assert runner.distinct_secondary_minima(rows, 0.296, 0.003) == []


def test_integrated_gain_transport_is_finite_and_normalized() -> None:
    runner = load_runner()
    payload = config()
    parents = {
        key: ROOT / value["path"]
        for key, value in payload["parents"].items()
        if key in {"v19dc_report", "v19dd_report"}
    }
    gain = runner.effective_integrated_gain(parents)
    assert np.isclose(sum(gain["weights_by_obsid"].values()), 1.0)
    assert gain["finite_symmetric_positive_semidefinite"] is True
    assert gain["one_sigma_equivalent_velocity_uncertainty_km_s"] > 0
    assert gain["weighted_rms_obsid_correction_dispersion_km_s"] > 0


def test_payload_blind_preflight_is_complete() -> None:
    runner = load_runner()
    payload = config()
    result = runner.preflight(payload)
    assert result["status"] == runner.PREFLIGHT_STATUS
    assert result["branches"] == ["apec", "mekal"]
    assert result["coarse_points_per_branch"] == 101
    assert result["fine_points_per_branch"] == 101
    assert result["multistarts_per_point"] == 2
    assert result["source_pha_response_scientific_arrays_opened"] is False


def test_runner_has_no_regional_or_gravity_engine() -> None:
    source = RUNNER.read_text(encoding="utf-8").lower()
    for forbidden in ("load_regional", "regional_source_path", "halo_map", "gravity_fit", "lens_model"):
        assert forbidden not in source
