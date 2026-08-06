from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19dd_bullet_gain_weight_transport.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19dd_bullet_gain_weight_transport.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19dd_weights", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_authorization_is_transport_only() -> None:
    auth = config()["authorization"]
    assert auth["combine_integrated_bullet_primary_spectrum"] is True
    assert auth["open_source_pha_header_and_arf_response"] is True
    assert auth["fit_temperature_abundance_redshift_or_velocity"] is False
    assert auth["open_obsid554_or_abell2146"] is False
    assert auth["open_lensing_halo_gravity_or_action"] is False


def test_payload_blind_plan_is_complete() -> None:
    runner = load_runner()
    payload = config()
    parents = runner.validate_frozen(payload)
    _, plan = runner.load_plan(payload, parents)
    assert len(plan) == 43
    cells = [cell["cell_name"] for region in plan for cell in region["cells"]]
    assert len(cells) == len(set(cells)) == 3483


def test_independent_gain_covariance_uses_squared_weights() -> None:
    runner = load_runner()
    weights = [
        {"obsid": 1, "normalized_weight": 0.25},
        {"obsid": 2, "normalized_weight": 0.75},
    ]
    covariance = [[4e-6, 0.0], [0.0, 9e-6]]
    gains = {
        1: {"gain": {"intercept_keV": 0.01, "slope": 0.99, "covariance_intercept_slope": covariance}},
        2: {"gain": {"intercept_keV": 0.03, "slope": 1.01, "covariance_intercept_slope": covariance}},
    }
    result = runner.effective_gain(7, weights, gains, 5.2)
    expected = (0.25**2 + 0.75**2) * np.asarray(covariance)
    assert np.allclose(result["covariance_intercept_slope"], expected)
    assert np.isclose(result["intercept_keV"], 0.025)
    assert np.isclose(result["slope"], 1.005)
    assert result["covariance_finite_symmetric_psd"] is True
    corrections = [0.01 + (0.99 - 1.0) * 5.2, 0.03 + (1.01 - 1.0) * 5.2]
    mean = 0.25 * corrections[0] + 0.75 * corrections[1]
    expected_dispersion = np.sqrt(
        0.25 * (corrections[0] - mean) ** 2 + 0.75 * (corrections[1] - mean) ** 2
    )
    assert np.isclose(result["correction_at_observed_fe_keV"], mean)
    assert np.isclose(result["weighted_rms_obsid_correction_dispersion_keV"], expected_dispersion)


def test_runner_contains_no_source_fit_engine() -> None:
    source = RUNNER.read_text(encoding="utf-8").lower()
    for forbidden in ("sherpa", "xspec", "apec", "mekal", "fit_spectrum(", "redshift_profile"):
        assert forbidden not in source
