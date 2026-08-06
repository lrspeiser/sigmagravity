from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19dd_bullet_gain_weight_transport.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19dd_bullet_gain_weight_transport.py"
RESULT = ROOT / "results" / "sigma_v19dd_bullet_gain_weight_transport" / "report.json"


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


def test_terminal_transport_result_is_complete() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["status"] == "bullet_integrated_spectrum_and_gain_weight_transport_passed"
    assert payload["bullet_source_redshift_fitter_authorized"] is True
    assert all(payload["gates"].values())
    assert payload["integrated_spectrum"]["cells"] == 3483
    assert payload["integrated_spectrum"]["combined_full_pha_source_counts"] == 674283
    assert payload["region_obsid_fe_weights"]["rows"] == 43 * 9
    assert payload["effective_gain_by_region"]["rows"] == 43
    assert payload["integrated_weight_equivalence"]["maximum_relative_difference"] <= 1e-6
    assert payload["source_line_temperature_abundance_redshift_or_velocity_fitted"] is False
    assert payload["obsid554_or_abell2146_opened"] is False
    assert payload["lensing_halo_gravity_or_action_opened"] is False
