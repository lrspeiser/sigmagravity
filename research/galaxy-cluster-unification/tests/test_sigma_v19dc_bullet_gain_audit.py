from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19dc_bullet_gain_audit.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19dc_bullet_gain_audit.py"
OUTPUT = ROOT / "results" / "sigma_v19dc_bullet_gain_audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19dc_gain", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_authorization_is_background_only() -> None:
    payload = config()
    auth = payload["authorization"]
    assert auth["open_original_bullet_background_pha_and_rmf_ebounds"] is True
    assert auth["open_bullet_source_pha"] is False
    assert auth["fit_temperature_abundance_redshift_or_velocity"] is False
    assert auth["open_obsid554_or_abell2146"] is False
    assert auth["open_lensing_halo_gravity_or_action"] is False


def test_plan_uses_every_primary_cell_once_without_payload_access() -> None:
    runner = load_runner()
    payload = config()
    parents = runner.validate_frozen(payload)
    plan = runner.build_plan(payload, parents["unified_product_index"])
    cells = [cell["cell_name"] for rows in plan.values() for cell in rows]
    assert len(plan) == 9
    assert len(cells) == len(set(cells)) == 3483
    assert set(plan) == set(payload["workload"]["obsids"])


def test_two_centroids_recover_known_linear_gain_and_covariance() -> None:
    runner = load_runner()
    intercept = 0.012
    slope = 0.998
    ni_ref, au_ref = 7.4782, 9.7133
    ni_recorded = (ni_ref - intercept) / slope
    au_recorded = (au_ref - intercept) / slope
    parameters, covariance = runner.gain_parameters(
        ni_recorded, au_recorded, 4e-6, 9e-6, ni_ref, au_ref
    )
    assert np.allclose(parameters, [intercept, slope], rtol=0.0, atol=1e-10)
    assert np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-14


def test_runner_never_opens_a_source_pha_or_fit_engine() -> None:
    source = RUNNER.read_text(encoding="utf-8").lower()
    assert '["source_pha_name"]' not in source
    assert '["source_pha"]' not in source
    for forbidden in ("sherpa", "xspec", "apec", "mekal", "fit_spectrum("):
        assert forbidden not in source


def test_terminal_gain_audit_is_current_and_passes() -> None:
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "bullet_per_obsid_gain_audit_passed"
    assert report["runner_sha256"] == sha256(RUNNER)
    assert report["config_sha256"] == sha256(CONFIG)
    assert report["bullet_source_pha_opened"] is False
    assert report["temperature_abundance_redshift_or_velocity_fitted"] is False
    assert all(report["gates"].values())
    assert len(report["obsids"]) == 9
    assert sum(item["cells"] for item in report["obsids"]) == 3483
    assert min(item["minimum_line_delta_cash"] for item in report["obsids"]) >= 25.0
    assert max(item["maximum_window_centroid_shift_keV"] for item in report["obsids"]) <= 0.015
    for item in report["obsids"]:
        covariance = np.asarray(item["gain"]["covariance_intercept_slope"], dtype=float)
        assert np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-14
    for key in ("background_input_manifest", "blank_sky_union_spectra"):
        artifact = report[key]
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert sha256(path) == artifact["sha256"]
