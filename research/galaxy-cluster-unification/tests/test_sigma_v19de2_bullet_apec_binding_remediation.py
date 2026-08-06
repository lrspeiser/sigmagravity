from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19de2_bullet_apec_binding_remediation.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19de2_bullet_apec_binding_remediation.py"
RESULT = ROOT / "results" / "sigma_v19de2_bullet_apec_binding_remediation" / "report.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19de2_remediation", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_remediation_cannot_change_scientific_method_or_open_new_data() -> None:
    payload = config()
    assert not any(key in payload for key in ("model", "optimization", "profile", "gates", "data"))
    auth = payload["authorization"]
    assert auth["bind_and_probe_xspec_model_data_without_source"] is True
    assert auth["rerun_unchanged_v19de_integrated_profile_after_committed_preflight"] is True
    assert auth["change_v19de_scientific_method_or_gate"] is False
    assert auth["open_any_regional_source_line_or_velocity"] is False
    assert auth["open_obsid554_or_abell2146"] is False
    assert auth["open_lensing_halo_gravity_or_action"] is False


def test_scientific_method_and_invalid_execution_parents_are_exact() -> None:
    payload = config()["parents"]
    base = ROOT / payload["v19de_config"]["path"]
    invalid = ROOT / payload["v19de_invalid_report"]["path"]
    runner = load_runner()
    assert runner.v19de.sha256(base) == payload["v19de_config"]["sha256"]
    assert runner.v19de.sha256(invalid) == payload["v19de_invalid_report"]["sha256"]
    assert json.loads(invalid.read_text(encoding="utf-8"))["status"] == payload["v19de_invalid_report"]["required_status"]


class PositiveComponent:
    name = "positive"
    type = "synthetic"

    def __call__(self, low, high):
        assert len(low) == len(high)
        return np.asarray(high) - np.asarray(low)


class InvalidComponent:
    name = "invalid"
    type = "synthetic"

    def __call__(self, low, high):
        return np.zeros(len(low))


def test_positive_probe_accepts_flux_and_rejects_zero_model() -> None:
    runner = load_runner()
    bins = [[2.0, 2.25], [2.25, 2.5]]
    result = runner.probe_component(PositiveComponent(), bins)
    assert result["integrated_flux"] == 0.5
    with pytest.raises(RuntimeError, match="not finite and positive"):
        runner.probe_component(InvalidComponent(), bins)


def test_model_data_contract_freezes_atomdb_pair() -> None:
    data = config()["xspec_model_data"]
    assert data["atomdb_version"] == "3.0.9"
    assert data["apec_root"].endswith("apec_v3.0.9")
    assert data["apec_continuum"]["path"] == data["apec_root"] + "_coco.fits"
    assert data["apec_lines"]["path"] == data["apec_root"] + "_line.fits"
    assert set(data["required_models"]) == {"xsapec", "xsmekal"}


def test_terminal_profile_is_complete_but_fails_the_frozen_secondary_minimum_gate() -> None:
    payload = config()
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    method = json.loads((ROOT / payload["parents"]["v19de_config"]["path"]).read_text(encoding="utf-8"))

    assert report["base_v19de_status"] == "bullet_integrated_redshift_profile_gate_failed"
    assert report["status"] == "bullet_integrated_profile_remediation_scientific_gate_failed"
    assert report["config_sha256"] == "eb9ba22888f0dff3b696834613caa890b2958820fd664876b13f8df13fbd1dcc"
    assert report["runner_sha256"] == "57b76be67f1d7325ec2eda78a3f86dee461a49917d5747a5d1ed214944216c1d"

    expected_gate_values = {
        "both_model_profiles_complete": True,
        "every_profile_point_has_a_finite_multistart_fit": True,
        "best_redshift_and_delta1_interval_interior": True,
        "no_distinct_secondary_minimum_within_delta_6p63": False,
        "each_best_redshift_within_0p01_of_optical": True,
        "apec_mekal_best_redshift_difference_at_most_0p003": True,
        "integrated_gain_covariance_finite_psd": True,
    }
    assert report["gates"] == expected_gate_values

    for branch in ("apec", "mekal"):
        coarse = report["profiles"][branch]["coarse"]
        fine = report["profiles"][branch]["fine"]
        assert len(coarse) == len(fine) == 101
        assert all(row["finite"] for row in coarse + fine)
        assert all(len(row["attempts"]) == 2 for row in coarse + fine)
        assert report["summaries"][branch]["profile_points"] == 202
        assert report["summaries"][branch]["finite_profile_points"] == 202

    apec_coarse = sorted(report["profiles"]["apec"]["coarse"], key=lambda row: row["redshift"])
    apec_best_z = report["summaries"]["apec"]["best_redshift"]
    coarse_best_stat = min(row["statistic"] for row in apec_coarse)
    secondary = []
    for before, row, after in zip(apec_coarse, apec_coarse[1:], apec_coarse[2:], strict=False):
        if (
            row["statistic"] <= before["statistic"]
            and row["statistic"] < after["statistic"]
            and abs(row["redshift"] - apec_best_z) >= method["profile"]["distinct_minimum_separation"]
        ):
            secondary.append(
                {
                    "redshift": row["redshift"],
                    "delta_statistic": row["statistic"] - coarse_best_stat,
                }
            )
    assert len(secondary) == 1
    assert secondary[0]["redshift"] == 0.305
    assert np.isclose(secondary[0]["delta_statistic"], 1.7801685437152628)
    assert secondary[0]["delta_statistic"] < method["profile"]["secondary_minimum_delta"]
    assert report["summaries"]["apec"]["secondary_minima"] == secondary
    assert report["summaries"]["mekal"]["secondary_minima"] == []

    apec_z = report["summaries"]["apec"]["best_redshift"]
    mekal_z = report["summaries"]["mekal"]["best_redshift"]
    assert np.isclose(apec_z, 0.3008)
    assert np.isclose(mekal_z, 0.2999)
    assert abs(apec_z - mekal_z) <= method["gates"]["apec_mekal_best_redshift_difference_at_most"]
    assert report["gain"]["finite_symmetric_positive_semidefinite"] is True

    frozen_model_data = payload["xspec_model_data"]
    runtime_model_data = report["model_data_binding"]
    assert runtime_model_data["atomdb_version"] == frozen_model_data["atomdb_version"]
    assert runtime_model_data["apec_continuum"]["sha256"] == frozen_model_data["apec_continuum"]["sha256"]
    assert runtime_model_data["apec_lines"]["sha256"] == frozen_model_data["apec_lines"]["sha256"]
    assert all(item["integrated_flux"] > 0 for item in runtime_model_data["positive_model_probes"])
    assert set(report["configured_component_probes"]) == {"apec", "mekal"}

    assert report["integrated_systematic_and_goodness_stage_authorized"] is False
    assert report["posterior_predictive_or_thermal_sobol_run"] is False
    assert report["regional_source_line_or_velocity_opened"] is False
    assert report["obsid554_or_abell2146_opened"] is False
    assert report["lensing_halo_gravity_or_action_opened"] is False
