from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19de2_bullet_apec_binding_remediation.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19de2_bullet_apec_binding_remediation.py"


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
