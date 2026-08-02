from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0676_spent_rxj2129_transverse_confinement_field"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_confinement_field_fails_only_strength_gates():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_spent_raw_lens_topology_audit"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {"response_ratio_lower", "confinement_change_nonperturbative"}


def test_confinement_is_converged_positive_and_has_the_right_sign():
    metrics = report()["metrics"]
    assert metrics["confinement_normalized_residual_RMS"] < 1e-5
    assert metrics["minimum_confinement_constitutive_eigenvalue"] > 0.0
    assert 1.15 < metrics["confinement_to_scalar_strong_lens_deflection_RMS_ratio"] < 1.2
    assert 0.16 < metrics["confinement_minus_scalar_strong_lens_relative_RMS"] < 0.2
    assert metrics["confinement_normalized_deflection_curl_RMS"] < 1e-8


def test_sources_fields_and_seals_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0676_spent_rxj2129_transverse_confinement_field.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0676_spent_rxj2129_transverse_confinement_field.py"
    )
    field_path = RESULTS / "rxj2129_absolute_scalar_confinement_fields.npz"
    assert result["field_sha256"] == digest(field_path)
    with np.load(field_path) as data:
        assert data["confinement_alpha_x_physical_arcsec"].shape == (33, 33)
        assert np.all(np.isfinite(data["confinement_alpha_y_physical_arcsec"]))
    assert result["raw_lens_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
