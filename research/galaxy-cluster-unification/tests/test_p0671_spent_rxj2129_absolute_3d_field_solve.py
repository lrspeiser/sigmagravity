from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0671_spent_rxj2129_absolute_3d_field_solve"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_field_solve_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_spent_raw_lens_topology_audit"] is True
    assert len(result["gate_results"]) == 22
    assert all(result["gate_results"].values())


def test_scalar_and_tensor_fields_converge_with_nonzero_stable_difference():
    metrics = report()["metrics"]
    assert metrics["scalar_normalized_residual_RMS"] < 9e-6
    assert metrics["tensor_normalized_residual_RMS"] < 9e-6
    assert metrics["boundary_maximum_relative_mismatch"] == 0.0
    assert metrics["minimum_tensor_constitutive_eigenvalue"] > 0.06
    assert metrics["scalar_strong_lens_median_physical_deflection_arcsec"] > 3.06
    assert 0.998 < metrics["tensor_to_scalar_strong_lens_deflection_RMS_ratio"] < 1.0
    assert 0.0016 < metrics["tensor_minus_scalar_strong_lens_relative_RMS"] < 0.0018


def test_zero_slip_deflections_are_numerically_curl_free():
    metrics = report()["metrics"]
    assert metrics["scalar_normalized_deflection_curl_RMS"] < 5e-16
    assert metrics["tensor_normalized_deflection_curl_RMS"] < 5e-16


def test_sources_fields_and_no_lens_score_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0671_spent_rxj2129_absolute_3d_field_solve.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0671_spent_rxj2129_absolute_3d_field_solve.py"
    )
    field_path = RESULTS / "rxj2129_absolute_scalar_tensor_fields.npz"
    assert result["field_sha256"] == digest(field_path)
    with np.load(field_path) as data:
        for key in (
            "scalar_alpha_x_physical_arcsec",
            "scalar_alpha_y_physical_arcsec",
            "tensor_alpha_x_physical_arcsec",
            "tensor_alpha_y_physical_arcsec",
        ):
            assert data[key].shape == (33, 33)
            assert np.all(np.isfinite(data[key]))
    assert result["raw_lens_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0671_absolute_deflection_fields.png").stat().st_size > 70000
