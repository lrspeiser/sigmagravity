from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0678_spent_rxj2129_required_field_decomposition"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_required_field_integrity_fails_only_coarse_grid_curl():
    result = report()
    assert result["status"] == "fail"
    assert result["all_integrity_gates_pass"] is False
    assert result["candidate_formula_advanced"] is False
    failed = {name for name, passed in result["gate_results"].items() if not passed}
    assert failed == {"halo_curl"}


def test_required_field_is_dominantly_broad_radial_strength():
    metrics = report()["metrics"]
    assert metrics["compact_halo_to_scalar_RMS_ratio"] > 3.3
    assert metrics["target_to_scalar_RMS_ratio"] > 4.7
    assert metrics["halo_monopole_RMS_fraction"] > 0.98
    assert metrics["halo_angular_residual_RMS_fraction"] < 0.16
    assert metrics["halo_scalar_vector_alignment_cosine"] > 0.99
    assert metrics["scalar_critical_sign_change_cells"] == 0
    assert metrics["scalar_plus_halo_critical_sign_change_cells"] > 0
    assert metrics["scalar_plus_halo_plus_shear_critical_sign_change_cells"] > 0


def test_all_baryonic_predictors_and_radial_bins_are_reported():
    correlations = pd.read_csv(RESULTS / "baryonic_predictor_correlations.csv")
    radial = pd.read_csv(RESULTS / "radial_required_field.csv")
    assert correlations.predictor.nunique() == 9
    assert correlations.target.nunique() == 4
    primary = radial[radial.center.eq("baryonic_center")]
    assert len(primary) == 8
    assert primary.samples.gt(0).all()
    assert primary.halo_to_scalar_magnitude_ratio.between(3.2, 3.5).all()


def test_hashes_no_candidate_no_raw_and_seals_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0678_spent_rxj2129_required_field_decomposition.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0678_spent_rxj2129_required_field_decomposition.py"
    )
    field_path = RESULTS / "rxj2129_required_field_decomposition.npz"
    assert result["common_field_sha256"] == digest(field_path)
    assert result["new_candidate_formula_fit"] is False
    assert result["new_raw_image_root_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0678_required_field_decomposition.png").stat().st_size > 130000
