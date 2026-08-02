from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0673_compound_path_activation"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_compound_activation_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_new_spent_field_solve"] is True
    assert len(result["gate_results"]) == 18
    assert all(result["gate_results"].values())


def test_registered_domains_separate_without_object_parameters():
    result = report()
    metrics = result["metrics"]
    assert metrics["registered_galaxy_nominal_median_sigma"] < 0.001
    assert metrics["registered_cluster_nominal_median_sigma"] > 0.5
    assert metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] > 10000
    assert min(metrics["mass_sensitivity_cluster_to_galaxy_sigma_ratios"].values()) > 9000
    scores = pd.read_csv(RESULTS / "registered_compound_scores.csv")
    assert scores[scores.domain.eq("registered_galaxy_baryons_only")].case.nunique() == 13
    assert scores[scores.domain.eq("registered_cluster_baryons_only")].case.nunique() == 4


def test_radial_null_spent_activation_and_positive_bound_are_preserved():
    metrics = report()["metrics"]
    assert metrics["radial_cocentered_synthetic_mass_weighted_sigma"] < 1e-8
    assert metrics["spent_RXJ2129_mass_weighted_sigma"] > 0.18
    assert metrics["maximum_sigma"] <= 0.999999
    assert metrics["minimum_constitutive_eigenvalue_proxy"] > 0.0


def test_sources_and_no_lens_score_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0673_compound_path_activation.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0673_compound_path_activation.py"
    )
    assert result["activation_source_sha256"] == digest(
        ROOT / "src/voidscreen/compound_activation_3d.py"
    )
    assert result["new_raw_lens_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0673_compound_path_activation.png").stat().st_size > 70000
