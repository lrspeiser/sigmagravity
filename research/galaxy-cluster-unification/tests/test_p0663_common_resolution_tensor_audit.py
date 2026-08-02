from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0663_common_resolution_tensor_audit"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_common_resolution_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_outcome_blind_real_map_field_solves"] is True
    assert len(result["gate_results"]) == 15
    assert all(result["gate_results"].values())


def test_primary_scores_are_unchanged_and_domain_lever_remains_large():
    result = report()
    metrics = result["metrics"]
    assert result["gate_results"]["primary_scores_unchanged"] is True
    assert metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] > 60.0
    assert metrics["minimum_mass_sensitivity_cluster_to_galaxy_ratio"] > 54.0


def test_common_resolution_and_mass_conservation_pass():
    metrics = report()["metrics"]
    assert metrics["maximum_common_resolution_mass_relative_error"] < 5e-16
    assert metrics["maximum_domain_median_common_resolution_change_fraction"] < 0.31
    summaries = {row["domain"]: row for row in report()["resolution_summary"]}
    assert summaries["registered_galaxy_baryons_only"][
        "median_absolute_fractional_change"
    ] < 0.31
    assert summaries["registered_cluster_baryons_only"][
        "median_absolute_fractional_change"
    ] < 0.12


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0663_common_resolution_tensor_audit.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0663_common_resolution_tensor_audit.py"
    )
    assert result["resampling_source_sha256"] == digest(
        ROOT / "src/voidscreen/observational_resampling.py"
    )
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0663_common_resolution_tensor.png").stat().st_size > 20000
