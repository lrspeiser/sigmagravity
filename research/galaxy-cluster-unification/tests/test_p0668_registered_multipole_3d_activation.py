from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0668_registered_multipole_3d_activation"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_result_fails_only_absolute_cluster_channel():
    result = report()
    assert result["status"] == "fail"
    assert result["all_progression_gates_pass"] is False
    assert result["candidate_advanced_to_spent_RXJ2129_3D_map_build"] is False
    failed = [name for name, passed in result["gate_results"].items() if not passed]
    assert failed == ["cluster_channel_present"]


def test_domain_separation_is_strong_and_mass_robust():
    metrics = report()["metrics"]
    assert metrics["registered_galaxy_nominal_median_sigma"] < 0.001
    assert metrics["registered_cluster_nominal_median_sigma"] < 0.001
    assert metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] > 150
    assert min(metrics["mass_sensitivity_cluster_to_galaxy_sigma_ratios"].values()) > 129
    assert metrics[
        "registered_cluster_to_galaxy_nominal_median_multipole_gate_ratio"
    ] > 2.5


def test_lift_is_conservative_and_constitutive_tensor_is_positive():
    metrics = report()["metrics"]
    assert metrics["maximum_component_mass_relative_error"] < 2e-15
    assert metrics["minimum_constitutive_eigenvalue_proxy"] > 0.0


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0668_registered_multipole_3d_activation.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0668_registered_multipole_3d_activation.py"
    )
    assert result["activation_source_sha256"] == digest(
        ROOT / "src/voidscreen/multipole_activation_3d.py"
    )
    assert result["spent_RXJ2129_lensing_outcomes_opened"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0668_registered_multipole_3d.png").stat().st_size > 20000
