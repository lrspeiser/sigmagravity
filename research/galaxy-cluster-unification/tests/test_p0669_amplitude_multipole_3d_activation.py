from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0669_amplitude_multipole_3d_activation"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_amplitude_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_spent_RXJ2129_3D_map_build"] is True
    assert len(result["gate_results"]) == 17
    assert all(result["gate_results"].values())


def test_absolute_channel_and_domain_separation_pass():
    metrics = report()["metrics"]
    assert metrics["registered_galaxy_nominal_median_sigma"] < 5.3e-5
    assert metrics["registered_cluster_nominal_median_sigma"] > 0.003
    assert metrics["registered_cluster_to_galaxy_nominal_median_sigma_ratio"] > 58
    assert min(metrics["mass_sensitivity_cluster_to_galaxy_sigma_ratios"].values()) > 53
    assert metrics[
        "registered_cluster_to_galaxy_nominal_median_amplitude_gate_ratio"
    ] > 1.58


def test_lift_and_constitutive_bounds_pass():
    metrics = report()["metrics"]
    assert metrics["maximum_component_mass_relative_error"] < 2e-15
    assert metrics["minimum_constitutive_eigenvalue_conservative_bound"] > 8e-7


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0669_amplitude_multipole_3d_activation.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0669_amplitude_multipole_3d_activation.py"
    )
    assert result["activation_source_sha256"] == digest(
        ROOT / "src/voidscreen/amplitude_activation_3d.py"
    )
    assert result["spent_RXJ2129_lensing_outcomes_opened"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0669_amplitude_multipole_3d.png").stat().st_size > 20000
