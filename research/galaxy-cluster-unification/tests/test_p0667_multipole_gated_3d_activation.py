from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0667_multipole_gated_3d_activation"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_multipole_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_registered_map_audit"] is True
    assert len(result["gate_results"]) == 18
    assert all(result["gate_results"].values())


def test_radial_artifact_is_removed_and_offset_signal_retained():
    metrics = report()["metrics"]
    assert metrics["radial_multipole_gate"] < 2e-32
    assert metrics["radial_sigma_mass_weighted_mean"] < 4e-37
    assert metrics["offset_multipole_gate"] > 0.34
    assert metrics["offset_sigma_mass_weighted_mean"] > 0.023
    assert metrics["offset_signal_retained_fraction_vs_P0666"] > 0.34


def test_all_geometric_symmetries_pass():
    metrics = report()["metrics"]
    assert metrics["rotation_covariance_relative_error"] < 3e-16
    assert metrics["component_exchange_relative_error"] == 0.0
    assert metrics["translation_covariance_relative_error"] < 0.0014
    assert metrics["scale_covariance_multipole_gate_relative_error"] < 1e-12
    assert metrics["direction_reversal_tensor_relative_error"] == 0.0


def test_sources_blindness_and_figure_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0667_multipole_gated_3d_activation.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0667_multipole_gated_3d_activation.py"
    )
    assert result["activation_source_sha256"] == digest(
        ROOT / "src/voidscreen/multipole_activation_3d.py"
    )
    assert result["spent_RXJ2129_lensing_outcomes_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0667_multipole_gated_activation.png").stat().st_size > 20000
