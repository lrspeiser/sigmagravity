from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0670_spent_rxj2129_absolute_3d_map_build"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_frozen_map_build_gate_passes():
    result = report()
    assert result["status"] == "pass"
    assert result["all_progression_gates_pass"] is True
    assert result["candidate_advanced_to_spent_scalar_tensor_field_solve"] is True
    assert len(result["gate_results"]) == 20
    assert all(result["gate_results"].values())


def test_absolute_source_mass_and_activation_are_physical():
    metrics = report()["metrics"]
    assert 7.88e12 < metrics["target_baryon_mass_msun"] < 7.90e12
    assert metrics["maximum_component_surface_mass_relative_error"] < 3e-16
    assert metrics["maximum_component_volume_mass_relative_error"] < 9e-16
    assert metrics["multipole_amplitude_gate"] > 0.128
    assert metrics["mass_weighted_sigma"] > 0.0048
    assert metrics["minimum_constitutive_eigenvalue_proxy"] > 0.017
    assert metrics["strong_lens_maximum_radius_grid_cells"] > 5.4


def test_stored_cube_contains_complete_field_inputs():
    map_path = RESULTS / "rxj2129_absolute_baryons_3d.npz"
    with np.load(map_path) as data:
        assert data["axis_kpc"].shape == (33,)
        for key in (
            "stellar_volume_density_kg_m3",
            "gas_volume_density_kg_m3",
            "sigma",
            "transport_direction_x",
            "transport_direction_y",
            "transport_direction_z",
            "simple_mond_boundary_m2_s2",
        ):
            assert data[key].shape == (33, 33, 33)
            assert np.all(np.isfinite(data[key]))
        assert float(np.min(data["sigma"])) >= 0.0
        assert float(np.max(data["sigma"])) < 1.0


def test_sources_and_no_lens_score_are_preserved():
    result = report()
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0670_spent_rxj2129_absolute_3d_map_build.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0670_spent_rxj2129_absolute_3d_map_build.py"
    )
    map_path = RESULTS / "rxj2129_absolute_baryons_3d.npz"
    assert result["map_sha256"] == digest(map_path)
    assert result["raw_lens_score_computed"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert (RESULTS / "p0670_absolute_3d_map.png").stat().st_size > 70000
