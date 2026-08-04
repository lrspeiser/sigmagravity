from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.sigma_tidal_misalignment import (
    bounded_misalignment_gradients,
    bounded_misalignment_potential,
    spectral_tidal_misalignment,
    tidal_commutator,
)
from voidscreen.sigma_triaxial_memory import (
    centered_axis,
    gaussian_mixture_density,
    symmetric_trace_free,
)

ROOT = Path(__file__).resolve().parents[1]


def test_commuting_tensors_are_null_and_noncommuting_tensors_activate() -> None:
    local = np.diag([1.0, 0.0, -1.0])
    commuting = np.diag([0.2, 0.5, -0.7])
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    noncommuting = rotation @ commuting @ rotation.T
    assert bounded_misalignment_potential(local, commuting) == pytest.approx(0.0)
    assert bounded_misalignment_potential(local, noncommuting) > 1e-4
    assert tidal_commutator(local, noncommuting) == pytest.approx(
        -tidal_commutator(local, noncommuting).T
    )


def test_potential_is_rotation_invariant_and_bounded() -> None:
    local = symmetric_trace_free(np.array([[0.8, 0.2, -0.1], [0.2, -0.3, 0.4], [-0.1, 0.4, -0.5]]))
    memory = symmetric_trace_free(np.array([[0.1, -0.4, 0.3], [-0.4, 0.7, 0.2], [0.3, 0.2, -0.8]]))
    rotation, _ = np.linalg.qr(np.array([[0.7, -0.3, 0.2], [0.4, 0.8, -0.1], [0.5, 0.1, 0.9]]))
    value = bounded_misalignment_potential(local, memory, screen=0.7)
    rotated = bounded_misalignment_potential(
        rotation @ local @ rotation.T,
        rotation @ memory @ rotation.T,
        screen=0.7,
    )
    assert rotated == pytest.approx(value, rel=1e-12)
    assert 0.0 <= value < 1.0


def test_analytic_gradients_match_stf_directional_differences() -> None:
    local = symmetric_trace_free(
        np.array([[0.8, 0.1, -0.2], [0.1, -0.25, 0.3], [-0.2, 0.3, -0.55]])
    )
    memory = symmetric_trace_free(np.array([[0.2, -0.3, 0.1], [-0.3, 0.6, 0.2], [0.1, 0.2, -0.8]]))
    direction = symmetric_trace_free(
        np.array([[0.1, 0.3, -0.4], [0.3, -0.5, 0.2], [-0.4, 0.2, 0.4]])
    )
    direction /= np.linalg.norm(direction)
    gradient_local, gradient_memory = bounded_misalignment_gradients(local, memory, screen=0.8)
    step = 1e-6
    finite_local = (
        bounded_misalignment_potential(local + step * direction, memory, screen=0.8)
        - bounded_misalignment_potential(local - step * direction, memory, screen=0.8)
    ) / (2.0 * step)
    finite_memory = (
        bounded_misalignment_potential(local, memory + step * direction, screen=0.8)
        - bounded_misalignment_potential(local, memory - step * direction, screen=0.8)
    ) / (2.0 * step)
    assert np.sum(gradient_local * direction) == pytest.approx(finite_local, rel=2e-7)
    assert np.sum(gradient_memory * direction) == pytest.approx(finite_memory, rel=2e-7)
    assert np.trace(gradient_local) == pytest.approx(0.0, abs=1e-14)
    assert np.trace(gradient_memory) == pytest.approx(0.0, abs=1e-14)


def test_quartic_weak_field_onset() -> None:
    local = np.diag([1.0, 0.0, -1.0])
    memory = np.array([[0.3, 0.4, 0.0], [0.4, -0.1, 0.2], [0.0, 0.2, -0.2]])
    scale = 1e-3
    commutator = tidal_commutator(local, memory)
    leading = scale**4 * np.sum(commutator**2) / 2.0
    exact = bounded_misalignment_potential(scale * local, scale * memory)
    assert exact == pytest.approx(leading, rel=3e-6)


def test_small_spectral_fixture_is_finite_and_trace_free() -> None:
    axis = centered_axis(17, 2.0)
    density = gaussian_mixture_density(
        axis,
        [
            {
                "mass_fraction": 1.0,
                "center_L_sigma": [0.0, 0.0, 0.0],
                "sigma_L_sigma": [0.35, 0.25, 0.2],
            }
        ],
        total_mass=1.0,
    )
    field = spectral_tidal_misalignment(
        density,
        spacing=float(axis[1] - axis[0]),
        gravitational_constant=1.0,
        a_sigma=1.0,
        memory_length=1.0,
    )
    assert np.all(np.isfinite(field.bounded_potential))
    assert np.min(field.bounded_potential) >= 0.0
    assert np.max(field.bounded_potential) <= 1.0
    assert np.max(np.abs(np.trace(field.local_tide, axis1=-2, axis2=-1))) < 1e-12
    assert np.max(np.abs(np.trace(field.memory_tide, axis1=-2, axis2=-1))) < 1e-12


def test_frozen_v3e_report_records_the_near_miss_without_advancing() -> None:
    config_path = ROOT / "configs" / "sigma_v3e_tidal_misalignment_action_audit.json"
    report_path = ROOT / "results" / "sigma_v3e_tidal_misalignment_action_audit" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert not report["all_preregistered_gates_pass"]
    assert not report["gates"]["primary_morphology_separation"]
    assert report["gates"]["each_mass_morphology_separation"]
    assert report["gates"]["resolution_stability"]
    assert report["primary_median_cluster_to_galaxy_response_ratio"] == pytest.approx(
        8.978878124616749
    )
    assert report["decision"] == "retire_v3e_misalignment_as_frozen_structural_mechanism"
    assert not report["raw_holdout_opened"]
