from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.sigma_triaxial_memory import (
    acceleration_power_screen,
    axisymmetric_tidal_tensor,
    bounded_triaxial_gradient,
    bounded_triaxial_potential,
    centered_axis,
    gaussian_mixture_density,
    high_acceleration_screen,
    integrated_response,
    spectral_tidal_memory,
    symmetric_trace_free,
    triaxial_invariants,
)

ROOT = Path(__file__).resolve().parents[1]


def test_axisymmetric_and_maximally_triaxial_values() -> None:
    axisymmetric = np.diag([2.0, -1.0, -1.0])
    triaxial = np.diag([1.0, 0.0, -1.0])
    assert bounded_triaxial_potential(axisymmetric) == pytest.approx(0.0, abs=1e-14)
    assert bounded_triaxial_potential(triaxial) == pytest.approx(8.0 / 27.0)
    assert triaxial_invariants(triaxial)[2] == pytest.approx(8.0)


def test_rotation_invariance_and_trace_free_gradient() -> None:
    matrix = symmetric_trace_free(np.array([[0.7, -0.3, 0.2], [-0.3, -0.1, 0.4], [0.2, 0.4, -0.6]]))
    rotation, _ = np.linalg.qr(np.array([[0.8, 0.2, -0.1], [0.3, -0.9, 0.4], [0.5, 0.1, 0.7]]))
    rotated = rotation @ matrix @ rotation.T
    assert bounded_triaxial_potential(rotated) == pytest.approx(
        bounded_triaxial_potential(matrix), rel=1e-12
    )
    gradient = bounded_triaxial_gradient(matrix)
    assert gradient == pytest.approx(gradient.T, abs=1e-14)
    assert np.trace(gradient) == pytest.approx(0.0, abs=1e-14)


def test_analytic_gradient_matches_trace_free_directional_difference() -> None:
    matrix = symmetric_trace_free(
        np.array([[0.8, 0.15, -0.2], [0.15, -0.3, 0.1], [-0.2, 0.1, -0.5]])
    )
    direction = symmetric_trace_free(
        np.array([[0.2, -0.4, 0.3], [-0.4, 0.6, 0.1], [0.3, 0.1, -0.8]])
    )
    direction /= np.linalg.norm(direction)
    step = 1e-6
    finite = float(
        (
            bounded_triaxial_potential(matrix + step * direction)
            - bounded_triaxial_potential(matrix - step * direction)
        )
        / (2.0 * step)
    )
    analytic = float(np.sum(bounded_triaxial_gradient(matrix) * direction))
    assert analytic == pytest.approx(finite, rel=2e-7, abs=2e-9)


def test_high_acceleration_screen_limits() -> None:
    values = high_acceleration_screen(np.array([0.0, 1.0, 1e5, 1e100]))
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1e-20)
    assert values[3] == pytest.approx(0.0)
    assert acceleration_power_screen(10.0, 2.0) == pytest.approx(1.0 / 101.0)


def test_each_isolated_tide_is_null_but_overlap_is_not() -> None:
    first = axisymmetric_tidal_tensor([1.0, 0.0, 0.0], 1.0)
    second = axisymmetric_tidal_tensor([1.0, 1.0, 0.3], 0.7)
    assert bounded_triaxial_potential(first) == pytest.approx(0.0, abs=1e-14)
    assert bounded_triaxial_potential(second) == pytest.approx(0.0, abs=1e-14)
    assert bounded_triaxial_potential(first + second) > 1e-4


def test_small_spectral_fixture_is_finite_trace_free_and_mass_normalized() -> None:
    axis = centered_axis(17, 2.0)
    components = [
        {
            "mass_fraction": 1.0,
            "center_L_sigma": [0.0, 0.0, 0.0],
            "sigma_L_sigma": [0.35, 0.35, 0.35],
        }
    ]
    density = gaussian_mixture_density(axis, components, total_mass=0.7)
    spacing = float(axis[1] - axis[0])
    assert np.sum(density) * spacing**3 == pytest.approx(0.7, rel=1e-12)
    field = spectral_tidal_memory(
        density,
        spacing=spacing,
        gravitational_constant=1.0,
        a_sigma=1.0,
        memory_length=1.0,
    )
    assert np.max(np.abs(np.trace(field.memory, axis1=-2, axis2=-1))) < 1e-12
    assert np.all(np.isfinite(field.bounded_potential))
    assert np.min(field.bounded_potential) >= 0.0
    assert np.max(field.bounded_potential) <= 1.0
    assert integrated_response(field.bounded_potential, axis, analysis_half_width=1.0) >= 0.0


def test_after_memory_screen_order_remains_finite_and_trace_free() -> None:
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
    field = spectral_tidal_memory(
        density,
        spacing=float(axis[1] - axis[0]),
        gravitational_constant=1.0,
        a_sigma=1.0,
        memory_length=0.5,
        screen_power=2.0,
        screen_order="after_memory",
    )
    assert np.all(np.isfinite(field.bounded_potential))
    assert np.max(np.abs(np.trace(field.memory, axis1=-2, axis2=-1))) < 1e-12


def test_frozen_v3d_report_records_the_preregistered_failure() -> None:
    config_path = ROOT / "configs" / "sigma_v3d_triaxial_memory_action_audit.json"
    report_path = ROOT / "results" / "sigma_v3d_triaxial_memory_action_audit" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert not report["all_preregistered_gates_pass"]
    assert not report["gates"]["primary_morphology_separation"]
    assert not report["gates"]["each_mass_morphology_separation"]
    assert report["gates"]["resolution_stability"]
    assert report["decision"] == "retire_v3d_discriminant_as_frozen_structural_mechanism"
    assert not report["raw_holdout_opened"]


def test_post_failure_report_cannot_change_the_v3d_decision() -> None:
    report_path = ROOT / "results" / "sigma_v3d_post_failure_diagnostics" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "diagnostic-only-after-frozen-failure"
    assert not report["frozen_v3d_decision_changed"]
    assert report["decision"] == "use_for_mechanism_selection_only"
    assert not report["raw_holdout_opened"]
