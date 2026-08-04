from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.sigma_variational_source import (
    helmholtz_memory,
    misalignment_potential_and_gradients,
)

ROOT = Path(__file__).resolve().parents[1]
from voidscreen.sigma_vector_stress import (
    projected_vector_stress_source,
    spectral_gradient,
    variational_source_from_potential,
    vector_chain_gradient,
    vector_stress,
)


def _periodic_fixture(points: int = 25) -> tuple[np.ndarray, float]:
    spacing = 0.4
    coordinate = np.arange(points) * spacing
    x, y = np.meshgrid(coordinate, coordinate)
    scalar = (
        0.8 * np.cos(2.0 * np.pi * x / (points * spacing))
        + 0.5 * np.sin(4.0 * np.pi * y / (points * spacing))
        + 0.25 * np.cos(2.0 * np.pi * (x + 2.0 * y) / (points * spacing))
    )
    return scalar, spacing


def test_stress_is_symmetric_trace_free_and_quadratic() -> None:
    scalar, spacing = _periodic_fixture()
    vector = spectral_gradient(scalar, spacing=spacing)
    normalized, stress = vector_stress(vector, vector_scale=1.7)
    assert stress == pytest.approx(np.swapaxes(stress, -1, -2))
    assert np.max(np.abs(np.trace(stress, axis1=-2, axis2=-1))) < 1e-14
    _, doubled = vector_stress(2.0 * vector, vector_scale=1.7)
    assert doubled == pytest.approx(4.0 * stress)
    assert normalized == pytest.approx(vector / 1.7)


def test_local_stress_chain_gradient_matches_directional_difference() -> None:
    scalar, spacing = _periodic_fixture()
    physical = spectral_gradient(scalar, spacing=spacing)
    direction = spectral_gradient(np.roll(scalar, (2, -1), axis=(0, 1)), spacing=spacing)
    direction /= np.sqrt(np.mean(np.square(direction)))
    scale = 1.3
    normalized, local = vector_stress(physical, vector_scale=scale)
    memory = helmholtz_memory(local, spacing=spacing, length=1.1)
    _, gradient_local, _ = misalignment_potential_and_gradients(local, memory)
    chain = vector_chain_gradient(normalized, gradient_local, vector_scale=scale)
    step = 1e-6
    plus_local = vector_stress(physical + step * direction, vector_scale=scale)[1]
    minus_local = vector_stress(physical - step * direction, vector_scale=scale)[1]
    finite = (
        np.sum(misalignment_potential_and_gradients(plus_local, memory)[0])
        - np.sum(misalignment_potential_and_gradients(minus_local, memory)[0])
    ) / (2.0 * step)
    analytic = np.sum(chain * direction)
    assert analytic == pytest.approx(finite, rel=1e-6, abs=1e-10)


def test_full_composed_functional_derivative_includes_memory() -> None:
    scalar, spacing = _periodic_fixture()
    direction = np.roll(scalar, (3, -2), axis=(0, 1))
    direction /= np.sqrt(np.mean(np.square(direction)))
    memory_length = 1.2
    vector_scale = 0.9
    density, source = variational_source_from_potential(
        scalar,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )
    step = 2e-6
    plus = variational_source_from_potential(
        scalar + step * direction,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[0]
    minus = variational_source_from_potential(
        scalar - step * direction,
        spacing=spacing,
        memory_length=memory_length,
        vector_scale=vector_scale,
    )[0]
    finite = np.sum(plus - minus) * spacing**2 / (2.0 * step)
    analytic = np.sum(source * direction) * spacing**2
    assert analytic == pytest.approx(finite, rel=1e-6, abs=1e-10)
    rms = float(np.sqrt(np.mean(source**2)))
    assert abs(float(np.mean(source))) / rms < 1e-12
    assert np.all(np.isfinite(density))


def test_projected_source_is_signed_and_rotation_covariant() -> None:
    scalar, spacing = _periodic_fixture()
    vector = spectral_gradient(scalar, spacing=spacing)
    result = projected_vector_stress_source(
        vector[..., 0],
        vector[..., 1],
        spacing=spacing,
        memory_length=1.0,
        vector_scale=1.1,
        padding_factor=1,
    )
    rotated_scalar = np.rot90(scalar)
    rotated_vector = spectral_gradient(rotated_scalar, spacing=spacing)
    rotated = projected_vector_stress_source(
        rotated_vector[..., 0],
        rotated_vector[..., 1],
        spacing=spacing,
        memory_length=1.0,
        vector_scale=1.1,
        padding_factor=1,
    )
    assert np.all(np.isfinite(result.source))
    assert np.mean(result.source > 0.0) > 0.1
    assert np.mean(result.source < 0.0) > 0.1
    assert rotated.source == pytest.approx(np.rot90(result.source), abs=2e-10)


def test_padding_preserves_shape_and_global_conservation() -> None:
    scalar, spacing = _periodic_fixture(17)
    vector = spectral_gradient(scalar, spacing=spacing)
    result = projected_vector_stress_source(
        vector[..., 0],
        vector[..., 1],
        spacing=spacing,
        memory_length=1.3,
        vector_scale=0.8,
        padding_factor=2,
    )
    assert result.source.shape == scalar.shape
    assert result.unit_eta_kappa.shape == scalar.shape
    assert result.unit_eta_shear_1.shape == scalar.shape
    assert result.unit_eta_shear_2.shape == scalar.shape
    rms = float(np.sqrt(np.mean(result.full_source**2)))
    assert rms > 0.0
    assert abs(float(np.mean(result.full_source))) / rms < 1e-12


def test_frozen_v4b_report_matches_protocol_and_decision() -> None:
    config_path = ROOT / "configs" / "sigma_v4b_vector_stress_memory_audit.json"
    report_path = ROOT / "results" / "sigma_v4b_vector_stress_memory_audit" / "report.json"
    if not report_path.exists():
        pytest.skip("frozen v4B audit has not been run yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert report["sample_is_spent"]
    assert not report["raw_holdout_opened"]
    assert report["all_preregistered_gates_pass"] == all(report["gates"].values())
    expected = (
        "advance_to_covariant_vector_stress_completion_before_holdout"
        if report["all_preregistered_gates_pass"]
        else "retire_exact_v4b_vector_stress_memory_source"
    )
    assert report["decision"] == expected
