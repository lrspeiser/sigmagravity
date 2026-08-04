from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.sigma_variational_source import (
    helmholtz_memory,
    kappa_to_shear,
    misalignment_potential_and_gradients,
    projected_variational_source,
    shear_to_stf,
    spectral_stf_hessian,
    variational_source_from_potential,
)

ROOT = Path(__file__).resolve().parents[1]


def _periodic_fixture(points: int = 24) -> tuple[np.ndarray, float]:
    spacing = 0.3
    coordinate = np.arange(points) * spacing
    x, y = np.meshgrid(coordinate, coordinate)
    potential = (
        0.7 * np.cos(2.0 * np.pi * x / (points * spacing))
        + 0.4 * np.sin(4.0 * np.pi * y / (points * spacing))
        + 0.3 * np.cos(2.0 * np.pi * (x + 2.0 * y) / (points * spacing))
    )
    return potential, spacing


def test_tensor_potential_gradients_match_directional_difference() -> None:
    potential, spacing = _periodic_fixture()
    local = spectral_stf_hessian(potential, spacing=spacing)
    memory = helmholtz_memory(local, spacing=spacing, length=0.8)
    direction = spectral_stf_hessian(np.roll(potential, 3, axis=0), spacing=spacing)
    direction /= np.sqrt(np.mean(np.square(direction)))
    density, gradient_local, gradient_memory = misalignment_potential_and_gradients(
        local, memory
    )
    step = 1e-6
    finite_local = (
        np.sum(misalignment_potential_and_gradients(local + step * direction, memory)[0])
        - np.sum(misalignment_potential_and_gradients(local - step * direction, memory)[0])
    ) / (2.0 * step)
    finite_memory = (
        np.sum(misalignment_potential_and_gradients(local, memory + step * direction)[0])
        - np.sum(misalignment_potential_and_gradients(local, memory - step * direction)[0])
    ) / (2.0 * step)
    assert np.sum(gradient_local * direction) == pytest.approx(finite_local, rel=1e-6)
    assert np.sum(gradient_memory * direction) == pytest.approx(finite_memory, rel=1e-6)
    assert np.min(density) >= -1e-14
    assert np.max(density) <= 1.0 + 1e-12


def test_composed_scalar_functional_derivative_includes_memory_pullback() -> None:
    potential, spacing = _periodic_fixture()
    direction = np.roll(potential, (2, -3), axis=(0, 1))
    direction /= np.sqrt(np.mean(np.square(direction)))
    memory_length = 0.9
    tensor_scale = 1.7
    density, source = variational_source_from_potential(
        potential,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )
    step = 2e-6
    plus = variational_source_from_potential(
        potential + step * direction,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )[0]
    minus = variational_source_from_potential(
        potential - step * direction,
        spacing=spacing,
        memory_length=memory_length,
        tensor_scale=tensor_scale,
    )[0]
    finite = np.sum(plus - minus) * spacing**2 / (2.0 * step)
    analytic = np.sum(source * direction) * spacing**2
    assert analytic == pytest.approx(finite, rel=1e-6, abs=1e-10)
    assert abs(float(np.mean(source))) <= 1e-12 * float(np.sqrt(np.mean(source**2)))
    assert np.all(np.isfinite(density))


def test_kappa_to_shear_is_an_integrable_e_mode_transform() -> None:
    kappa, spacing = _periodic_fixture()
    kappa -= np.mean(kappa)
    shear_1, shear_2 = kappa_to_shear(kappa, spacing=spacing)
    kappa_transform = np.fft.fft2(kappa, norm="ortho")
    first_transform = np.fft.fft2(shear_1, norm="ortho")
    second_transform = np.fft.fft2(shear_2, norm="ortho")
    ky = 2.0 * np.pi * np.fft.fftfreq(kappa.shape[0], d=spacing)
    kx = 2.0 * np.pi * np.fft.fftfreq(kappa.shape[1], d=spacing)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    squared = kx_grid**2 + ky_grid**2
    first_kernel = np.divide(
        kx_grid**2 - ky_grid**2,
        squared,
        out=np.zeros_like(squared),
        where=squared > 0.0,
    )
    second_kernel = np.divide(
        2.0 * kx_grid * ky_grid,
        squared,
        out=np.zeros_like(squared),
        where=squared > 0.0,
    )
    reconstructed = first_kernel * first_transform + second_kernel * second_transform
    assert reconstructed[1:, :] == pytest.approx(kappa_transform[1:, :], abs=1e-11)
    assert reconstructed[0, 1:] == pytest.approx(kappa_transform[0, 1:], abs=1e-11)
    assert reconstructed[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_projected_source_is_signed_finite_and_rotation_covariant() -> None:
    potential, spacing = _periodic_fixture(25)
    tide = spectral_stf_hessian(potential, spacing=spacing)
    shear_1 = tide[..., 0, 0]
    shear_2 = tide[..., 0, 1]
    result = projected_variational_source(
        shear_1,
        shear_2,
        spacing=spacing,
        memory_length=0.8,
        tensor_scale=1.2,
        padding_factor=1,
    )
    rotated_potential = np.rot90(potential)
    rotated_tide = spectral_stf_hessian(rotated_potential, spacing=spacing)
    rotated = projected_variational_source(
        rotated_tide[..., 0, 0],
        rotated_tide[..., 0, 1],
        spacing=spacing,
        memory_length=0.8,
        tensor_scale=1.2,
        padding_factor=1,
    )
    assert np.all(np.isfinite(result.source))
    assert np.mean(result.source > 0.0) > 0.1
    assert np.mean(result.source < 0.0) > 0.1
    assert rotated.source == pytest.approx(np.rot90(result.source), abs=2e-10)
    assert result.local_tide == pytest.approx(shear_to_stf(shear_1, shear_2) / 1.2)


def test_padding_preserves_shape_and_full_periodic_conservation() -> None:
    potential, spacing = _periodic_fixture(17)
    tide = spectral_stf_hessian(potential, spacing=spacing)
    result = projected_variational_source(
        tide[..., 0, 0],
        tide[..., 0, 1],
        spacing=spacing,
        memory_length=1.1,
        tensor_scale=0.7,
        padding_factor=2,
    )
    assert result.source.shape == potential.shape
    assert result.unit_eta_kappa.shape == potential.shape
    assert result.unit_eta_shear_1.shape == potential.shape
    assert result.unit_eta_shear_2.shape == potential.shape
    rms = float(np.sqrt(np.mean(result.full_source**2)))
    assert rms > 0.0
    assert abs(float(np.mean(result.full_source))) / rms < 1e-12


def test_frozen_v4a_report_matches_protocol_and_decision() -> None:
    config_path = ROOT / "configs" / "sigma_v4a_projected_variational_source_audit.json"
    report_path = (
        ROOT / "results" / "sigma_v4a_projected_variational_source_audit" / "report.json"
    )
    if not report_path.exists():
        pytest.skip("frozen v4A audit has not been run yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert report["sample_is_spent"]
    assert not report["raw_holdout_opened"]
    assert report["all_preregistered_gates_pass"] == all(report["gates"].values())
    expected = (
        "advance_to_covariant_3d_completion_before_holdout"
        if report["all_preregistered_gates_pass"]
        else "retire_exact_v4a_projected_variational_source"
    )
    assert report["decision"] == expected
