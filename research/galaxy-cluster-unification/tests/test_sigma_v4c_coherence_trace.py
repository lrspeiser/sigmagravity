from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voidscreen.sigma_coherence_trace import (
    coherence_trace_state,
    directional_disorder,
    helmholtz_relative_residual,
    projected_coherence_trace,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "sigma_v4c_baryon_seeded_coherence_trace_audit" / "report.json"


def manufactured_vector(points: int = 25) -> np.ndarray:
    coordinate = np.arange(points, dtype=float)
    x, y = np.meshgrid(coordinate, coordinate)
    angle = 2.0 * np.pi * x / points + 0.35 * np.sin(2.0 * np.pi * y / points)
    magnitude = 2.0 + 0.4 * np.cos(2.0 * np.pi * (x + y) / points)
    return np.stack([magnitude * np.cos(angle), magnitude * np.sin(angle)], axis=-1)


def test_uniform_vector_has_zero_directional_disorder() -> None:
    vector = np.zeros((24, 24, 2), dtype=float)
    vector[..., 0] = 3.0
    _, _, raw, disorder, _ = directional_disorder(
        vector, spacing=0.4, memory_length=1.7, vector_scale=0.8
    )
    assert np.max(np.abs(raw)) < 1.0e-12
    assert np.max(np.abs(disorder)) < 1.0e-12


def test_directional_disorder_is_rotation_invariant() -> None:
    vector = manufactured_vector()
    angle = 0.713
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = np.einsum("ij,...j->...i", rotation, vector)
    original = directional_disorder(
        vector, spacing=0.5, memory_length=2.1, vector_scale=0.7
    )[3]
    transformed = directional_disorder(
        rotated, spacing=0.5, memory_length=2.1, vector_scale=0.7
    )[3]
    assert np.linalg.norm(transformed - original) / np.linalg.norm(original) < 1.0e-12


def test_trace_is_unique_positive_and_preserves_seed_integral() -> None:
    vector = manufactured_vector()
    baryons = np.ones(vector.shape[:2])
    *_, seed, trace = coherence_trace_state(
        vector,
        baryons,
        spacing=0.5,
        memory_length=2.1,
        vector_scale=0.7,
    )
    assert np.min(seed) >= 0.0
    assert np.min(trace) >= -1.0e-12
    assert abs(np.sum(trace) - np.sum(seed)) / np.sum(seed) < 1.0e-12
    assert (
        helmholtz_relative_residual(
            trace, seed, spacing=0.5, length=2.1
        )
        < 1.0e-10
    )


def test_high_field_activation_suppresses_the_seed() -> None:
    vector = manufactured_vector()
    baryons = np.ones(vector.shape[:2])
    seed = coherence_trace_state(
        vector,
        baryons,
        spacing=0.5,
        memory_length=2.1,
        vector_scale=0.02,
    )[-2]
    high = coherence_trace_state(
        1000.0 * vector,
        baryons,
        spacing=0.5,
        memory_length=2.1,
        vector_scale=0.02,
    )[-2]
    assert np.sum(high) / np.sum(seed) < 1.0e-4


def test_projected_trace_shapes_and_padding_stability() -> None:
    vector = manufactured_vector(21)
    baryons = np.exp(
        -0.5
        * (
            np.square(np.arange(21)[:, None] - 10)
            + np.square(np.arange(21)[None, :] - 10)
        )
        / 8.0**2
    )
    first = projected_coherence_trace(
        vector[..., 0],
        vector[..., 1],
        baryons,
        spacing=0.5,
        memory_length=2.1,
        vector_scale=0.7,
        padding_factor=2,
    )
    second = projected_coherence_trace(
        vector[..., 0],
        vector[..., 1],
        baryons,
        spacing=0.5,
        memory_length=2.1,
        vector_scale=0.7,
        padding_factor=3,
    )
    assert first.trace_state.shape == baryons.shape
    assert first.unit_eta_shear_1.shape == baryons.shape
    assert first.unit_eta_shear_2.shape == baryons.shape
    assert np.all(np.isfinite(first.trace_state))
    relative = np.linalg.norm(first.trace_state - second.trace_state) / np.linalg.norm(
        second.trace_state
    )
    assert relative < 0.2


@pytest.mark.skipif(not REPORT.exists(), reason="v4C report has not been generated")
def test_frozen_report_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["sample_is_spent"] is True
    assert report["raw_holdout_opened"] is False
    assert set(report["primary_shared_fit"]["per_channel"][0]) == {
        "AQUAL_baseline_normalized_RMSE",
        "channel",
        "cluster",
        "improved",
        "prediction_normalized_RMSE",
    }
    assert report["decision"] in {
        "advance_to_covariant_coherence_trace_action_before_holdout",
        "retire_exact_v4c_baryon_seeded_coherence_trace",
    }
