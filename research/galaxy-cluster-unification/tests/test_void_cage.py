from __future__ import annotations

import numpy as np

from voidscreen.void_cage import (
    balanced_rank_folds,
    power_law_hessian,
    tensor_metrics,
    yukawa_hessian,
)


def symmetric_six_source_shell(radius: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    axes = np.vstack([np.eye(3), -np.eye(3)]) * radius
    return axes, np.ones(6, dtype=np.float64)


def test_inverse_square_symmetric_shell_has_zero_hessian() -> None:
    offsets, charges = symmetric_six_source_shell()
    hessian = power_law_hessian(offsets, charges, force_power=2.0)
    assert np.allclose(hessian, 0.0, atol=1e-14)


def test_faster_than_inverse_square_shell_is_compressive() -> None:
    offsets, charges = symmetric_six_source_shell()
    metrics = tensor_metrics(power_law_hessian(offsets, charges, force_power=3.0))
    assert metrics["kappa_unit"] > 0.0
    assert metrics["fully_compressive"] is True
    assert metrics["compressive_directions"] == 3
    assert np.isclose(metrics["anisotropy"], 0.0, atol=1e-14)


def test_yukawa_symmetric_shell_is_compressive() -> None:
    offsets, charges = symmetric_six_source_shell()
    metrics = tensor_metrics(yukawa_hessian(offsets, charges, range_hmpc=1.5))
    assert metrics["kappa_unit"] > 0.0
    assert metrics["fully_compressive"] is True
    assert metrics["compressive_directions"] == 3
    assert np.isclose(metrics["anisotropy"], 0.0, atol=1e-14)


def test_balanced_rank_folds_cover_every_item() -> None:
    scores = np.linspace(0.1, 2.0, 23)
    assignments = balanced_rank_folds(scores, 5)
    assert assignments.shape == scores.shape
    assert set(assignments) == set(range(5))
    counts = np.bincount(assignments, minlength=5)
    assert counts.max() - counts.min() <= 1
