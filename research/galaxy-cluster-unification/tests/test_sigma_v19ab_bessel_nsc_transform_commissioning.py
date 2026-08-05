from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sigma_v19ab_bessel_nsc_transform_commissioning import (
    fit_transform,
    observed_colors,
    observed_offsets,
    photometric_features,
    predict_transform,
    predicted_colors,
)

MODEL_SPEC = {
    "feature_centers": {"B_minus_R": 2.4, "R_minus_I": 1.1},
    "feature_scales": {"B_minus_R": 1.0, "R_minus_I": 0.5},
    "output_offsets": ["g_minus_B", "r_minus_R", "i_minus_I", "z_minus_I"],
    "robust_residual_scale_mag": 0.25,
    "ridge_penalty": 0.25,
    "predictive_scale_floor_mag": 0.15,
    "maximum_function_evaluations": 5000,
}


def test_features_use_only_two_published_colors() -> None:
    row = {"B": 22.0, "R": 19.6, "I": 18.5}
    assert photometric_features(row, MODEL_SPEC) == pytest.approx([1.0, 0.0, 0.0])


def test_offset_and_color_algebra_is_exact() -> None:
    row = {"B": 22.0, "R": 19.6, "I": 18.5, "g": 21.4, "r": 20.1, "i": 19.5, "z": 19.2}
    offsets = observed_offsets(row)
    assert offsets == pytest.approx([-0.6, 0.5, 1.0, 0.7])
    assert predicted_colors(row, offsets) == pytest.approx(observed_colors(row))


def test_robust_affine_fit_recovers_manufactured_transform() -> None:
    coefficient = np.asarray(
        [
            [-0.5, 0.4, 1.0, 0.8],
            [-0.3, 0.1, 0.2, 0.1],
            [0.2, -0.1, 0.1, -0.2],
        ]
    )
    rows = []
    for index, b_minus_r in enumerate(np.linspace(1.4, 3.4, 12)):
        r_minus_i = 0.7 + 0.35 * (index % 4)
        row = {"B": 22.0 + 0.1 * index, "R": 22.0 + 0.1 * index - b_minus_r}
        row["I"] = row["R"] - r_minus_i
        offset = photometric_features(row, MODEL_SPEC) @ coefficient
        row.update(
            {
                "g": row["B"] + offset[0],
                "r": row["R"] + offset[1],
                "i": row["I"] + offset[2],
                "z": row["I"] + offset[3],
            }
        )
        rows.append(row)
    model = fit_transform(rows, MODEL_SPEC)
    assert all(record["success"] for record in model["optimizer_records"])
    for row in rows:
        assert predict_transform(model, row, MODEL_SPEC) == pytest.approx(
            observed_offsets(row), abs=0.04
        )
    assert model["predictive_scales"] == pytest.approx([0.15] * 4)
