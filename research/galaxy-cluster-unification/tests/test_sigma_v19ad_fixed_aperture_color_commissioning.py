from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sigma_v19ad_fixed_aperture_color_commissioning import (
    aggregate_measurements,
    color_vector,
    feature_vector,
    fit_color_model,
    predict,
)

MODEL = {
    "feature_centers": {"B_minus_R": 2.4, "R_minus_I": 1.1},
    "feature_scales": {"B_minus_R": 1.0, "R_minus_I": 0.5},
    "outputs": ["g_minus_r", "r_minus_i", "i_minus_z"],
    "robust_residual_scale_mag": 0.15,
    "ridge_penalty": 0.25,
    "predictive_scale_floor_mag": 0.05,
    "maximum_function_evaluations": 5000,
}


def test_aggregate_uses_only_frozen_instrument_flag_and_aperture() -> None:
    sample = [{"cluster": "BULLET", "object_id": "01", "nsc_id": "n1", "split": "development", "B": "22", "R": "20", "I": "19"}]
    base = {"objectid": "n1", "filter": "g", "mag_aper4": "21", "magerr_aper4": "0.1", "flags": "0", "instrument": "c4d"}
    measurements = [base, {**base, "mag_aper4": "21.2"}, {**base, "instrument": "ksb", "mag_aper4": "10"}, {**base, "flags": "1", "mag_aper4": "11"}]
    rows = aggregate_measurements(sample, measurements, aperture_arcsec=4, instrument="c4d", filters=["g"])
    assert rows[0]["g"] == pytest.approx(21.1)
    assert rows[0]["g_measurements"] == 2


def test_color_and_feature_vectors_are_exact() -> None:
    row = {"B": 22.0, "R": 19.6, "I": 18.5, "g": 21.4, "r": 20.1, "i": 19.5, "z": 19.2}
    assert feature_vector(row, MODEL) == pytest.approx([1.0, 0.0, 0.0])
    assert color_vector(row) == pytest.approx([1.3, 0.6, 0.3])


def test_manufactured_color_model_recovers_affine_relation() -> None:
    coefficient = np.asarray([[1.3, 0.5, 0.3], [-0.2, 0.1, 0.05], [0.1, -0.1, 0.02]])
    rows = []
    for index, b_minus_r in enumerate(np.linspace(1.4, 3.4, 12)):
        r_minus_i = 0.7 + 0.35 * (index % 4)
        row = {"B": 22.0, "R": 22.0 - b_minus_r, "I": 22.0 - b_minus_r - r_minus_i}
        colors = feature_vector(row, MODEL) @ coefficient
        row["g"] = 21.0
        row["r"] = row["g"] - colors[0]
        row["i"] = row["r"] - colors[1]
        row["z"] = row["i"] - colors[2]
        rows.append(row)
    model = fit_color_model(rows, MODEL)
    assert all(record["success"] for record in model["optimizers"])
    for row in rows:
        assert predict(model, row, MODEL) == pytest.approx(color_vector(row), abs=0.03)
