import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0605_is_frozen_zero_gravity_fit_decomposition():
    base = ROOT / "results/p0605_strict_route_raw_lensing"
    report = json.loads((base / "report.json").read_text())
    assert report["coverage"] == {
        "models": 2,
        "training_images": 15,
        "spent_heldout_images": 7,
        "optimization_starts_per_model": 16,
        "fitted_gravity_parameters": 0,
    }
    assert report["selected_route"]["fraction_max"] == 1.0
    assert report["selected_route"]["length_over_R80"] == 0.25
    assert report["selected_route"]["width_over_R80"] == 0.5
    assert report["strict_interpretation"]["raw_data_are_fresh"] is False


def test_p0605_outputs_are_complete():
    base = ROOT / "results/p0605_strict_route_raw_lensing"
    scores = pd.read_csv(base / "scores.csv")
    predictions = pd.read_csv(base / "image_predictions.csv")
    parameters = pd.read_csv(base / "fitted_parameters.csv")
    profiles = pd.read_csv(base / "radial_profiles.csv")
    assert set(scores.model) == {"strict_route_RAR", "strict_route_P0599"}
    assert len(predictions) == 44
    assert len(parameters) == 12
    assert len(profiles) > 300
    assert np.all(np.isfinite(profiles.select_dtypes(include=["number"])))
    assert predictions[predictions.stage.eq("heldout")].root_converged.astype(bool).sum() >= 7
