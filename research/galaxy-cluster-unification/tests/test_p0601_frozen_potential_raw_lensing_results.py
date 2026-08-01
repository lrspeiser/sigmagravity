import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0601_frozen_scope_and_coverage():
    report = json.loads(
        (ROOT / "results/p0601_frozen_potential_raw_lensing/report.json").read_text()
    )
    assert report["coverage"]["training_images"] == 15
    assert report["coverage"]["heldout_images"] == 7
    assert report["coverage"]["source_families"] == 7
    assert report["coverage"]["fitted_gravity_parameters"] == 0
    assert report["frozen_constants"]["amplitude_A"] == 3.0
    assert report["frozen_constants"]["potential_threshold_chi"] == 1e-6


def test_p0601_outputs_are_finite_and_complete():
    base = ROOT / "results/p0601_frozen_potential_raw_lensing"
    scores = pd.read_csv(base / "scores.csv")
    predictions = pd.read_csv(base / "image_predictions.csv")
    parameters = pd.read_csv(base / "fitted_parameters.csv")
    profiles = pd.read_csv(base / "radial_profiles.csv")
    assert set(scores.model) == {"fixed_RAR", "P0599_potential_shape"}
    assert len(predictions) == 44
    assert len(parameters) == 12
    assert len(profiles) == 412
    # Fixed RAR fails to recover all exact roots, so its aggregate RMS is
    # intentionally infinite.  The frozen P0599 candidate must be finite.
    p0599 = scores[scores.model.eq("P0599_potential_shape")]
    assert np.all(np.isfinite(p0599.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(profiles.select_dtypes(include=["number"])))
    assert predictions[predictions.stage.eq("heldout")].root_converged.astype(bool).sum() >= 7
