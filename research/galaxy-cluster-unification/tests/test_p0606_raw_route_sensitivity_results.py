import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0606_is_explicitly_spent_and_complete():
    base = ROOT / "results/p0606_raw_route_sensitivity"
    report = json.loads((base / "report.json").read_text())
    scores = pd.read_csv(base / "variant_scores.csv")
    impacts = pd.read_csv(base / "parameter_impacts.csv")
    predictions = pd.read_csv(base / "image_predictions.csv")
    assert report["status"] == "complete_posthoc_spent_raw_route_response"
    assert report["strict_interpretation"]["heldout_is_fresh"] is False
    assert len(scores) == 16
    assert len(impacts) == 3
    assert len(predictions) == 16 * 22
    assert set(impacts.parameter) == {"fraction_max", "length_over_R80", "width_over_R80"}
    assert np.all(np.isfinite(impacts.select_dtypes(include=["number"])))
