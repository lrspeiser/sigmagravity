import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0602_is_explicitly_posthoc_and_complete():
    base = ROOT / "results/p0602_raw_radial_structure_diagnostic"
    report = json.loads((base / "report.json").read_text())
    scores = pd.read_csv(base / "variant_scores.csv")
    impacts = pd.read_csv(base / "parameter_impacts.csv")
    assert report["status"] == "complete_posthoc_spent_data_diagnostic"
    assert report["interpretation"]["heldout_is_fresh"] is False
    assert len(scores) == 17
    assert set(impacts.family) == {"amplitude", "carrier", "power", "radial_power", "threshold"}
    # Some variants legitimately fail exact image-root recovery and therefore
    # carry infinite aggregate RMS. They must never be missing/NaN, while the
    # baseline and all reported impact summaries must be finite.
    assert not scores.select_dtypes(include=["number"]).isna().any().any()
    baseline = scores[scores.variant_id.eq("P0599_baseline")]
    assert np.all(np.isfinite(baseline.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(impacts.select_dtypes(include=["number"])))
