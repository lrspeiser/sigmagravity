import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0600_coverage_and_primary_are_frozen():
    report = json.loads((ROOT / "results/p0600_bcg_potential_gap/report.json").read_text())
    assert report["coverage"] == {
        "systems": 34,
        "direct_Tian2024": 11,
        "calibrated_DynPop_proxy": 23,
        "variants": 24,
    }
    assert report["primary"]["variant_id"] == (
        "BCG_plus_eRASS_median_gas__CLASH_median_C__weak_host_S1"
    )


def test_p0600_tables_are_finite_and_complete():
    scores = pd.read_csv(ROOT / "results/p0600_bcg_potential_gap/variant_scores.csv")
    predictions = pd.read_csv(ROOT / "results/p0600_bcg_potential_gap/predictions.csv")
    impacts = pd.read_csv(ROOT / "results/p0600_bcg_potential_gap/parameter_impacts.csv")
    assert len(scores) == 24
    assert len(predictions) == 24 * 34
    assert set(impacts.parameter) == {"potential_source", "radial_shape", "source_screen"}
    for frame in (scores, predictions, impacts):
        assert np.all(np.isfinite(frame.select_dtypes(include=["number"])))
