import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0596_shape_gate_cv_coverage():
    report = json.loads((ROOT / "results/p0596_radial_shape_gate_cv/report.json").read_text())
    assert report["coverage"] == {"galaxies": 131, "outer_points": 968, "candidates": 360, "folds": 5}
    selections = pd.read_csv(ROOT / "results/p0596_radial_shape_gate_cv/fold_selections.csv")
    assert len(selections) == 10
    assert set(selections.family) == {"shape_gate", "no_shape"}


def test_p0596_result_tables_are_finite():
    candidates = pd.read_csv(ROOT / "results/p0596_radial_shape_gate_cv/candidate_fold_scores.csv")
    galaxies = pd.read_csv(ROOT / "results/p0596_radial_shape_gate_cv/galaxy_scores.csv")
    impacts = pd.read_csv(ROOT / "results/p0596_radial_shape_gate_cv/parameter_impacts.csv")
    assert len(candidates) == 360 * 5
    assert len(galaxies) == 131
    assert set(impacts.parameter) == {"q_R80", "route_fraction_max", "acceleration_gate_power", "shape_gate"}
    nullable_gate_columns = {"shape_midpoint", "shape_width"}
    required_numeric = candidates.select_dtypes(include=["number"]).drop(
        columns=list(nullable_gate_columns), errors="ignore"
    )
    assert np.all(np.isfinite(required_numeric))
    assert candidates.loc[candidates.shape_gate != "none", ["shape_midpoint", "shape_width"]].notna().all().all()
    assert np.all(np.isfinite(galaxies.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(impacts.select_dtypes(include=["number"])))
