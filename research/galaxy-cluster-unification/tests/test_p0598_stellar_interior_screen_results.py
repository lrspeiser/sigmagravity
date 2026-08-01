import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_p0598_stellar_interior_screen import weighted_radii


ROOT = Path(__file__).resolve().parents[1]


def test_p0598_gate_grid_and_selected_screen_are_complete():
    report = json.loads((ROOT / "results/p0598_stellar_interior_screen/report.json").read_text())
    scores = pd.read_csv(ROOT / "results/p0598_stellar_interior_screen/gate_scores.csv")
    assert len(scores) == 4
    assert set(scores.gate_power) == {0.0, 1.0, 2.0, 4.0}
    assert report["selected_safe_gate"]["gate_power"] in {1.0, 2.0, 4.0}
    assert report["gates"]["stellar_interior_screen_pass"]


def test_p0598_tables_are_finite():
    scores = pd.read_csv(ROOT / "results/p0598_stellar_interior_screen/gate_scores.csv")
    profiles = pd.read_csv(ROOT / "results/p0598_stellar_interior_screen/solar_profiles.csv")
    assert len(profiles) == 4 * 1024
    assert np.all(np.isfinite(scores.select_dtypes(include=["number"])))
    assert np.all(np.isfinite(profiles.select_dtypes(include=["number"])))


def test_weighted_radii_does_not_mutate_source_masses():
    frame = pd.DataFrame(
        {"x_arcsec": [0.0, 1.0, 2.0], "y_arcsec": [0.0, 0.0, 0.0], "mass_msun": [1.0, 2.0, 3.0]}
    )
    original = frame.mass_msun.copy()
    weighted_radii(frame)
    pd.testing.assert_series_equal(frame.mass_msun, original)
