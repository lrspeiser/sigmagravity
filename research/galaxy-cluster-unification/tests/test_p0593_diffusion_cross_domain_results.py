import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0593_coverage_conservation_and_solar():
    report = json.loads((ROOT / "results/p0593_diffusion_cross_domain/report.json").read_text())
    assert report["coverage"] == {"candidates": 320, "galaxies": 131, "outer_points": 968, "route_profiles": 1310}
    assert report["maximum_mass_conservation_error"] < 1e-10
    assert report["solar"]["finite_source_tail_pass"]
    assert report["solar"]["planetary_force_null_pass"]
    assert report["solar"]["PPN_Cassini_defined"] is False


def test_p0593_scores_and_impacts_are_finite():
    scores = pd.read_csv(ROOT / "results/p0593_diffusion_cross_domain/candidate_scores.csv")
    impacts = pd.read_csv(ROOT / "results/p0593_diffusion_cross_domain/parameter_impacts.csv")
    assert len(scores) == 320
    assert set(scores.scalar_completion) == {"none", "RAR"}
    assert np.all(np.isfinite(scores.select_dtypes(include=["number"])))
    assert set(impacts.parameter) == {"route_geometry", "strength", "route_fraction", "gate_power", "scalar_completion"}
