import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0597_cross_domain_coverage_and_fixed_parameters():
    report = json.loads((ROOT / "results/p0597_simple_shape_law_cross_domain/report.json").read_text())
    assert report["coverage"] == {
        "galaxies": 131,
        "galaxy_outer_points": 968,
        "clusters": 10,
        "cluster_holdouts": 3,
        "lenstool_realizations": 1000,
    }
    assert report["parameters"] == {
        "q_R80": 3.0,
        "route_fraction_max": 0.3,
        "shape_midpoint": 0.6,
        "shape_width": 0.1,
        "acceleration_gate_power": 0.0,
    }


def test_p0597_result_tables_are_finite_and_complete():
    galaxies = pd.read_csv(ROOT / "results/p0597_simple_shape_law_cross_domain/galaxy_scores.csv")
    systems = pd.read_csv(ROOT / "results/p0597_simple_shape_law_cross_domain/cluster_system_scores.csv")
    uncertainty = pd.read_csv(ROOT / "results/p0597_simple_shape_law_cross_domain/cluster_uncertainty.csv")
    glafic = pd.read_csv(ROOT / "results/p0597_simple_shape_law_cross_domain/cluster_glafic_scores.csv")
    assert len(galaxies) == 131
    assert len(systems) == 30
    assert len(uncertainty) == 1000
    assert len(glafic) == 30
    for frame in (galaxies, systems, uncertainty, glafic):
        assert np.all(np.isfinite(frame.select_dtypes(include=["number"])))
