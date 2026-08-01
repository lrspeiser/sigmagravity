import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gravity_flow_inverse"


def test_inverse_report_has_complete_conserved_sample():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["systems"] == 10
    assert report["coverage"]["hard_photoz_baryonic_sources"] == 832
    assert report["coverage"]["transport_solutions"] == 820
    assert report["aggregate_primary_inverse"]["maximum_source_marginal_error"] < 1e-8
    assert report["aggregate_primary_inverse"]["maximum_target_marginal_error"] < 1e-8
    assert report["radial_shuffle_control"]["maximum_source_marginal_error"] < 5e-5


def test_inverse_outputs_cover_both_methods_and_all_systems():
    routes = pd.read_csv(RESULTS / "route_statistics.csv")
    primary = routes[
        routes.destination_kind.eq("local_projected_excess")
        & routes.entropy_length_kpc.eq(50.0)
    ]
    assert len(primary) == 20
    assert primary.system.nunique() == 10
    assert set(primary.target_kind) == {"lenstool_ensemble_mean", "glafic_best"}
    assert np.all(primary.median_path_kpc > 0.0)
    assert np.all(primary.p90_path_kpc >= primary.median_path_kpc)
    maps = np.load(RESULTS / "path_maps.npz")
    assert len(maps.files) == 20
    assert all(np.isclose(maps[name].sum(), 1.0) for name in maps.files)


def test_driver_analysis_is_labeled_exploratory():
    report = json.loads((RESULTS / "driver_report.json").read_text(encoding="utf-8"))
    assert report["cluster_level_tests"] == 132
    assert "post hoc" in report["interpretation_limit"]
    correlations = pd.read_csv(RESULTS / "cluster_driver_correlations.csv")
    assert correlations.benjamini_hochberg_q.between(0.0, 1.0).all()
