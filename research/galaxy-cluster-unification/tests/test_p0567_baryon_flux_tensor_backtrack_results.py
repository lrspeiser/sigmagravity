import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0567_baryon_flux_tensor_backtrack"


def test_p0567_report_covers_the_frozen_cluster_and_map_sample():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["systems"] == 13
    assert report["coverage"]["development_systems"] == 7
    assert report["coverage"]["new_analysis_holdout_systems"] == 3
    assert report["coverage"]["lenstool_realizations_read"] == 1300
    assert report["coverage"]["uncertainty_field_realizations"] == 13 * 16
    assert report["coverage"]["glafic_method_controls"] == 10


def test_p0567_tensor_bounds_are_finite_and_physically_ordered():
    metrics = pd.read_csv(RESULTS / "field_metrics.csv")
    primary = metrics[metrics.method.eq("lenstool_ensemble")]
    assert primary.system.nunique() == 13
    for column in [
        "weighted_feasible_fraction",
        "residual_weighted_feasible_fraction",
    ]:
        assert primary[column].between(0.0, 1.0 + 1e-12).all()
    for column in [
        "weighted_median_chi_min_feasible",
        "weighted_p90_chi_min_feasible",
        "residual_weighted_median_chi_min_feasible",
        "residual_weighted_p90_chi_min_feasible",
    ]:
        assert np.isfinite(primary[column]).all()
        assert (primary[column] >= 1.0).all()
    assert (
        primary.weighted_p90_chi_min_feasible
        >= primary.weighted_median_chi_min_feasible
    ).all()


def test_p0567_uncertainty_and_backtracks_are_reproducible():
    uncertainty = pd.read_csv(RESULTS / "uncertainty.csv")
    assert uncertainty.groupby("system").size().eq(16).all()
    paths = pd.read_csv(RESULTS / "peak_backtracks.csv")
    assert len(paths) == 60
    assert paths.path_length_kpc.ge(0.0).all()
    assert paths.direct_distance_kpc.ge(0.0).all()
    assert paths.path_x_kpc.str.len().gt(0).all()
    fresh_primary = paths[
        paths.method.eq("lenstool_ensemble") & paths.cohort.ne("spent_pilot")
    ]
    assert np.isclose(fresh_primary.reached_baryon.mean(), 0.96)


def test_p0567_gates_do_not_promote_a_predictive_formula():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["gates"]["local_feasibility_gate"]
    assert report["gates"]["practical_distortion_gate"]
    assert report["gates"]["method_robustness_gate"]
    assert report["gates"]["no_formula_promoted"]
    assert "No smooth universal K is fitted" in report["interpretation"]["what_is_not_measured"]
