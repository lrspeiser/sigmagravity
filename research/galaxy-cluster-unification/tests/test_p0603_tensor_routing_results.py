import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0603_coverage_and_conservation():
    base = ROOT / "results/p0603_tensor_routing"
    report = json.loads((base / "report.json").read_text())
    assert report["coverage"] == {
        "tensor_candidates": 30,
        "clusters": 10,
        "cluster_targets": 20,
        "cluster_folds": 5,
        "galaxies": 131,
        "galaxy_outer_points": 968,
    }
    diagnostic = report["field_equation_diagnostics"]
    assert diagnostic["maximum_full_grid_mass_conservation_error"] < 1e-12
    assert diagnostic["maximum_relative_curl_norm"] < 1e-12
    assert diagnostic["maximum_relative_Poisson_residual"] < 1e-12
    assert diagnostic["maximum_radial_projection_conservation_error"] < 1e-12


def test_p0603_tables_are_finite_and_complete():
    base = ROOT / "results/p0603_tensor_routing"
    cluster = pd.read_csv(base / "cluster_candidate_scores.csv")
    folds = pd.read_csv(base / "cluster_fold_selections.csv")
    oof = pd.read_csv(base / "cluster_oof_scores.csv")
    impacts = pd.read_csv(base / "parameter_impacts.csv")
    galaxies = pd.read_csv(base / "galaxy_candidate_scores.csv")
    assert len(cluster) == 10 * 2 * 34
    assert len(folds) == 5
    assert len(oof) == 10 * 2 * 5
    assert len(impacts) == 8
    assert len(galaxies) == 30
    metric_columns = ["jensen_shannon", "pearson", "normalized_RMSE", "centroid_offset_kpc"]
    # Control rows do not have tensor parameters, so those parameter columns
    # are intentionally empty. Every calculated score and every tensor-only
    # table must remain finite.
    assert np.all(np.isfinite(cluster[metric_columns]))
    assert np.all(np.isfinite(oof[metric_columns]))
    for frame in (folds, impacts, galaxies):
        assert np.all(np.isfinite(frame.select_dtypes(include=["number"])))
