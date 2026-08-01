import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0599_coverage_and_whole_object_folds():
    report = json.loads((ROOT / "results/p0599_bounded_potential_amplitude/report.json").read_text())
    assert report["coverage"] == {
        "candidates": 480,
        "galaxies": 131,
        "galaxy_outer_points": 968,
        "clusters": 20,
        "cluster_points": 84,
        "folds": 5,
    }
    selections = pd.read_csv(ROOT / "results/p0599_bounded_potential_amplitude/fold_selections.csv")
    assert len(selections) == 5
    assert set(selections.fold) == set(range(5))
    assert (selections.training_galaxy_RMSE_ratio_to_fixed_RAR <= 1.02).all()


def test_p0599_tables_are_finite_and_complete():
    candidates = pd.read_csv(ROOT / "results/p0599_bounded_potential_amplitude/candidate_scores.csv")
    folds = pd.read_csv(ROOT / "results/p0599_bounded_potential_amplitude/candidate_fold_scores.csv")
    impacts = pd.read_csv(ROOT / "results/p0599_bounded_potential_amplitude/parameter_impacts.csv")
    galaxies = pd.read_csv(ROOT / "results/p0599_bounded_potential_amplitude/galaxy_oof_predictions.csv")
    clusters = pd.read_csv(ROOT / "results/p0599_bounded_potential_amplitude/cluster_oof_predictions.csv")
    assert len(candidates) == 480
    assert len(folds) == 480 * 5
    assert len(galaxies) == 968
    assert len(clusters) == 84
    assert set(impacts.parameter) == {
        "spatial_mode",
        "carrier",
        "amplitude_A",
        "potential_threshold_chi",
        "potential_power",
    }
    # Two inherited SPARC metadata placeholders are intentionally empty for
    # every row; all quantities calculated by P0599 must remain finite.
    galaxy_calculated = galaxies.drop(columns=["local_density_g_cm3", "coherence"])
    for frame in (candidates, folds, impacts, galaxy_calculated, clusters):
        assert np.all(np.isfinite(frame.select_dtypes(include=["number"])))
