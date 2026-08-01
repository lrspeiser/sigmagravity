import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rxj2129_raw_theory_lensing"


def test_raw_lensing_report_preserves_the_predictive_verdict():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    scores = report["model_scores"]
    candidate = scores["locked_universal_candidate"]["heldout"]
    mond = scores["fixed_simple_MOND"]["heldout"]
    halo = scores["GR_plus_cluster_halo"]["heldout"]

    assert report["inputs"]["training_images"] == 15
    assert report["inputs"]["heldout_images"] == 7
    assert candidate["converged_roots"] == 7
    assert candidate["exact_radial_RMS_arcsec"] == pytest.approx(1.0642772678)
    assert mond["converged_roots"] == 3
    assert mond["exact_radial_RMS_arcsec"] is None
    assert report["comparisons"]["candidate_vs_fixed_simple_MOND_heldout_RMS_ratio"] is None
    assert halo["exact_radial_RMS_arcsec"] == pytest.approx(2.5361068844)
    assert report["strict_interpretation"]["independent_cluster_validation"] is False
    assert report["advance_gate_audit"]["passes_all"] is False


def test_raw_lensing_prediction_artifact_has_every_model_and_split():
    predictions = pd.read_csv(RESULTS / "image_predictions.csv")
    predictive = predictions[predictions["stage"].isin(["training", "heldout"])]
    assert predictive["model"].nunique() == 5
    assert (predictive.groupby("model").size() == 22).all()
    assert set(predictions[predictions["stage"] == "heldout"]["image_id"]) == {
        "1c",
        "2c",
        "3d",
        "4c",
        "5c",
        "6c",
        "7c",
    }
