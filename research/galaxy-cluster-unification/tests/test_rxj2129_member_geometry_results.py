import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rxj2129_member_geometry"


def test_member_geometry_report_preserves_the_controlled_negative_result():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    scores = report["headline_scores"]
    gates = report["interpretation_gates"]

    assert report["inputs"]["members"] == 66
    assert report["controlled_change"]["net_added_member_mass_msun"] == 0.0
    assert report["controlled_change"]["azimuthally_averaged_radial_deflection_change"] == 0.0
    assert scores["smooth_baseline_heldout_RMS_arcsec"] == pytest.approx(3.8079150389)
    assert scores["actual_layout_refitted_geometry_heldout_RMS_arcsec"] == pytest.approx(
        3.5433439322
    )
    assert scores["fractional_improvement_after_full_geometry_refit"] == pytest.approx(
        0.0694792568
    )
    assert scores["refitted_geometry_randomization_empirical_p"] == pytest.approx(
        0.0606060606
    )
    assert gates["meaningful_improvement_passed"] is False
    assert gates["strong_absolute_score_passed"] is False
    assert gates["observed_arrangement_specificity_passed"] is False


def test_member_geometry_artifacts_keep_all_controls_and_images():
    variants = pd.read_csv(RESULTS / "variant_scores.csv")
    predictions = pd.read_csv(RESULTS / "image_predictions.csv")
    randomizations = pd.read_csv(RESULTS / "randomization_scores.csv")
    diagnostics = pd.read_csv(RESULTS / "image_diagnostics.csv")

    assert set(variants["variant"]) == {
        "smooth_baseline",
        "central_catalog_fixed_geometry",
        "central_catalog_matched_optimizer",
        "central_catalog",
        "half_catalog_mass",
        "double_catalog_mass",
        "half_catalog_size",
        "double_catalog_size",
    }
    assert (predictions.groupby("variant").size() == 22).all()
    assert (randomizations.groupby("mode").size().to_dict()) == {
        "fixed_geometry": 256,
        "refitted_geometry": 32,
    }
    assert len(diagnostics) == 7
    assert (diagnostics["residual_improvement_arcsec"] > 0.0).all()
    doubled = variants[variants["variant"] == "double_catalog_mass"].iloc[0]
    assert doubled["fractional_heldout_improvement_vs_baseline"] == pytest.approx(
        0.1308047183
    )
    assert doubled["heldout_RMS_arcsec"] > 3.0
