import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sigma_action_exploration"


def test_action_report_preserves_best_rows_and_scope():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    best = report["best_joint_rows"]["sigma_refracted_AQUAL"]

    assert report["model_rows"] == {
        "conformal_symmetron": 150,
        "sigma_gated_AQUAL": 150,
        "sigma_refracted_AQUAL": 125,
    }
    assert best["eta"] == pytest.approx(0.6)
    assert best["log10_rho_s_g_cm3"] == pytest.approx(-23.5)
    assert best["L_Sigma_kpc"] == pytest.approx(3.0)
    assert best["joint_descriptive_score_dex"] == pytest.approx(0.2303812065)
    assert best["galaxy_velocity_log_slope_100_250kpc"] == pytest.approx(-0.0109496341)
    assert report["interpretation"]["weak_backreaction_only"] is True
    assert report["interpretation"]["covariant_completion"] is False


def test_action_lensing_artifact_keeps_joint_and_diagnostic_rows_separate():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    lens = pd.read_csv(RESULTS / "raw_lensing_predictions.csv")
    diagnostics = report["raw_lensing_diagnostics"]

    assert set(lens["candidate_selection"]) == {"joint", "cluster_target_only"}
    assert (lens.groupby("candidate_selection").size() == 22).all()
    assert diagnostics["joint_row"]["scores"]["heldout"][
        "exact_radial_RMS_arcsec"
    ] == pytest.approx(3.8530143571)
    assert diagnostics["cluster_derived_target_only_row"]["scores"]["heldout"][
        "exact_radial_RMS_arcsec"
    ] == pytest.approx(2.2790307222)
    assert diagnostics["joint_row"]["scores"][
        "gravity_or_lensing_amplitudes_fit_to_images"
    ] == 0
