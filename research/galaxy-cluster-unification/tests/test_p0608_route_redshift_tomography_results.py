import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0608 = ROOT / "results/p0608_route_redshift_tomography"
P0608B = ROOT / "results/p0608b_tomography_optimizer_robustness"
P0608C = ROOT / "results/p0608c_tomography_random_start_robustness"


def test_redshift_tomography_has_real_geometric_leverage_but_no_identification():
    report = json.loads((P0608 / "report.json").read_text(encoding="utf-8"))
    ratios = pd.read_csv(P0608 / "distance_ratios.csv")
    assert report["coverage"]["source_redshifts"] == 7
    assert ratios.relative_to_reference.min() < 0.75
    assert ratios.relative_to_reference.max() > 1.05
    assert report["training_selected"]["gamma"] == 0.0
    assert report["primary_gamma_response"]["training_RMS_span_arcsec"] < 0.001
    assert report["primary_gamma_response"]["heldout_RMS_span_arcsec"] < 0.001
    assert report["interpretation"]["gamma_identified"] is False
    assert report["interpretation"]["hidden_arc_height_identified"] is False


def test_deterministic_repeat_is_explicitly_superseded():
    report = json.loads((P0608B / "report.json").read_text(encoding="utf-8"))
    assert report["status"].startswith("superseded_")
    assert report["interpretation"]["independent_random_start_realized"] is False
    assert report["interpretation"]["superseded_by"].endswith(
        "p0608c_tomography_random_start_robustness/report.json"
    )


def test_corrected_random_start_audit_rejects_gamma_identification():
    report = json.loads((P0608C / "report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["starts_per_repeat"] == 2
    assert report["coverage"]["complete_fits"] == 48
    assert report["interpretation"]["independent_random_start_realized"] is True
    assert report["interpretation"]["gamma_separable_from_optimizer_basin"] is False
    assert report["interpretation"]["heldout_order_is_stable"] is False
    assert report["interpretation"]["gamma_identified"] is False
    assert report["basin_noise"]["median_training_gamma_difference_to_basin_span_ratio"] < 1.0
