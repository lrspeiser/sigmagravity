import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sigma_covariant_weak_field"


def test_covariant_gateway_preserves_the_mathematical_verdict():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    health = report["mathematical_health"]

    assert health["positive_time_kinetic_pass"] is True
    assert health["positive_parallel_gradient_pass"] is True
    assert health["maximum_parallel_speed_over_c"] == pytest.approx(1.4095367330)
    assert health["same_or_narrower_than_metric_light_cone_pass"] is False
    assert health["full_metric_slip_action_derived"] is False
    assert health["all_mathematical_gates_pass"] is False


def test_radially_selected_slip_improves_but_does_not_solve_raw_lensing():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    selection = report["radial_metric_slip_selection"]
    raw = report["raw_lensing"]
    gates = report["gate_audit"]

    assert selection["selected_zeta"] == pytest.approx(1.5)
    assert selection["selected_radial_cluster_RMSE_dex"] == pytest.approx(0.0947333402)
    assert selection["selected_lensing_to_dynamics_at_100kpc"] == pytest.approx(
        1.5682406650
    )
    assert raw["zero_slip"]["heldout"]["exact_radial_RMS_arcsec"] == pytest.approx(
        3.8076597158
    )
    assert raw["radial_selected"]["heldout"][
        "exact_radial_RMS_arcsec"
    ] == pytest.approx(2.2481778925)
    assert gates["raw_heldout_improves_zero_slip_pass"] is True
    assert gates["raw_heldout_below_1_arcsec_pass"] is False
    assert gates["all_mathematical_and_observational_gates_pass"] is False


def test_covariant_gateway_artifacts_have_all_frozen_rows():
    scan = pd.read_csv(RESULTS / "zeta_scan.csv")
    characteristics = pd.read_csv(RESULTS / "characteristics.csv")
    predictions = pd.read_csv(RESULTS / "raw_lensing_predictions.csv")

    assert len(scan) == 11
    assert set(characteristics["domain"]) == {"galaxy_archetype", "RXJ2129"}
    assert len(characteristics) == 1180
    assert set(predictions["closure"]) == {
        "conformal",
        "zero_slip",
        "radial_selected",
    }
    assert (predictions.groupby("closure").size() == 22).all()
