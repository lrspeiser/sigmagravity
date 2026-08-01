import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0560 = ROOT / "results/p0560_accept_tensor_coupling_response"
P0561 = ROOT / "results/p0561_accept_tensor_extended_response"


def load_report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_response_protocols_are_explicitly_diagnostic_and_parameter_free_maps():
    for name in [
        "p0560_accept_tensor_coupling_response_protocol.json",
        "p0561_accept_tensor_extended_response_protocol.json",
    ]:
        protocol = json.loads((ROOT / "configs" / name).read_text())
        assert protocol["status"].startswith("frozen_before_any_")
        assert "diagnostic" in protocol["interpretation"]
        assert protocol["locked_map"]["new_map_parameters"] == 0
        assert protocol["locked_map"]["per_cluster_parameters"] == 0
        assert "No formula can be promoted" in protocol["claim_limits"][-1]


def test_initial_boundary_gain_does_not_survive_extended_robustness_run():
    first = load_report(P0560)["primary"]
    robust = load_report(P0561)["primary"]
    assert first["best_common_coupling"] == -1.0
    assert np.isclose(first["best_common_improvement_fraction"], 0.027420500331318598)
    assert robust["best_common_coupling"] == 0.0
    assert robust["best_nonzero_common_coupling"] == -1.0
    assert np.isclose(
        robust["best_nonzero_common_improvement_fraction"],
        -0.028308782867017213,
    )


def test_extended_range_is_elliptic_but_most_common_points_lose_roots():
    scores = pd.read_csv(P0561 / "scores.csv")
    assert scores.minimum_permittivity_eigenvalue.min() > 0.10
    complete = (
        scores.groupby("coupling").all_heldout_roots.all().loc[lambda value: value]
    )
    assert set(complete.index) == {-1.0, 0.0}
    assert scores.fit_success.all()


def test_cluster_signs_and_grid_optima_are_not_universal():
    response = pd.read_csv(P0561 / "response_summary.csv").set_index("system_label")
    assert response.near_zero_preferred_sign.value_counts().to_dict() == {
        "negative": 2,
        "positive": 2,
    }
    assert response.best_grid_coupling.to_dict() == {
        "MACS0329": -1.0,
        "MACS0429": -2.0,
        "MACS1115": 4.0,
        "MACS1931": -1.0,
    }
    assert np.isclose(
        response.loc["MACS1115", "best_grid_improvement_fraction"],
        0.402390,
        atol=1e-6,
    )


def test_leave_one_cluster_out_transfer_does_not_improve():
    loo = pd.read_csv(P0561 / "leave_one_out.csv").set_index("heldout_system")
    assert not load_report(P0561)["primary"]["leave_one_out_all_improve"]
    assert loo.loc["MACS0429", "chosen_coupling_from_other_three"] == -1.0
    assert np.isclose(
        loo.loc["MACS0429", "heldout_improvement_fraction_vs_zero"],
        -0.311619,
        atol=1e-6,
    )
    assert (loo.drop(index="MACS0429").chosen_coupling_from_other_three == 0.0).all()


def test_optimizer_basin_change_can_reverse_shared_point_claim():
    first = pd.read_csv(P0560 / "scores.csv")
    robust = pd.read_csv(P0561 / "scores.csv")
    old = first[first.coupling.eq(-1.0)].set_index("system_label")
    new = robust[robust.coupling.eq(-1.0)].set_index("system_label")
    assert new.loc["MACS0429", "fit_cost"] < old.loc["MACS0429", "fit_cost"]
    assert new.loc["MACS0429", "heldout_exact_RMS_arcsec"] > old.loc[
        "MACS0429", "heldout_exact_RMS_arcsec"
    ] + 4.9
    assert not load_report(P0561)["verdict"]["common_sign_supported"]
    assert not load_report(P0561)["verdict"]["formula_promoted"]
