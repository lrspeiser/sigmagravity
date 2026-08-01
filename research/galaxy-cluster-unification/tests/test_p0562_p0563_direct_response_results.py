import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P0562 = ROOT / "results/p0562_accept_tensor_direct_response"
P0563 = ROOT / "results/p0563_accept_tensor_source_plane_response"


def load_report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_protocols_freeze_distinct_response_questions():
    local = json.loads(
        (ROOT / "configs/p0562_accept_tensor_direct_response_protocol.json").read_text()
    )
    source = json.loads(
        (ROOT / "configs/p0563_accept_tensor_source_plane_response_protocol.json").read_text()
    )
    assert local["status"].startswith("frozen_before_any_")
    assert source["status"].startswith("frozen_after_p0562_")
    assert "no inverse lens root" in local["response_grid"]["metric"]
    assert source["geometry"]["refit"] is False
    assert "never invert" in source["geometry"]["reason"]
    assert local["locked_map"]["new_map_parameters"] == 0
    assert source["locked_map"]["per_cluster_parameters"] == 0


def test_local_jacobian_metric_produces_a_large_but_nonuniversal_gain():
    report = load_report(P0562)
    ensembles = pd.DataFrame(report["ensemble_summary"]).set_index("ensemble")
    assert (ensembles.best_common_coupling == -3.75).all()
    assert np.isclose(
        ensembles.loc["seed_1", "best_common_improvement_fraction"],
        0.8684711045884683,
    )
    assert np.isclose(
        ensembles.loc["seed_2", "best_common_improvement_fraction"],
        0.87771131192628,
    )
    assert report["primary"]["near_zero_sign_agreement_between_seed_ensembles"]
    assert not report["primary"]["all_systems_share_one_near_zero_sign"]


def test_conditioning_explains_the_apparent_local_gain():
    conditioning = load_report(P0563)["conditioning"]
    assert conditioning[
        "log_inverse_gain_vs_log_local_to_source_ratio_correlation"
    ] > 0.87
    assert conditioning["maximum_inverse_jacobian_gain"] > 3900.0
    assert conditioning["maximum_local_to_source_plane_RMS_ratio"] > 200.0
    audit = pd.read_csv(P0563 / "conditioning_audit.csv")
    macs1931_zero = audit[
        audit.system_label.eq("MACS1931") & audit.coupling.eq(0.0)
    ]
    assert (macs1931_zero.heldout_max_inverse_jacobian_gain > 330.0).all()
    assert (macs1931_zero.local_to_source_plane_RMS_ratio > 60.0).all()


def test_conditioning_robust_common_response_is_only_two_tenths_percent():
    report = load_report(P0563)
    ensembles = pd.DataFrame(report["ensemble_summary"]).set_index("ensemble")
    assert ensembles.loc["seed_1", "best_common_coupling"] == 2.5
    assert ensembles.loc["seed_2", "best_common_coupling"] == 2.75
    assert np.isclose(
        ensembles.loc["seed_1", "best_common_improvement_fraction"],
        0.0016873083386534926,
    )
    assert np.isclose(
        ensembles.loc["seed_2", "best_common_improvement_fraction"],
        0.001783634736838069,
    )
    assert not report["primary"][
        "common_optimum_agreement_between_geometry_ensembles"
    ]


def test_source_plane_signs_are_stable_but_not_universal():
    response = pd.read_csv(P0563 / "per_system_summary.csv")
    signs = response.pivot(
        index="system_label", columns="ensemble", values="near_zero_preferred_sign"
    )
    assert (signs.seed_1 == signs.seed_2).all()
    assert signs.seed_1.to_dict() == {
        "MACS0329": "positive",
        "MACS0429": "negative",
        "MACS1115": "positive",
        "MACS1931": "positive",
    }
    report = load_report(P0563)
    assert report["primary"]["near_zero_sign_agreement_between_geometry_ensembles"]
    assert not report["primary"]["all_systems_share_one_near_zero_sign"]
    assert not report["verdict"]["formula_promoted"]
