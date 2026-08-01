import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/p0558_mass_anchor_proxy_tidal"


def report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_locks_gravity_before_mass_anchor_scores():
    protocol = json.loads(
        (ROOT / "configs/p0558_mass_anchor_proxy_tidal_protocol.json").read_text()
    )
    assert protocol["status"].startswith("frozen_before_any_")
    assert protocol["locked_field"]["tensor_t"] == 0.3
    assert protocol["locked_field"]["subtract_circular_mean"] is True
    assert protocol["locked_field"]["new_gravity_parameters"] == 0
    assert protocol["locked_field"]["per_cluster_gravity_parameters"] == 0
    assert len(protocol["models"]) == 5


def test_published_mass_anchors_produce_expected_object_inputs():
    anchors = pd.read_csv(RESULTS / "mass_anchors.csv").set_index("system_label")
    assert set(anchors.index) == {"MACS0329", "MACS0429", "MACS1115", "MACS1931"}
    expected = {
        "MACS0329": 0.6073249884098284,
        "MACS0429": 0.3605588393336916,
        "MACS1115": 0.6590909090909091,
        "MACS1931": 0.17520858164481523,
    }
    for system, value in expected.items():
        assert np.isclose(anchors.loc[system, "nominal_gas_fraction"], value)
        assert anchors.loc[system, "conservative_gas_fraction_low"] < value
        assert anchors.loc[system, "conservative_gas_fraction_high"] > value


def test_measured_mass_primary_worsens_four_cluster_aggregate():
    primary = report()["primary"]
    assert primary["all_heldout_roots_converged"]
    assert not primary["all_four_systems_improve"]
    assert np.isclose(primary["zero_all_four_RMS_arcsec"], 17.89236924269774)
    assert np.isclose(
        primary["measured_sqrt_all_four_RMS_arcsec"], 18.164777807776527
    )
    assert np.isclose(primary["improvement_fraction"], -0.015224845932014563)


def test_effect_is_cluster_specific_and_low_gas_sensitivity_is_least_bad():
    scores = pd.read_csv(RESULTS / "scores.csv")
    systems = scores[scores.row_type.eq("system")]
    zero = systems[systems.model_id.eq("zero")].set_index("system_label")
    primary = systems[systems.model_id.eq("measured_sqrt")].set_index("system_label")
    assert primary.loc["MACS0329", "heldout_exact_RMS_arcsec"] > zero.loc[
        "MACS0329", "heldout_exact_RMS_arcsec"
    ]
    assert primary.loc["MACS0429", "heldout_exact_RMS_arcsec"] > zero.loc[
        "MACS0429", "heldout_exact_RMS_arcsec"
    ]
    assert primary.loc["MACS1115", "heldout_exact_RMS_arcsec"] < zero.loc[
        "MACS1115", "heldout_exact_RMS_arcsec"
    ]
    assert primary.loc["MACS1931", "heldout_exact_RMS_arcsec"] < zero.loc[
        "MACS1931", "heldout_exact_RMS_arcsec"
    ]
    aggregates = scores[
        scores.row_type.eq("aggregate") & scores.system_label.eq("all_four")
    ].set_index("model_id")
    nonzero = aggregates.drop(index="zero")
    assert nonzero.heldout_exact_RMS_arcsec.idxmin() == "half_gas_sqrt"
    assert (nonzero.heldout_exact_RMS_arcsec > aggregates.loc["zero", "heldout_exact_RMS_arcsec"]).all()


def test_brightness_transform_is_not_identified_and_large_gas_loses_root():
    scores = pd.read_csv(RESULTS / "scores.csv")
    aggregates = scores[
        scores.row_type.eq("aggregate") & scores.system_label.eq("all_four")
    ].set_index("model_id")
    assert abs(
        aggregates.loc["measured_linear", "heldout_exact_RMS_arcsec"]
        - aggregates.loc["measured_sqrt", "heldout_exact_RMS_arcsec"]
    ) < 0.02
    assert not bool(aggregates.loc["double_gas_sqrt", "all_heldout_roots"])


def test_advancement_and_compact_halo_gates_fail():
    result = report()
    assert np.isclose(result["comparators"]["ratio"], 1.8386383408501363)
    assert not result["gate_audit"]["primary_all_four_systems_improve"]
    assert not result["gate_audit"]["primary_improvement_pass"]
    assert not result["gate_audit"]["compact_halo_ratio_pass"]
    assert not result["verdict"]["all_advancement_gates_pass"]
    assert not result["verdict"]["formula_promoted"]
    assert result["verdict"]["X5_projected_gas_mass_still_required"]
