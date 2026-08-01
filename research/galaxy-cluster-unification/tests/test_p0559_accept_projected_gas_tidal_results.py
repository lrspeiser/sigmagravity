import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/p0559_accept_projected_gas_tidal"


def report():
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def test_protocol_was_frozen_and_adds_no_gravity_parameters():
    protocol = json.loads(
        (ROOT / "configs/p0559_accept_projected_gas_tidal_protocol.json").read_text()
    )
    assert protocol["status"].startswith("frozen_before_any_")
    assert protocol["locked_field"]["tensor_t"] == 0.3
    assert protocol["locked_field"]["operator"] == "contrast"
    assert protocol["locked_field"]["new_gravity_parameters"] == 0
    assert protocol["locked_field"]["per_cluster_gravity_parameters"] == 0
    assert [m["model_id"] for m in protocol["exact_models"]] == [
        "zero",
        "accept_absolute_sqrt",
        "tian_anchor_sqrt",
    ]


def test_accept_projection_and_tian_anchor_are_incompatible():
    audits = pd.read_csv(RESULTS / "physical_map_audits.csv").set_index(
        "system_label"
    )
    assert set(audits.index) == {"MACS0329", "MACS0429", "MACS1115", "MACS1931"}
    ratios = audits["ACCEPT_to_Tian_anchor_mass_ratio"]
    assert np.allclose(
        ratios.loc[["MACS0329", "MACS0429", "MACS1115", "MACS1931"]],
        [0.0838392643, 0.0693237901, 0.0732446397, 0.2371451717],
    )
    assert (ratios < 0.5).all()
    assert (audits["absolute_projected_map_gas_fraction"] > 0.94).all()


def test_physical_accept_primary_worsens_all_cluster_score():
    primary = report()["primary"]
    assert primary["all_roots"]
    assert not primary["all_systems_improve"]
    assert np.isclose(primary["zero_all_four_RMS_arcsec"], 17.88096773689163)
    assert np.isclose(
        primary["absolute_ACCEPT_all_four_RMS_arcsec"], 18.132481246288595
    )
    assert np.isclose(primary["improvement_fraction"], -0.014065989777390264)


def test_response_sign_is_cluster_specific():
    scores = pd.read_csv(RESULTS / "scores.csv")
    systems = scores[scores.row_type.eq("system")]
    zero = systems[systems.model_id.eq("zero")].set_index("system_label")
    primary = systems[systems.model_id.eq("accept_absolute_sqrt")].set_index(
        "system_label"
    )
    for label in ["MACS0329", "MACS0429"]:
        assert primary.loc[label, "heldout_exact_RMS_arcsec"] > zero.loc[
            label, "heldout_exact_RMS_arcsec"
        ]
    for label in ["MACS1115", "MACS1931"]:
        assert primary.loc[label, "heldout_exact_RMS_arcsec"] < zero.loc[
            label, "heldout_exact_RMS_arcsec"
        ]


def test_cross_catalog_amplitude_change_does_not_repair_direction():
    scores = pd.read_csv(RESULTS / "scores.csv")
    aggregates = scores[
        scores.row_type.eq("aggregate") & scores.system_label.eq("all_four")
    ].set_index("model_id")
    accept = aggregates.loc["accept_absolute_sqrt", "heldout_exact_RMS_arcsec"]
    tian = aggregates.loc["tian_anchor_sqrt", "heldout_exact_RMS_arcsec"]
    assert abs(tian - accept) < 0.02
    assert tian > aggregates.loc["zero", "heldout_exact_RMS_arcsec"]


def test_advancement_gates_fail_without_a_solver_failure():
    result = report()
    assert result["gate_audit"]["absolute_primary_all_roots"]
    assert not result["gate_audit"]["absolute_primary_all_systems_improve"]
    assert not result["gate_audit"]["absolute_primary_improvement_pass"]
    assert not result["gate_audit"]["compact_halo_ratio_pass"]
    assert not result["gate_audit"]["accept_to_tian_anchor_mass_compatibility_pass"]
    assert not result["verdict"]["all_advancement_gates_pass"]
    assert not result["verdict"]["formula_promoted"]
