from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_xmm_gas_protocol_promotes_only_response_calibration_after_X3() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_xmm_gas_likelihood_protocol.json").read_text()
    )
    assert protocol["prerequisites"]["required_X3_status"] == "pass"
    assert protocol["prerequisites"]["required_passing_annuli"] == 6
    assert protocol["X4_response_calibration"]["direct_responses"]["required_RMFs"] == 12
    assert (
        protocol["X4_response_calibration"]["cross_region_responses"]
        ["required_ARFs_total"]
        == 72
    )
    assert (
        protocol["X4_response_calibration"]["central_unresolved_source_response"]
        ["required_point_to_output_ARFs"]
        == 12
    )
    authorization = protocol["authorization"]
    assert authorization["construct_X4_direct_and_cross_region_responses"] is True
    assert authorization["fit_X5_temperature_or_density_before_X4_pass"] is False
    assert authorization["infer_dynamical_or_Weyl_response"] is False
    assert authorization["fit_new_force_or_action"] is False


def test_xmm_gas_protocol_keeps_backgrounds_and_gravity_out_of_gas_fit() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_xmm_gas_likelihood_protocol.json").read_text()
    )
    likelihood = protocol["X5_joint_spectral_likelihood"]
    assert "un-subtracted" in likelihood["data_model"]
    assert set(likelihood["backgrounds"]) == {
        "QPB",
        "soft_protons",
        "fluorescence",
        "cosmic_and_galactic_sky",
        "SWCX",
        "outer_cluster_rule",
    }
    forbidden = set(likelihood["hot_gas"]["forbidden"])
    assert "hydrostatic equilibrium" in forbidden
    assert "a lens model or lens residual" in forbidden
    assert "a void field or negative-gravity prediction" in forbidden
    acceptance = likelihood["acceptance"]
    assert acceptance["minimum_accepted_annuli"] == 5
    assert acceptance["maximum_fractional_density_uncertainty_each_accepted_annulus"] == 0.2
    assert acceptance["maximum_fractional_temperature_uncertainty_each_accepted_annulus"] == 0.3
