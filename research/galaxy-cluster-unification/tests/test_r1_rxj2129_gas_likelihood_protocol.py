import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gas_likelihood_protocol_is_frozen_before_calibrated_inspection() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_gas_likelihood_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == (
        "frozen_before_calibrated_reduction_product_or_gas_fit_inspection"
    )
    assert protocol["observations"]["obsids"] == [552, 9370]
    assert protocol["observations"]["fit_separately_not_coadded"] is True
    assert len(protocol["fixed_geometry"]["training_sector_indices_zero_based"]) == 8
    assert len(protocol["fixed_geometry"]["heldout_sector_indices_zero_based"]) == 4
    assert protocol["density_models"]["published_mass_anchor_used_in_fit"] is False
    assert protocol["density_models"]["gravity_or_lens_residual_used_in_fit"] is False


def test_gas_protocol_maps_one_density_to_both_mass_responses() -> None:
    protocol = json.loads(
        (ROOT / "configs/r1_rxj2129_gas_likelihood_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    projection = protocol["projection_and_mass"]
    assert "same selected ne posterior" in projection["projected_gas_mass"]
    assert "M3D" in projection["covariance"]
    assert "M2D" in projection["covariance"]
    assert protocol["authorization"]["gas_likelihood_after_calibrated_reduction_pass"] is True
    assert protocol["authorization"]["use_published_mass_anchor_as_fit_constraint"] is False
    assert protocol["authorization"]["gravity_response_fit"] is False
    assert protocol["authorization"]["weyl_response_reconstruction"] is False
    assert protocol["authorization"]["new_force_or_action_fit"] is False
    assert protocol["authorization"]["strict_r1_ready"] is False
