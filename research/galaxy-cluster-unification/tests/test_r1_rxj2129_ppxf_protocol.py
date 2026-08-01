from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_protocol_is_structurally_preselected_and_blind() -> None:
    config = json.loads(
        (ROOT / "configs/r1_rxj2129_ppxf_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    structural = config["structural_preselection"]
    assert structural["spectroscopic_images_inside_5_arcsec"] == 3
    assert structural["source_families_inside_5_arcsec"] == 3
    assert structural["family_wide_position_dof_after_source_coordinates"] == 12
    assert structural["conditional_radial_rank_upper_bound"] == 3
    assert config["spatial_extraction"]["annulus_semimajor_edges_arcsec"] == [
        0.0,
        0.6,
        1.5,
        3.0,
        5.0,
    ]
    assert config["spatial_extraction"]["axis_ratio_b_over_a"] == 1.0
    assert config["authorization"]["structural_promotion_after_baseline_pass"]
    assert not config["authorization"]["gravity_response_fit"]


def test_rxj2129_readiness_gate_blocks_gravity_until_inputs_are_complete() -> None:
    config = json.loads(
        (ROOT / "configs/r1_rxj2129_readiness_targets.json").read_text(
            encoding="utf-8"
        )
    )
    current = config["current_state"]
    lens = config["workstreams"]["observable_lens_likelihood"]
    authorization = config["authorization"]

    assert current["structural_promotion"]
    assert current["four_bin_kinematic_baseline_internal_consistency"]
    assert not current["strict_r1_ready"]
    assert lens["fixed_observables"]["spectroscopic_image_positions"] == 21
    assert lens["fixed_observables"]["strict_images_inside_5_arcsec"] == 3
    assert lens["advance_thresholds"]["image_plane_rms_arcsec_maximum"] == 0.5
    assert authorization["nuisance_marginalized_jacobian_after_all_readiness_gates_pass"]
    assert not authorization["jacobian_before_all_readiness_gates_pass"]
    assert not authorization["gravity_response_fit"]
