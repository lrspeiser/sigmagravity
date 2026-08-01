from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rxj2129_X2b_protocol_is_frozen_without_gravity_authorization() -> None:
    config = json.loads(
        (ROOT / "configs/r1_rxj2129_xmm_background_mask_protocol.json").read_text()
    )
    assert config["status"] == (
        "completed_X2b2_with_MOS2_and_pn_before_X3_annular_construction"
    )
    assert config["protocol_version"].endswith("1.3")
    emanom_correction = next(
        item for item in config["correction_log"] if item["version"] == "0.5"
    )
    assert emanom_correction["data_driven_field"] == (
        "only the predeclared emanom CCD anomaly decision"
    )
    assert config["correction_log"][-1]["data_driven"] is False
    assert config["correction_log"][-1]["invalid_partial_products_admitted"] is False
    assert config["correction_log"][-1]["annulus_counts_or_scale_inspected"] is False
    assert config["mask_application_to_ESAS_spectra"]["withsrcrem"] is True
    assert config["MOS_anomaly_gate"]["frozen_result_after_valid_corner_run"]["MOS2_excluded_CCD"] == 5
    assert config["source_detection"]["bands_eV"] == [
        [500, 1200],
        [1200, 2000],
        [2000, 7000],
    ]
    assert config["source_detection"]["cheese_parameters"]["mlmin"] == 10.0
    assert config["immutable_mask"]["radius_arcsec"] == (
        "max(15,min(60,1.5*maximum_local_r80_arcsec))"
    )
    assert config["background"]["local_outer_annulus"]["kpc"] == [650.0, 900.0]
    assert config["background"]["FWC_corner_scale_allowed_open_interval"] == [0.5, 2.0]
    assert config["background"]["FWC_corner_scale_definition"]["PI_channels_inclusive"] == [
        101,
        1400,
    ]
    assert config["background"]["FWC_corner_subgate_result"]["passing_instruments"] == [
        "MOS2",
        "pn",
    ]
    local_transfer = config["background"]["local_outer_annulus_transfer_scale_definition"]
    assert local_transfer["frozen_before_annulus_extraction"] is True
    assert local_transfer["MOS_energy_band_eV"] == [9500.0, 11500.0]
    assert local_transfer["pn_energy_band_eV"] == [10000.0, 12000.0]
    assert local_transfer["minimum_observed_hard_band_counts"] == 25
    assert config["background"]["local_outer_annulus_subgate_result"][
        "passing_instruments"
    ] == ["MOS2", "pn"]
    assert config["background"]["local_outer_annulus_subgate_result"][
        "full_X2b2_background_gate_passed"
    ] is True
    assert config["authorization"]["construct_X3_annular_count_response_products"] is True
    assert config["authorization"]["fit_temperature_or_density_before_full_X2_pass"] is False
    assert config["authorization"]["fit_temperature_or_density_before_X3_adequacy_pass"] is False
    assert config["authorization"]["infer_dynamical_or_Weyl_response"] is False
    assert config["authorization"]["fit_new_force_or_action"] is False
