from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_rxj2129_x2b2_stage_boundary_is_consistent_and_narrow() -> None:
    manifest = load("data/derived/r1_rxj2129_xmm_reduction_manifest.json")
    report = load("results/r1_rxj2129_xmm_event_processing/report.json")
    protocol = load("configs/r1_rxj2129_xmm_background_mask_protocol.json")
    next_stage = load("configs/r1_rxj2129_strict_observable_next_stage.json")
    targets = load("configs/r1_execution_targets.json")

    assert manifest["manifest_version"].endswith("X2b2-1.0")
    assert manifest["X2b2_background"]["passing_instruments"] == ["MOS2", "pn"]
    assert manifest["X2b2_background"]["invalid_partial_products_admitted"] is False
    assert manifest["gates"]["R1B3_XMM_X2_flare_background_gate_passed"] is True
    assert manifest["gates"]["R1B3_XMM_X3_gas_likelihood_gate_passed"] is False

    assert report["status"] == "pass"
    assert report["passing_instruments"] == ["MOS2", "pn"]
    assert report["authorization"]["construct_X3_annular_count_response_products"] is True
    assert report["authorization"]["fit_temperature_or_density"] is False
    assert report["authorization"]["infer_dynamical_or_Weyl_response"] is False

    assert protocol["status"] == (
        "completed_X2b2_with_MOS2_and_pn_before_X3_annular_construction"
    )
    assert protocol["background"]["local_outer_annulus_subgate_result"][
        "full_X2b2_background_gate_passed"
    ] is True
    assert protocol["authorization"]["construct_X3_annular_count_response_products"] is True
    assert protocol["authorization"]["fit_temperature_or_density_before_X3_adequacy_pass"] is False

    assert next_stage["execution_status"][
        "XMM_X3_annular_count_response_adequacy"
    ].startswith("pass: 6/6 annuli")
    assert next_stage["execution_status"][
        "XMM_X4_PSF_cross_region_response_matrix"
    ].startswith("active: full 12 RMF")
    assert next_stage["execution_status"][
        "HST_42x42_measurement_covariance"
    ].startswith("active H2 measurement")
    assert next_stage["authorization"]["fit_XMM_temperature_or_density"] is False
    assert targets["baseline"]["rxj2129_xmm_X2b2_background_gate_pass"] is True
    assert targets["baseline"]["rxj2129_xmm_X3_gas_likelihood_gate_pass"] is False
