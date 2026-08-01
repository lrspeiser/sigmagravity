import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_protocol_freezes_the_universal_grid_and_predictive_split():
    protocol = _load("configs/clash_stellar_morphology_response_protocol.json")
    grid = protocol["factor_grid"]
    split = protocol["systems_and_split"]

    assert protocol["protocol_version"] == "CLASH-STELLAR-MORPHOLOGY-RESPONSE-0.2.0"
    assert protocol["status"] == "frozen_before_light_template_construction_or_lens_scores"
    assert grid["carrier"] == ["baryonic", "extra", "full"]
    assert grid["smoothing_kpc"] == [5.0, 10.0, 20.0]
    assert grid["contrast_cap"] == [2.0, 5.0, 20.0]
    assert grid["redistribution_fraction"] == [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
    assert split["selection_labels"] == ["MACS0329", "MACS0429"]
    assert split["validation_labels"] == ["MACS1115", "MACS1931"]
    assert protocol["formula"]["new_object_specific_gravity_parameters"] == 0


def test_map_audit_covers_every_frozen_nonzero_shape_cell_and_passes():
    protocol = _load("configs/clash_stellar_morphology_response_protocol.json")
    report = _load("results/clash_stellar_morphology_response/map_audit_report.json")
    rows = pd.read_csv(ROOT / "results/clash_stellar_morphology_response/map_audit.csv")
    expected = (
        4
        * len(protocol["parent_radial_laws"])
        * len(protocol["factor_grid"]["carrier"])
        * len(protocol["factor_grid"]["smoothing_kpc"])
        * len(protocol["factor_grid"]["contrast_cap"])
    )

    assert report["status"] == "all frozen map-construction audits passed"
    assert report["fields"] == expected == len(rows) == 324
    assert set(rows["label"]) == {"MACS0329", "MACS0429", "MACS1115", "MACS1931"}
    assert set(rows["parent"]) == set(protocol["parent_radial_laws"])


def test_map_audit_stays_inside_every_frozen_numerical_gate():
    protocol = _load("configs/clash_stellar_morphology_response_protocol.json")
    report = _load("results/clash_stellar_morphology_response/map_audit_report.json")
    limits = protocol["numerical_audits"]
    maxima = report["maximum_audits"]

    assert maxima["maximum_carrier_weighted_annular_mean_error"] <= limits[
        "maximum_annular_weight_mean_error"
    ]
    assert maxima["maximum_annular_convergence_mean_fraction"] <= limits[
        "maximum_annular_convergence_mean_fraction"
    ]
    assert maxima["maximum_independent_circular_mean_deflection_arcsec"] <= limits[
        "maximum_independent_circular_mean_deflection_arcsec"
    ]
    assert maxima["normalized_curl_RMS"] <= limits["maximum_normalized_curl_RMS"]
    assert maxima["maximum_edge_delta_convergence"] <= limits[
        "maximum_edge_delta_convergence"
    ]
    assert all(
        item["valid_fraction_within_60_arcsec"] == 1.0
        for item in report["light_preprocessing"]
    )
