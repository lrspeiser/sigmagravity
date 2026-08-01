from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "p0635_ddo154_map_commissioning"
SENSITIVITY = ROOT / "results" / "p0635_ddo154_map_geometry_sensitivity"


def load_report(path: Path) -> dict:
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_commissioning_preserves_the_sealed_target_boundary():
    report = load_report(BASELINE)
    assert report["status"] == "commissioned"
    assert report["galaxy"] == "DDO154"
    assert report["data_boundary"]["little_things_velocity_products_downloaded"] is False
    assert report["data_boundary"]["little_things_cube_downloaded"] is False
    assert report["data_boundary"]["p0633_target_products_opened"] is False
    assert all(product["verified"] for product in report["raw_provenance"])


def test_real_map_conserves_mass_and_is_gas_dominated():
    inventory = load_report(BASELINE)["mass_inventory_solar"]
    assert 0.98 <= inventory["gas_grid_fraction_of_raw"] <= 1.02
    assert np.isclose(inventory["gridded_stars"], 23663761.610801324)
    assert inventory["gas_fraction"] > 0.9
    with np.load(BASELINE / "baryonic_maps.npz") as maps:
        assert maps["gas_surface_density_solar_kpc2"].shape == (65, 65)
        assert np.all(maps["total_surface_density_solar_kpc2"] >= 0.0)


def test_every_real_map_field_equation_converges():
    solvers = load_report(BASELINE)["field_solvers"]
    assert set(solvers) == {"newtonian", "QUMOND", "AQUAL"}
    assert all(result["converged"] for result in solvers.values())
    assert solvers["newtonian"]["normalized_residual_RMS"] < 1e-5
    assert solvers["QUMOND"]["normalized_residual_RMS"] < 1e-5
    assert solvers["AQUAL"]["normalized_residual_RMS"] < 1e-5


def test_spent_galaxy_score_is_diagnostic_and_mond_fields_close_most_of_gap():
    scores = load_report(BASELINE)["spent_DDO154_rotation_scores"]
    assert scores["newtonian_3d_map"]["RMSE_km_s"] > 20.0
    assert scores["QUMOND_3d_map"]["RMSE_km_s"] < 5.0
    assert scores["AQUAL_3d_map"]["RMSE_km_s"] < 5.0
    assert scores["algebraic_simple_mond"]["RMSE_km_s"] < scores["QUMOND_3d_map"]["RMSE_km_s"]


def test_geometry_ablation_is_complete_and_does_not_hide_field_difference():
    report = load_report(SENSITIVITY)
    assert report["status"] == "complete"
    assert len(report["variants"]) == 5
    diagnostics = report["diagnostics"]
    assert abs(diagnostics["QUMOND_axisymmetry_RMSE_change_km_s"]) < 0.2
    assert abs(diagnostics["QUMOND_razor_minus_baseline_RMSE_km_s"]) < 0.2
    assert diagnostics["axisymmetric_QUMOND_minus_algebraic_RMSE_km_s"] > 0.5
    scores = pd.read_csv(SENSITIVITY / "geometry_scores.csv")
    assert scores["converged"].all()
    assert (SENSITIVITY / "geometry_sensitivity.png").stat().st_size > 10000
