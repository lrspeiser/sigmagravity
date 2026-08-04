from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.sigma_boundary_inference import (
    decompose_boundary_shear,
    fit_harmonic_shear,
    harmonic_shear_basis,
    radial_taper,
    shear_alignment_and_power_closed,
)
from voidscreen.sigma_covariant_feature_inference import convergence_to_shear

ROOT = Path(__file__).resolve().parents[1]


def test_radial_taper_has_exact_plateaus_and_smooth_transition() -> None:
    radius = np.array([[0.0, 1.0, 1.5, 2.0, 3.0]])
    taper = radial_taper(radius, start=1.0, end=2.0)
    np.testing.assert_allclose(taper, [[1.0, 1.0, 0.5, 0.0, 0.0]], atol=1.0e-15)


def test_harmonic_basis_contains_exact_uniform_shear_modes() -> None:
    axis = np.linspace(-2.0, 2.0, 21)
    east, north = np.meshgrid(axis, axis)
    basis = harmonic_shear_basis(
        east,
        north,
        minimum_order=2,
        maximum_order=4,
        reference_radius_kpc=2.0,
    )
    np.testing.assert_allclose(basis["harmonic_m2_cos"][0], 1.0)
    np.testing.assert_allclose(basis["harmonic_m2_cos"][1], 0.0)
    np.testing.assert_allclose(basis["harmonic_m2_sin"][0], 0.0)
    np.testing.assert_allclose(basis["harmonic_m2_sin"][1], 1.0)


def test_harmonic_fit_recovers_known_mixed_orders() -> None:
    axis = np.linspace(-3.0, 3.0, 61)
    east, north = np.meshgrid(axis, axis)
    radius = np.hypot(east, north)
    basis = harmonic_shear_basis(
        east,
        north,
        minimum_order=2,
        maximum_order=5,
        reference_radius_kpc=2.5,
    )
    coefficients = {
        "harmonic_m2_cos": 0.23,
        "harmonic_m2_sin": -0.11,
        "harmonic_m4_cos": 0.07,
        "harmonic_m5_sin": -0.03,
    }
    shear_1 = np.zeros_like(east)
    shear_2 = np.zeros_like(east)
    for name, coefficient in coefficients.items():
        shear_1 += coefficient * basis[name][0]
        shear_2 += coefficient * basis[name][1]
    fit = fit_harmonic_shear(shear_1, shear_2, radius < 2.4, basis)
    assert fit.normalized_RMSE < 1.0e-12
    assert fit.power_closed > 1.0 - 1.0e-12
    for name, coefficient in coefficients.items():
        np.testing.assert_allclose(fit.coefficients[name], coefficient, atol=1.0e-12)


def test_boundary_decomposition_recovers_external_uniform_shear() -> None:
    axis = np.linspace(-3.0, 3.0, 81)
    east, north = np.meshgrid(axis, axis)
    radius = np.hypot(east, north)
    convergence = np.exp(-(east**2 + north**2) / 0.4) * (radius < 1.5)
    internal_1, internal_2 = convergence_to_shear(convergence, padding_factor=3)
    missing_1 = internal_1 + 0.18
    missing_2 = internal_2 - 0.09
    basis = harmonic_shear_basis(
        east,
        north,
        minimum_order=2,
        maximum_order=5,
        reference_radius_kpc=2.5,
    )
    mask = radius < 2.5
    result = decompose_boundary_shear(
        convergence,
        missing_1,
        missing_2,
        radius,
        mask,
        basis,
        taper_start_kpc=2.0,
        taper_end_kpc=2.5,
        padding_factor=3,
    )
    assert result.harmonic_fit.power_closed > 1.0 - 1.0e-12
    np.testing.assert_allclose(
        result.harmonic_fit.coefficients["harmonic_m2_cos"],
        0.18,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.harmonic_fit.coefficients["harmonic_m2_sin"],
        -0.09,
        atol=1.0e-12,
    )
    score = shear_alignment_and_power_closed(
        result.harmonic_fit.predicted_shear_1,
        result.harmonic_fit.predicted_shear_2,
        result.boundary_shear_1,
        result.boundary_shear_2,
        mask,
    )
    assert score["boundary_shear_alignment_cosine"] > 1.0 - 1.0e-12
    assert score["boundary_shear_power_closed"] > 1.0 - 1.0e-12


def test_completed_v16_report_obeys_frozen_boundary_protocol() -> None:
    path = ROOT / "results" / "sigma_v16_spent_boundary_decomposition" / "report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["protocol_version"] == "SIGMA-V16-SPENT-BOUNDARY-DECOMPOSITION-1.0.0"
    assert report["sample_is_spent"] is True
    assert report["observational_validation_claim"] is False
    assert report["per_cluster_gravity_parameters"] == 0
    assert report["per_cluster_shear_or_orientation_parameters"] == 0
    assert report["one_metric_feature_triplets"] is True
    assert len(report["decomposition_results"]) == 2
    assert set(report["family_results"]) == {
        "internal_only",
        "boundary_total",
        "boundary_components",
    }
    assert report["selected_family"] in report["family_results"]
    assert report["best_boundary_family"] in {
        "boundary_total",
        "boundary_components",
    }
    assert report["decomposition_integrity"]["passes_declared_separation"] is False
    assert report["decomposition_integrity"]["taper_start_minus_analysis_radius_kpc"] < 0.0

    replacement_path = ROOT / "results" / "sigma_v16b_spent_boundary_interior" / "report.json"
    if replacement_path.exists():
        replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
        assert replacement["protocol_version"] == ("SIGMA-V16B-SPENT-BOUNDARY-INTERIOR-1.0.0")
        assert replacement["decomposition_integrity"]["passes_declared_separation"] is True
        assert replacement["gate_results"]["harmonic_oracle_each_cluster"] is False
        assert replacement["gate_results"]["measured_outer_baryon_boundary_source"] is False
        rows = {row["cluster"]: row for row in replacement["decomposition_results"]}
        np.testing.assert_allclose(
            rows["AS295"]["harmonic_oracle_power_closed"],
            0.619390612164938,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            rows["PLCKG287"]["harmonic_oracle_power_closed"],
            0.3729131095648175,
            atol=1.0e-12,
        )

    order_path = ROOT / "results" / "sigma_v16c_harmonic_order_sensitivity" / "report.json"
    if order_path.exists():
        order = json.loads(order_path.read_text(encoding="utf-8"))
        assert order["gate_results"]["both_clusters_half_power_at_m12"] is False
        assert order["gate_results"]["PLCKG287_material_order_gain"] is False
        assert order["PLCKG287_power_gain_m6_to_m12"] < 0.01

    incremental_path = ROOT / "results" / "sigma_v16d_incremental_boundary_control" / "report.json"
    if incremental_path.exists():
        incremental = json.loads(incremental_path.read_text(encoding="utf-8"))
        assert incremental["gate_results"]["zero_boundary_limit_reproduces_internal"] is True
        assert incremental["gate_results"]["advance"] is False
        np.testing.assert_allclose(
            incremental["relative_boundary_improvement"],
            0.0050247806908245846,
            atol=1.0e-14,
        )


def test_v17_dynamical_stress_gate_requires_matched_data_and_transfer() -> None:
    path = ROOT / "configs" / "sigma_v17_dynamical_stress_data_gate.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["protocol_version"] == "SIGMA-V17-DYNAMICAL-STRESS-DATA-GATE-1.0.0"
    assert config["sample"]["sample_is_spent"] is True
    assert config["sample"]["clusters"] == ["AS295", "PLCKG287"]
    assert config["frozen_fit_rule"]["per_cluster_coefficients"] is False
    assert config["frozen_fit_rule"]["per_cluster_amplitude_scale_shear_or_orientation"] is False
    assert config["frozen_fit_rule"]["one_metric_feature_triplets"] is True
    assert (
        config["stage_A_thermal_stress"]["required_for_each_cluster"][
            "minimum_independent_temperature_regions_inside_350_kpc"
        ]
        >= 12
    )
    assert (
        config["stage_B_collisionless_member_stress"]["required_for_each_cluster"][
            "minimum_secure_members_inside_1_8_Mpc"
        ]
        >= 50
    )
    assert config["diagnostic_gates"]["minimum_relative_improvement_over_static_baseline"] == 0.1
    assert (
        config["inventory_at_freeze"]["PLCKG287"][
            "secure_member_velocity_catalog_publicly_verified"
        ]
        is True
    )
    assert (
        config["inventory_at_freeze"]["AS295"]["secure_member_velocity_catalog_publicly_verified"]
        is False
    )
