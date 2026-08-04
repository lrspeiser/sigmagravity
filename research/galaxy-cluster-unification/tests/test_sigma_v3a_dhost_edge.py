from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from voidscreen.sigma_dhost_edge import (
    dhost_degeneracy_residuals,
    luminal_beyond_horndeski_coefficients,
    maximum_smooth_power_law_weyl_fraction,
    smooth_power_law_weyl_fraction,
    spherical_accelerations,
    uniform_density_acceleration_ratios,
    weyl_edge_correction_from_density_gradient,
)

ROOT = Path(__file__).resolve().parents[1]


def test_coefficients_are_luminal_degenerate_and_reconstruct_alpha_h() -> None:
    alpha_h = 0.2
    x = -np.geomspace(1e-4, 1e4, 2000)
    coefficients = luminal_beyond_horndeski_coefficients(
        x, x_background=-1.0, alpha_h=alpha_h
    )
    residuals = dhost_degeneracy_residuals(coefficients, x)

    reconstructed_alpha_h = -2.0 * x * coefficients["F_X"] / coefficients["F"]
    beta_1_numerator = x * (
        4.0 * coefficients["F_X"] + x * coefficients["A3"]
    )
    tensor_speed_squared = coefficients["F"] / (
        coefficients["F"] - x * coefficients["A1"]
    )

    assert np.allclose(reconstructed_alpha_h, alpha_h, rtol=2e-15)
    assert np.allclose(beta_1_numerator, 0.0, atol=1e-13)
    assert np.allclose(tensor_speed_squared, 1.0)
    assert np.all(coefficients["F"] > 0.0)
    assert np.allclose(residuals["A1"], 0.0)
    assert np.allclose(residuals["A2"], 0.0)
    assert np.max(
        np.abs(residuals["A4"])
        / np.maximum(np.abs(coefficients["A4"]), np.finfo(float).tiny)
    ) < 1e-12
    a5_cancellation_scale = (
        np.abs(4.0 * coefficients["F_X"])
        + np.abs(x * coefficients["A3"])
    ) * np.abs(coefficients["A3"]) / (2.0 * coefficients["F"])
    assert np.max(
        np.abs(residuals["A5"])
        / np.maximum(a5_cancellation_scale, np.finfo(float).tiny)
    ) < 1e-12


def test_spherical_weyl_response_is_exactly_a_density_gradient_edge_term() -> None:
    radius = np.geomspace(0.1, 10.0, 1000)
    slope = 1.4
    alpha_h = 0.2
    density = radius**-slope
    density_gradient = -slope * density / radius
    mass = 4.0 * np.pi * radius ** (3.0 - slope) / (3.0 - slope)
    mass_prime = 4.0 * np.pi * radius**2 * density
    mass_second = 4.0 * np.pi * (2.0 - slope) * radius * density

    acceleration = spherical_accelerations(
        radius, mass, mass_prime, mass_second, alpha_h=alpha_h
    )
    expected = weyl_edge_correction_from_density_gradient(
        radius, density_gradient, alpha_h=alpha_h
    )

    assert np.allclose(
        acceleration["photon_weyl"] - acceleration["newtonian"],
        expected,
        rtol=2e-13,
        atol=1e-13,
    )


def test_uniform_core_bound_and_smooth_profile_maximum() -> None:
    alpha_supremum = 1.0 / 3.0
    interior = uniform_density_acceleration_ratios(
        np.nextafter(alpha_supremum, 0.0)
    )
    assert interior["matter_psi_over_newtonian"] > 0.0
    assert np.isclose(interior["photon_weyl_over_newtonian"], 1.0)

    slopes = np.linspace(0.0, 3.0, 10001)
    corrections = np.asarray(
        [smooth_power_law_weyl_fraction(slope, alpha_supremum) for slope in slopes]
    )
    assert np.isclose(slopes[int(np.argmax(corrections))], 1.5)
    assert np.isclose(corrections.max(), 3.0 / 16.0)
    assert np.isclose(
        maximum_smooth_power_law_weyl_fraction(alpha_supremum), 3.0 / 16.0
    )


def test_completed_audit_retires_the_local_edge_as_the_sole_cluster_response() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "sigma_v3a_dhost_edge_audit.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (
            ROOT / "results" / "sigma_v3a_dhost_edge_audit" / "report.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["parameters"]["total_provisional_physical_parameter_count"] == 2
    assert protocol["parameters"]["per_object_gravity_parameters"] == 0
    assert protocol["parameters"]["lensing_only_parameters"] == 0
    assert report["gate_results"]["degeneracy_tensor_speed_and_local_identities"]
    assert not report["gate_results"]["smooth_amplitude_feasibility"]
    assert not report["advances_to_full_2d_solver"]
    feasibility = report["spent_amplitude_feasibility"]
    assert np.isclose(
        feasibility["best_ultra_generous_gap_closure_fraction"],
        0.39824225318737116,
    )
    assert (
        feasibility["best_ultra_generous_gap_closure_fraction"]
        < feasibility["required_gap_closure_fraction"]
    )
