from __future__ import annotations

import pytest

from voidscreen.sigma_v12a_homogeneous_dirac import (
    audit_v12a_homogeneous_dirac,
    homogeneous_dhost_invariants,
    homogeneous_kinetic_coefficients,
    homogeneous_momenta,
    homogeneous_primary,
    homogeneous_primary_secondary_bracket,
    homogeneous_reduced_hamiltonian,
    homogeneous_secondary,
)


def test_homogeneous_covariant_invariants() -> None:
    invariants = homogeneous_dhost_invariants(2.0, 3.0, 5.0)
    assert invariants["L3"] == pytest.approx(-156.0)
    assert invariants["L4"] == pytest.approx(-36.0)
    assert invariants["L5"] == pytest.approx(144.0)


@pytest.mark.parametrize("clock", [-3.0, -0.4, 0.0, 1.0, 2.7])
def test_homogeneous_class_ia_coefficients_form_degenerate_square(clock: float) -> None:
    coefficients = homogeneous_kinetic_coefficients(clock, f0=1.0, a3=0.7)
    assert coefficients["kappa"] == pytest.approx(-2.0 / 3.0)
    assert coefficients["scalar_a_direct"] == pytest.approx(
        coefficients["scalar_a_schur"], abs=1.0e-12
    )


def test_primary_identity_and_reduced_hamiltonian_drop_velocity() -> None:
    coefficients = homogeneous_kinetic_coefficients(1.4, f0=1.0, a3=0.6)
    for v_star, trace in [(-1.2, 0.7), (0.0, 2.0), (3.1, -0.8)]:
        p_clock, p_metric = homogeneous_momenta(
            v_star,
            trace,
            kappa=coefficients["kappa"],
            mixing_b=coefficients["mixing_b"],
        )
        assert homogeneous_primary(
            p_clock,
            p_metric,
            kappa=coefficients["kappa"],
            mixing_b=coefficients["mixing_b"],
        ) == pytest.approx(0.0, abs=1.0e-12)
        kinetic = (
            coefficients["kappa"] * trace**2
            + 2.0 * coefficients["mixing_b"] * v_star * trace
            + coefficients["scalar_a_schur"] * v_star**2
        )
        legendre = p_clock * v_star + p_metric * trace - kinetic
        assert legendre == pytest.approx(p_metric**2 / (4.0 * coefficients["kappa"]))


def test_flat_clock_pair_is_regular_when_dhost_activation_vanishes() -> None:
    coefficients = homogeneous_kinetic_coefficients(1.0, f0=1.0, a3=0.0)
    assert coefficients["mixing_b"] == pytest.approx(0.0)
    assert coefficients["scalar_a_direct"] == pytest.approx(0.0)
    assert homogeneous_primary_secondary_bracket(k_2=2.0) == pytest.approx(-8.0)
    assert homogeneous_secondary(1.0, 0.0, k_2=2.0, background_clock=1.0) == pytest.approx(0.0)


def test_reduced_hamiltonian_contains_aest_clock_susceptibility() -> None:
    value = homogeneous_reduced_hamiltonian(
        -2.0,
        1.5,
        0.7,
        kappa=-2.0 / 3.0,
        k_2=2.0,
        background_clock=1.0,
    )
    expected = (-2.0) ** 2 / (4.0 * (-2.0 / 3.0)) + 0.7 * 1.5 - 4.0 * 0.5**2
    assert value == pytest.approx(expected)


def test_homogeneous_audit_passes_without_claiming_arbitrary_tilt() -> None:
    report = audit_v12a_homogeneous_dirac(
        f0=1.0,
        k_2=2.0,
        orientation_strength=1.0,
        background_clock=1.0,
        clock_scan_minimum=-2.0,
        clock_scan_maximum=2.0,
        clock_scan_points=101,
        random_velocity_trials=2,
        random_seed=12005,
    )
    assert report["homogeneous_aligned_dirac_pair_regular"]
    assert all(report["gates"].values())
    assert not report["arbitrary_gradient_tilt_regular"]
    assert not report["complete_dirac_chain_derived"]
    assert not report["physical_degree_count_proven_unchanged"]
    assert not report["theory_viable"]
    assert not report["observational_data_accessed"]


def test_invalid_homogeneous_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        homogeneous_primary_secondary_bracket(k_2=0.0)
    with pytest.raises(ValueError):
        homogeneous_kinetic_coefficients(1.0, f0=0.0, a3=1.0)
