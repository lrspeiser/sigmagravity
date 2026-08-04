from __future__ import annotations

import pytest

from voidscreen.sigma_v13a_clock_constraint import (
    ClockConstraintParameters,
    auxiliary_lagrangian,
    clock_constraint_no_go_audit,
    exact_constraint_homogeneous_state,
    finite_difference_stationarity_residual,
    regularized_auxiliary_reduction,
)


def test_exact_clock_constraint_carries_arbitrary_dust_charge() -> None:
    row = exact_constraint_homogeneous_state(
        scale_factor=2.0,
        comoving_charge=3.0,
    )
    assert row["clock_rate"] == pytest.approx(1.0)
    assert row["constraint_residual"] == pytest.approx(0.0)
    assert row["conserved_current"] == pytest.approx(3.0)
    assert row["physical_energy_density"] == pytest.approx(3.0 / 8.0)
    assert row["pressure"] == pytest.approx(0.0)


def test_signed_charge_makes_reduced_hamiltonian_unbounded() -> None:
    negative = exact_constraint_homogeneous_state(
        scale_factor=1.0,
        comoving_charge=-10.0,
    )
    positive = exact_constraint_homogeneous_state(
        scale_factor=1.0,
        comoving_charge=10.0,
    )
    assert negative["comoving_hamiltonian"] < 0.0
    assert positive["comoving_hamiltonian"] > 0.0
    assert negative["comoving_hamiltonian"] == pytest.approx(-10.0)


def test_positive_charge_restriction_does_not_restore_source_uniqueness() -> None:
    zero = exact_constraint_homogeneous_state(
        scale_factor=1.0,
        comoving_charge=0.0,
    )
    one = exact_constraint_homogeneous_state(
        scale_factor=1.0,
        comoving_charge=1.0,
    )
    assert zero["clock_rate"] == one["clock_rate"]
    assert zero["physical_energy_density"] != one["physical_energy_density"]


def test_regular_multiplier_is_only_a_k2_shift() -> None:
    parameters = ClockConstraintParameters(q0=1.0, k2=4.0)
    row = regularized_auxiliary_reduction(
        delta_q=0.37,
        auxiliary_curvature=0.5,
        parameters=parameters,
    )
    assert row["stationary_multiplier"] == pytest.approx(0.74)
    assert row["direct_stationary_lagrangian"] == pytest.approx(
        row["effective_lagrangian"]
    )
    assert row["effective_k2"] == pytest.approx(4.5)
    assert not row["is_exact_constraint"]
    assert not row["adds_new_constraint"]
    assert finite_difference_stationarity_residual(
        delta_q=0.37,
        auxiliary_curvature=0.5,
    ) < 1.0e-9


def test_clock_constraint_no_go_audit_verifies_both_branches() -> None:
    report = clock_constraint_no_go_audit(
        scale_factors=(0.5, 1.0, 2.0),
        signed_charges=(-1.0, 0.0, 1.0),
        positive_source_uniqueness_charges=(0.0, 1.0),
        auxiliary_curvatures=(0.1, 1.0, 10.0),
        delta_q=0.37,
    )
    assert report["maximum_conserved_current_residual"] < 1.0e-12
    assert report["maximum_dust_redshift_residual"] < 1.0e-12
    assert report["negative_hamiltonian_row_count"] > 0
    assert report["hamiltonian_unbounded_for_unrestricted_signed_charge"]
    assert report["source_uniqueness_violated_even_for_nonnegative_charge"]
    assert report["finite_regularization_is_only_k2_renormalization"]
    assert report["maximum_regularization_identity_residual"] < 1.0e-12


def test_invalid_clock_constraint_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        ClockConstraintParameters(q0=0.0).validated()
    with pytest.raises(ValueError):
        exact_constraint_homogeneous_state(
            scale_factor=0.0,
            comoving_charge=1.0,
        )
    with pytest.raises(ValueError):
        auxiliary_lagrangian(
            delta_q=1.0,
            multiplier=1.0,
            auxiliary_curvature=-1.0,
        )
    with pytest.raises(ValueError):
        regularized_auxiliary_reduction(
            delta_q=1.0,
            auxiliary_curvature=0.0,
        )
