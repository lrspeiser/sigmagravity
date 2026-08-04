from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v8b_source_constraint import (
    audit_v8b_source_constraint_gate,
    dirac_constraint_count,
    published_aest_quadratic_cosmological_state,
    v8b_flrw_current_density,
    v8b_zero_charge_flrw_branches,
)


def test_published_aest_constraint_count_has_six_physical_dof() -> None:
    count = dirac_constraint_count(
        configuration_variables=12,
        first_class_constraints=4,
        second_class_constraints=4,
    )
    assert count.phase_space_dimension == 24
    assert count.physical_degrees_of_freedom == pytest.approx(6.0)


def test_published_nonzero_shift_charge_contains_dustlike_density() -> None:
    first = published_aest_quadratic_cosmological_state(
        scale_factor=1.0,
        k_2=2.0,
        q_0=0.5,
        conserved_charge=1.0e-5,
    )
    second = published_aest_quadratic_cosmological_state(
        scale_factor=2.0,
        k_2=2.0,
        q_0=0.5,
        conserved_charge=1.0e-5,
    )
    assert first.leading_dust_density_times_8pi_g == pytest.approx(
        8.0 * second.leading_dust_density_times_8pi_g
    )
    assert first.subleading_stiff_density_times_8pi_g == pytest.approx(
        64.0 * second.subleading_stiff_density_times_8pi_g
    )
    assert first.density_times_8pi_g == pytest.approx(
        first.leading_dust_density_times_8pi_g
        + first.subleading_stiff_density_times_8pi_g
    )


def test_zero_shift_charge_removes_the_published_clock_density() -> None:
    state = published_aest_quadratic_cosmological_state(
        scale_factor=0.7,
        k_2=2.0,
        q_0=0.5,
        conserved_charge=0.0,
    )
    assert state.q_value == pytest.approx(0.5)
    assert state.density_times_8pi_g == pytest.approx(0.0)


def test_v8b_zero_charge_selects_only_the_stable_clock_minimum() -> None:
    values = {
        "k_2": 2.0,
        "alpha": 16.0 / 9.0,
        "horndeski_length": 1.0,
        "hubble_inverse_length": 0.25,
        "q_0": 0.5,
    }
    branches = v8b_zero_charge_flrw_branches(**values)
    assert [item.name for item in branches] == ["clock_minimum", "completion_root"]
    assert all(abs(item.current_density) < 1.0e-12 for item in branches)
    assert branches[0].q_value == pytest.approx(values["q_0"])
    assert branches[0].positive_clock
    assert not branches[1].positive_clock
    assert branches[1].current_slope < 0.0


def test_v8b_current_equation_matches_direct_reduced_lagrangian_derivative() -> None:
    q_value = 0.8
    q_zero = 0.5
    k_2 = 2.0
    alpha = 16.0 / 9.0
    length = 1.3
    hubble = 0.2
    coefficient = (alpha - 1.0) * length**2
    step = 1.0e-6

    def reduced_lagrangian(q: float) -> float:
        displacement = q - q_zero
        return 2.0 * k_2 * displacement**2 - 3.0 * coefficient * hubble * q * displacement**2

    numerical = (
        reduced_lagrangian(q_value + step) - reduced_lagrangian(q_value - step)
    ) / (2.0 * step)
    analytic = v8b_flrw_current_density(
        q_value,
        k_2=k_2,
        alpha=alpha,
        horndeski_length=length,
        hubble_inverse_length=hubble,
        q_0=q_zero,
    )
    assert analytic == pytest.approx(numerical, rel=1.0e-9)


def test_source_constraint_audit_holds_v8b_before_data() -> None:
    audit = audit_v8b_source_constraint_gate(
        k_2=2.0,
        alpha=16.0 / 9.0,
        horndeski_length=1.0,
        hubble_inverse_length=0.25,
        q_0=0.5,
        frozen_cosmological_charge=0.0,
        physical_parameter_count=5,
        maximum_physical_parameters=5,
    )
    assert all(audit["completed_subgates"].values())
    assert not any(audit["unresolved_kill_gates"].values())
    assert not audit["pre_data_source_constraint_gate_pass"]
    assert audit["published_base_constraint_count"]["physical_degrees_of_freedom"] == 6.0


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "configuration_variables": 1,
            "first_class_constraints": 1,
            "second_class_constraints": 1,
        },
        {
            "configuration_variables": -1,
            "first_class_constraints": 0,
            "second_class_constraints": 0,
        },
    ],
)
def test_invalid_constraint_counts_are_rejected(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        dirac_constraint_count(**arguments)


def test_invalid_source_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        published_aest_quadratic_cosmological_state(
            scale_factor=0.0,
            k_2=2.0,
            q_0=0.5,
            conserved_charge=0.0,
        )
    with pytest.raises(ValueError):
        v8b_flrw_current_density(
            np.nan,
            k_2=2.0,
            alpha=16.0 / 9.0,
            horndeski_length=1.0,
            hubble_inverse_length=0.2,
            q_0=0.5,
        )
