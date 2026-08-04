"""Source-uniqueness and inherited-constraint checks for Sigma v8B.

The published AeST base has a conserved homogeneous shift charge.  For a
quadratic clock minimum, a nonzero charge produces a leading density scaling
as ``a^-3`` whose normalization is an initial condition rather than a baryonic
prediction.  The Sigma goal therefore freezes the charge to zero and forbids
using it as an observational parameter.

The v8B causal completion changes the homogeneous Noether current.  This module
derives its exact current and checks all zero-current FLRW branches.  It also
reproduces the published AeST Dirac count, while deliberately refusing to
inherit that count for the AeST-plus-completion action without a new nonlinear
Hamiltonian derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstraintCount:
    configuration_variables: int
    phase_space_dimension: int
    first_class_constraints: int
    second_class_constraints: int
    physical_degrees_of_freedom: float


@dataclass(frozen=True)
class PublishedAestCosmologicalState:
    scale_factor: float
    conserved_charge: float
    q_value: float
    q_displacement: float
    k_q: float
    density_times_8pi_g: float
    leading_dust_density_times_8pi_g: float
    subleading_stiff_density_times_8pi_g: float


@dataclass(frozen=True)
class V8bFlrwBranch:
    name: str
    q_value: float
    current_density: float
    current_slope: float
    positive_clock: bool


def _finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def dirac_constraint_count(
    *,
    configuration_variables: int,
    first_class_constraints: int,
    second_class_constraints: int,
) -> ConstraintCount:
    """Return ``(2N-2F-S)/2`` for a constrained Hamiltonian system."""

    variables = int(configuration_variables)
    first = int(first_class_constraints)
    second = int(second_class_constraints)
    if variables < 0 or first < 0 or second < 0:
        raise ValueError("constraint counts must be non-negative")
    phase_dimension = 2 * variables
    remaining = phase_dimension - 2 * first - second
    if remaining < 0 or remaining % 2:
        raise ValueError("constraint count does not define an integral phase-space pair count")
    return ConstraintCount(
        configuration_variables=variables,
        phase_space_dimension=phase_dimension,
        first_class_constraints=first,
        second_class_constraints=second,
        physical_degrees_of_freedom=remaining / 2.0,
    )


def published_aest_quadratic_cosmological_state(
    *,
    scale_factor: float,
    k_2: float,
    q_0: float,
    conserved_charge: float,
) -> PublishedAestCosmologicalState:
    """Solve the published quadratic AeST clock branch exactly.

    For ``K(Q)=K2 (Q-Q0)^2`` and the integrated scalar equation
    ``a^3 K_Q=I0``, the displacement is ``I0/(2 K2 a^3)``.  The
    dimensionless energy density reported here is ``8 pi G rho=Q K_Q-K``.
    """

    a = _finite(scale_factor, name="scale_factor")
    clock = _finite(k_2, name="k_2")
    q_zero = _finite(q_0, name="q_0")
    charge = _finite(conserved_charge, name="conserved_charge")
    if a <= 0.0 or clock <= 0.0:
        raise ValueError("scale_factor and k_2 must be positive")
    displacement = charge / (2.0 * clock * a**3)
    q_value = q_zero + displacement
    k_q = 2.0 * clock * displacement
    potential = clock * displacement**2
    density = q_value * k_q - potential
    leading = q_zero * charge / a**3
    subleading = charge**2 / (4.0 * clock * a**6)
    return PublishedAestCosmologicalState(
        scale_factor=a,
        conserved_charge=charge,
        q_value=q_value,
        q_displacement=displacement,
        k_q=k_q,
        density_times_8pi_g=density,
        leading_dust_density_times_8pi_g=leading,
        subleading_stiff_density_times_8pi_g=subleading,
    )


def v8b_flrw_current_density(
    q_value: float,
    *,
    k_2: float,
    alpha: float,
    horndeski_length: float,
    hubble_inverse_length: float,
    q_0: float,
) -> float:
    """Return ``I0/a^3`` for the selected v8B homogeneous scalar.

    The selected action contains ``+2 K2 (Q-Q0)^2`` and the completion
    reduces to ``-3 C a^3 H Q(Q-Q0)^2``, with
    ``C=(alpha-1)L_H^2``.  Shift symmetry therefore gives

    ``I0/a^3=(Q-Q0)[4K2-3 C H(3Q-Q0)]``.
    """

    q = _finite(q_value, name="q_value")
    clock = _finite(k_2, name="k_2")
    completion = _finite(alpha, name="alpha")
    length = _finite(horndeski_length, name="horndeski_length")
    hubble = _finite(hubble_inverse_length, name="hubble_inverse_length")
    q_zero = _finite(q_0, name="q_0")
    if clock <= 0.0 or completion < 1.0 or length < 0.0 or hubble < 0.0:
        raise ValueError("K2 must be positive; alpha, L_H, and H must be non-negative")
    coupling = (completion - 1.0) * length**2
    displacement = q - q_zero
    return displacement * (
        4.0 * clock - 3.0 * coupling * hubble * (3.0 * q - q_zero)
    )


def v8b_flrw_current_slope(
    q_value: float,
    *,
    k_2: float,
    alpha: float,
    horndeski_length: float,
    hubble_inverse_length: float,
    q_0: float,
) -> float:
    """Return the derivative of the v8B current density with respect to Q."""

    q = _finite(q_value, name="q_value")
    clock = _finite(k_2, name="k_2")
    completion = _finite(alpha, name="alpha")
    length = _finite(horndeski_length, name="horndeski_length")
    hubble = _finite(hubble_inverse_length, name="hubble_inverse_length")
    q_zero = _finite(q_0, name="q_0")
    if clock <= 0.0 or completion < 1.0 or length < 0.0 or hubble < 0.0:
        raise ValueError("K2 must be positive; alpha, L_H, and H must be non-negative")
    coupling = (completion - 1.0) * length**2
    return 4.0 * clock - 3.0 * coupling * hubble * (6.0 * q - 4.0 * q_zero)


def v8b_zero_charge_flrw_branches(
    *,
    k_2: float,
    alpha: float,
    horndeski_length: float,
    hubble_inverse_length: float,
    q_0: float,
) -> tuple[V8bFlrwBranch, ...]:
    """Return every algebraic ``I0=0`` branch of the homogeneous current."""

    clock = _finite(k_2, name="k_2")
    completion = _finite(alpha, name="alpha")
    length = _finite(horndeski_length, name="horndeski_length")
    hubble = _finite(hubble_inverse_length, name="hubble_inverse_length")
    q_zero = _finite(q_0, name="q_0")
    if clock <= 0.0 or completion < 1.0 or length < 0.0 or hubble < 0.0:
        raise ValueError("K2 must be positive; alpha, L_H, and H must be non-negative")
    common = {
        "k_2": clock,
        "alpha": completion,
        "horndeski_length": length,
        "hubble_inverse_length": hubble,
        "q_0": q_zero,
    }

    def branch(name: str, q_value: float) -> V8bFlrwBranch:
        current = v8b_flrw_current_density(q_value, **common)
        slope = v8b_flrw_current_slope(q_value, **common)
        return V8bFlrwBranch(
            name=name,
            q_value=q_value,
            current_density=current,
            current_slope=slope,
            positive_clock=slope > 0.0,
        )

    result = [branch("clock_minimum", q_zero)]
    coupling_hubble = (completion - 1.0) * length**2 * hubble
    if coupling_hubble > 0.0:
        alternative_q = (q_zero + 4.0 * clock / (3.0 * coupling_hubble)) / 3.0
        if not np.isclose(alternative_q, q_zero, rtol=0.0, atol=1.0e-14):
            result.append(branch("completion_root", alternative_q))
    return tuple(result)


def audit_v8b_source_constraint_gate(
    *,
    k_2: float,
    alpha: float,
    horndeski_length: float,
    hubble_inverse_length: float,
    q_0: float,
    frozen_cosmological_charge: float,
    physical_parameter_count: int,
    maximum_physical_parameters: int,
) -> dict[str, object]:
    """Audit the inherited count and homogeneous source-uniqueness subgate."""

    base_count = dirac_constraint_count(
        configuration_variables=12,
        first_class_constraints=4,
        second_class_constraints=4,
    )
    zero_state = published_aest_quadratic_cosmological_state(
        scale_factor=1.0,
        k_2=k_2,
        q_0=q_0,
        conserved_charge=0.0,
    )
    charged_now = published_aest_quadratic_cosmological_state(
        scale_factor=1.0,
        k_2=k_2,
        q_0=q_0,
        conserved_charge=1.0e-8,
    )
    charged_later = published_aest_quadratic_cosmological_state(
        scale_factor=2.0,
        k_2=k_2,
        q_0=q_0,
        conserved_charge=1.0e-8,
    )
    branches = v8b_zero_charge_flrw_branches(
        k_2=k_2,
        alpha=alpha,
        horndeski_length=horndeski_length,
        hubble_inverse_length=hubble_inverse_length,
        q_0=q_0,
    )
    minimum = next(item for item in branches if item.name == "clock_minimum")
    alternatives = [item for item in branches if item.name != "clock_minimum"]
    dust_ratio = (
        charged_now.leading_dust_density_times_8pi_g
        / charged_later.leading_dust_density_times_8pi_g
    )
    count = int(physical_parameter_count)
    maximum = int(maximum_physical_parameters)
    if count < 0 or maximum < 0:
        raise ValueError("parameter counts must be non-negative")
    charge = _finite(frozen_cosmological_charge, name="frozen_cosmological_charge")
    completed_subgates = {
        "published_base_dirac_count_reproduced": bool(
            base_count.physical_degrees_of_freedom == 6.0
        ),
        "published_nonzero_charge_has_a_minus_three_density": bool(
            np.isclose(dust_ratio, 8.0, rtol=1.0e-12, atol=0.0)
        ),
        "zero_charge_removes_published_dustlike_density": bool(
            zero_state.density_times_8pi_g == 0.0
        ),
        "frozen_charge_is_zero_and_not_a_parameter": bool(charge == 0.0),
        "v8b_clock_minimum_is_a_zero_current_branch": bool(
            abs(minimum.current_density) < 1.0e-12
        ),
        "v8b_clock_minimum_has_positive_current_slope": minimum.positive_clock,
        "other_v8b_zero_current_branches_are_not_stable": bool(
            all(not item.positive_clock for item in alternatives)
        ),
        "parameter_count": count <= maximum,
    }
    unresolved_kill_gates = {
        "combined_v8b_has_published_six_dof_count": False,
        "combined_v8b_hamiltonian_bounded_on_required_backgrounds": False,
        "arbitrary_baryonic_data_select_unique_inhomogeneous_state": False,
        "published_ir_zero_frequency_sector_resolved": False,
    }
    return {
        "published_base_constraint_count": {
            "configuration_variables": base_count.configuration_variables,
            "phase_space_dimension": base_count.phase_space_dimension,
            "first_class_constraints": base_count.first_class_constraints,
            "second_class_constraints": base_count.second_class_constraints,
            "physical_degrees_of_freedom": base_count.physical_degrees_of_freedom,
        },
        "published_quadratic_clock": {
            "conserved_equation": "a^3 K_Q=I0",
            "leading_density_times_8pi_G": "Q0 I0/a^3",
            "subleading_density_times_8pi_G": "I0^2/(4 K2 a^6)",
            "dust_density_ratio_a1_to_a2": dust_ratio,
            "zero_charge_density_times_8pi_G": zero_state.density_times_8pi_g,
        },
        "v8b_current_equation": (
            "I0/a^3=(Q-Q0)[4 K2-3(alpha-1)L_H^2 H(3Q-Q0)]"
        ),
        "v8b_zero_charge_branches": [
            {
                "name": item.name,
                "q_value": item.q_value,
                "current_density": item.current_density,
                "current_slope": item.current_slope,
                "positive_clock": item.positive_clock,
            }
            for item in branches
        ],
        "frozen_cosmological_charge": charge,
        "completed_subgates": completed_subgates,
        "all_completed_subgates_pass": bool(all(completed_subgates.values())),
        "unresolved_kill_gates": unresolved_kill_gates,
        "pre_data_source_constraint_gate_pass": bool(
            all(completed_subgates.values()) and all(unresolved_kill_gates.values())
        ),
    }
