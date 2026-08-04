"""Exact clock-constraint audit for the proposed Sigma v13A successor.

The minimal covariant attempt to remove the AeST clock/Jeans sector is

``L_constraint = Lambda (U^mu partial_mu phi - Q0)``.

On a homogeneous aligned background the multiplier equation fixes the clock,
while the shift-symmetric scalar equation conserves ``a^3 Lambda`` (up to the
base scalar current).  The associated reduced Hamiltonian is linear in that
conserved charge.  The module keeps this small argument executable and also
checks that giving the multiplier a regular quadratic potential merely
integrates it out into another ``(Q-Q0)^2`` coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockConstraintParameters:
    """Dimensionless homogeneous parameters for the theory-only audit."""

    q0: float = 1.0
    k2: float = 4.0

    def validated(self) -> ClockConstraintParameters:
        values = np.asarray([self.q0, self.k2], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("clock-constraint parameters must be finite")
        if self.q0 <= 0.0 or self.k2 <= 0.0:
            raise ValueError("q0 and k2 must be positive")
        return self


DEFAULT_CLOCK_CONSTRAINT_PARAMETERS = ClockConstraintParameters()


def exact_constraint_homogeneous_state(
    *,
    scale_factor: float,
    comoving_charge: float,
    parameters: ClockConstraintParameters = DEFAULT_CLOCK_CONSTRAINT_PARAMETERS,
    base_current: float = 0.0,
) -> dict[str, float]:
    """Return the exact homogeneous multiplier and Hamiltonian density.

    The scalar equation is

    ``d_t[a^3 (J_base + Lambda)] = 0``.

    Consequently ``Lambda=I/a^3-J_base``.  On the aligned vacuum clock the
    base current vanishes and the constraint-sector pressure is zero, while
    ``rho=Q0 I/a^3``.  The charge ``I`` is initial data, not a baryonic source.
    """

    params = parameters.validated()
    a = float(scale_factor)
    charge = float(comoving_charge)
    current = float(base_current)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("scale_factor must be finite and positive")
    if not np.isfinite(charge) or not np.isfinite(current):
        raise ValueError("charge and base current must be finite")
    multiplier = charge / a**3 - current
    physical_energy_density = params.q0 * charge / a**3
    return {
        "scale_factor": a,
        "comoving_charge": charge,
        "base_current": current,
        "multiplier": multiplier,
        "clock_rate": params.q0,
        "constraint_residual": 0.0,
        "conserved_current": a**3 * (current + multiplier),
        "comoving_hamiltonian": params.q0 * charge,
        "physical_energy_density": physical_energy_density,
        "pressure": 0.0,
    }


def auxiliary_lagrangian(
    *,
    delta_q: float,
    multiplier: float,
    auxiliary_curvature: float,
) -> float:
    """Return ``Lambda deltaQ-chi Lambda^2/2`` for ``chi>=0``."""

    dq = float(delta_q)
    lam = float(multiplier)
    chi = float(auxiliary_curvature)
    if any(not np.isfinite(value) for value in (dq, lam, chi)):
        raise ValueError("auxiliary values must be finite")
    if chi < 0.0:
        raise ValueError("auxiliary_curvature must be nonnegative")
    return lam * dq - 0.5 * chi * lam**2


def regularized_auxiliary_reduction(
    *,
    delta_q: float,
    auxiliary_curvature: float,
    parameters: ClockConstraintParameters = DEFAULT_CLOCK_CONSTRAINT_PARAMETERS,
) -> dict[str, float | bool]:
    """Eliminate a regular quadratic multiplier and return its effective K2.

    For finite positive ``chi``, variation gives ``Lambda=deltaQ/chi`` and
    ``Delta L_eff=deltaQ^2/(2 chi)``.  Since the AeST convention is
    ``2 K2 deltaQ^2``, this is exactly ``K2 -> K2+1/(4 chi)``.  It is a soft
    clock susceptibility, not a new constraint.
    """

    params = parameters.validated()
    dq = float(delta_q)
    chi = float(auxiliary_curvature)
    if not np.isfinite(dq) or not np.isfinite(chi) or chi <= 0.0:
        raise ValueError("delta_q must be finite and auxiliary_curvature positive")
    stationary_multiplier = dq / chi
    direct = auxiliary_lagrangian(
        delta_q=dq,
        multiplier=stationary_multiplier,
        auxiliary_curvature=chi,
    )
    effective = dq**2 / (2.0 * chi)
    effective_k2 = params.k2 + 1.0 / (4.0 * chi)
    return {
        "delta_q": dq,
        "auxiliary_curvature": chi,
        "stationary_multiplier": stationary_multiplier,
        "direct_stationary_lagrangian": direct,
        "effective_lagrangian": effective,
        "effective_k2": effective_k2,
        "is_exact_constraint": False,
        "adds_new_constraint": False,
    }


def finite_difference_stationarity_residual(
    *,
    delta_q: float,
    auxiliary_curvature: float,
    step: float = 1.0e-6,
) -> float:
    """Check the auxiliary stationary point independently by finite difference."""

    chi = float(auxiliary_curvature)
    dq = float(delta_q)
    h = float(step)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("step must be finite and positive")
    stationary = dq / chi
    upper = auxiliary_lagrangian(
        delta_q=dq,
        multiplier=stationary + h,
        auxiliary_curvature=chi,
    )
    lower = auxiliary_lagrangian(
        delta_q=dq,
        multiplier=stationary - h,
        auxiliary_curvature=chi,
    )
    return abs((upper - lower) / (2.0 * h))


def clock_constraint_no_go_audit(
    *,
    scale_factors: tuple[float, ...],
    signed_charges: tuple[float, ...],
    positive_source_uniqueness_charges: tuple[float, ...],
    auxiliary_curvatures: tuple[float, ...],
    delta_q: float,
    parameters: ClockConstraintParameters = DEFAULT_CLOCK_CONSTRAINT_PARAMETERS,
) -> dict[str, object]:
    """Evaluate conservation, energy, uniqueness, and regularization branches."""

    params = parameters.validated()
    if not scale_factors or not signed_charges or not auxiliary_curvatures:
        raise ValueError("all clock-constraint audit grids must be nonempty")
    exact_rows = [
        exact_constraint_homogeneous_state(
            scale_factor=float(a),
            comoving_charge=float(charge),
            parameters=params,
        )
        for charge in signed_charges
        for a in scale_factors
    ]
    conservation_residual = max(
        abs(float(row["conserved_current"]) - float(row["comoving_charge"]))
        for row in exact_rows
    )
    dust_scaling_residual = 0.0
    nonzero_charges = [float(value) for value in signed_charges if float(value) != 0.0]
    for charge in nonzero_charges:
        reference = exact_constraint_homogeneous_state(
            scale_factor=1.0,
            comoving_charge=charge,
            parameters=params,
        )
        for a in scale_factors:
            row = exact_constraint_homogeneous_state(
                scale_factor=float(a),
                comoving_charge=charge,
                parameters=params,
            )
            expected = float(reference["physical_energy_density"]) / float(a) ** 3
            scale = max(1.0, abs(expected))
            dust_scaling_residual = max(
                dust_scaling_residual,
                abs(float(row["physical_energy_density"]) - expected) / scale,
            )

    same_background_rows = [
        exact_constraint_homogeneous_state(
            scale_factor=1.0,
            comoving_charge=float(charge),
            parameters=params,
        )
        for charge in positive_source_uniqueness_charges
    ]
    same_background_energies = {
        float(row["physical_energy_density"]) for row in same_background_rows
    }
    regularized_rows = [
        regularized_auxiliary_reduction(
            delta_q=float(delta_q),
            auxiliary_curvature=float(curvature),
            parameters=params,
        )
        for curvature in auxiliary_curvatures
    ]
    regularization_identity_residual = max(
        abs(
            float(row["direct_stationary_lagrangian"])
            - float(row["effective_lagrangian"])
        )
        for row in regularized_rows
    )
    stationarity_residual = max(
        finite_difference_stationarity_residual(
            delta_q=float(delta_q),
            auxiliary_curvature=float(curvature),
        )
        for curvature in auxiliary_curvatures
    )
    negative_hamiltonian_rows = [
        row for row in exact_rows if float(row["comoving_hamiltonian"]) < 0.0
    ]
    return {
        "action_term": "Lambda (U^mu nabla_mu phi-Q0)",
        "homogeneous_scalar_equation": "d_t[a^3(J_base+Lambda)]=0",
        "reduced_constraint_hamiltonian": "H_constraint=Q0 I",
        "exact_constraint_rows": exact_rows,
        "maximum_conserved_current_residual": conservation_residual,
        "maximum_dust_redshift_residual": dust_scaling_residual,
        "negative_hamiltonian_row_count": len(negative_hamiltonian_rows),
        "hamiltonian_unbounded_for_unrestricted_signed_charge": bool(
            params.q0 > 0.0 and any(charge < 0.0 for charge in signed_charges)
        ),
        "same_baryonic_background_charge_rows": same_background_rows,
        "same_baryonic_background_distinct_energy_count": len(
            same_background_energies
        ),
        "source_uniqueness_violated_even_for_nonnegative_charge": bool(
            len(same_background_energies) > 1
        ),
        "regularized_auxiliary_rows": regularized_rows,
        "maximum_regularization_identity_residual": regularization_identity_residual,
        "maximum_finite_difference_stationarity_residual": stationarity_residual,
        "finite_regularization_is_only_k2_renormalization": all(
            not bool(row["adds_new_constraint"]) for row in regularized_rows
        ),
    }
