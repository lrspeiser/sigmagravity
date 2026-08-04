from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10c_nonlinear_kinetic import (
    audit_v10c_nonlinear_kinetic,
    critical_isotropic_carrier_amplitude,
    reduced_homogeneous_kinetic_lagrangian,
    reduced_homogeneous_velocity_hessian,
    selected_mixing_beta,
    velocity_hessian_finite_difference_error,
)


def test_selected_critical_amplitude_is_sqrt_eleven_over_two() -> None:
    beta = selected_mixing_beta(k_b=1.0)
    critical = critical_isotropic_carrier_amplitude(k_b=1.0, beta=beta)
    assert beta == pytest.approx(np.sqrt(2.0 / 11.0))
    assert critical == pytest.approx(np.sqrt(11.0 / 2.0))


def test_reduced_velocity_hessian_matches_finite_difference() -> None:
    error = velocity_hessian_finite_difference_error(
        np.array([[0.2, 0.1, 0.0], [0.1, -0.3, 0.05], [0.0, 0.05, 0.4]]),
        k_b=1.0,
        beta=np.sqrt(2.0 / 11.0),
    )
    assert error < 1.0e-9


def test_vector_kinetic_eigenvalue_crosses_zero_at_finite_p() -> None:
    beta = selected_mixing_beta(k_b=1.0)
    critical = critical_isotropic_carrier_amplitude(k_b=1.0, beta=beta)
    at_critical = np.linalg.eigvalsh(
        reduced_homogeneous_velocity_hessian(
            critical * np.eye(3), k_b=1.0, beta=beta
        )
    )
    above = np.linalg.eigvalsh(
        reduced_homogeneous_velocity_hessian(
            1.01 * critical * np.eye(3), k_b=1.0, beta=beta
        )
    )
    assert at_critical[0] == pytest.approx(0.0, abs=1.0e-12)
    assert np.count_nonzero(above < 0.0) == 3


def test_nonlinear_kinetic_gate_retires_exact_v10c() -> None:
    report = audit_v10c_nonlinear_kinetic(k_b=1.0)
    assert all(report["derivation_gates"].values())
    assert report["all_viability_gates_pass"] is False
    assert report["retire_exact_v10c"] is True
    assert report["minimum_eigenvalues"]["above"] < 0.0


def test_invalid_nonlinear_kinetic_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        selected_mixing_beta(k_b=0.0)
    with pytest.raises(ValueError):
        reduced_homogeneous_kinetic_lagrangian(
            np.zeros(3),
            np.zeros(6),
            np.zeros((2, 2)),
            k_b=1.0,
            beta=0.4,
        )
