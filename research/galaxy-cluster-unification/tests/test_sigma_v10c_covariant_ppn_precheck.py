from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10c_covariant_ppn_precheck import (
    aether_action_density_pair,
    audit_v10c_covariant_ppn_precheck,
    electric_magnetic_invariants,
    mapped_einstein_aether_coefficients,
    pure_einstein_aether_alpha1,
)


def test_unit_aether_electric_magnetic_identity_and_action_map() -> None:
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    aether = np.array([1.0, 0.0, 0.0, 0.0])
    field = np.array(
        [
            [0.0, 0.2, -0.4, 0.1],
            [-0.2, 0.0, 0.5, 0.3],
            [0.4, -0.5, 0.0, -0.7],
            [-0.1, -0.3, 0.7, 0.0],
        ]
    )
    invariants = electric_magnetic_invariants(metric, aether, field)
    assert invariants.decomposition_residual == pytest.approx(0.0, abs=1.0e-14)
    densities = aether_action_density_pair(invariants, k_b=1.0, u=0.75)
    assert densities["residual"] == pytest.approx(0.0, abs=1.0e-14)


def test_v10c_coefficient_map_preserves_c13_c14_and_sets_vector_speed() -> None:
    result = mapped_einstein_aether_coefficients(k_b=1.0, u=0.75)
    assert result["c1"] == pytest.approx(0.75)
    assert result["c2"] == pytest.approx(0.0)
    assert result["c3"] == pytest.approx(-0.75)
    assert result["c4"] == pytest.approx(0.25)
    assert result["c13"] == pytest.approx(0.0)
    assert result["c14"] == pytest.approx(1.0)
    assert result["c123"] == pytest.approx(0.0)
    assert result["pure_aether_vector_speed_squared"] == pytest.approx(0.75)


def test_pure_aether_alpha1_proxy_is_minus_four_kb_for_base_and_v10c() -> None:
    base = mapped_einstein_aether_coefficients(k_b=0.3, u=1.0)
    modified = mapped_einstein_aether_coefficients(k_b=0.3, u=0.75)
    alpha_base = pure_einstein_aether_alpha1(
        c1=base["c1"], c3=base["c3"], c4=base["c4"]
    )
    alpha_modified = pure_einstein_aether_alpha1(
        c1=modified["c1"], c3=modified["c3"], c4=modified["c4"]
    )
    assert alpha_base == pytest.approx(-1.2)
    assert alpha_modified == pytest.approx(alpha_base)


def test_precheck_passes_exact_algebra_but_does_not_claim_full_ppn() -> None:
    report = audit_v10c_covariant_ppn_precheck(k_b=1.0, u=0.75)
    assert all(report["exact_gates"].values())
    assert report["all_exact_precheck_gates_pass"] is True
    assert report["full_AeST_plus_P_PPN_derived"] is False
    assert report["counterterm_independently_retired"] is False


def test_invalid_tensor_and_coefficient_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        mapped_einstein_aether_coefficients(k_b=0.0, u=0.75)
    with pytest.raises(ValueError):
        mapped_einstein_aether_coefficients(k_b=1.0, u=1.1)
    with pytest.raises(ValueError):
        electric_magnetic_invariants(
            np.eye(4),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros((4, 4)),
        )
