from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10d_tensor_cone import (
    audit_v10d_tensor_cone,
    axisymmetric_carrier_background,
    axisymmetric_tensor_speed_squared,
    linearized_spatial_connection_residual,
    stable_relative_speed_excess,
    tensor_carrier_characteristic_roots,
    tt_polarization_basis,
)


def test_tt_basis_is_orthonormal_transverse_and_traceless() -> None:
    direction = np.array([0.2, -0.3, 0.7])
    unit = direction / np.linalg.norm(direction)
    plus, cross = tt_polarization_basis(direction)
    for tensor in (plus, cross):
        assert tensor == pytest.approx(tensor.T)
        assert np.trace(tensor) == pytest.approx(0.0, abs=1.0e-14)
        assert tensor @ unit == pytest.approx(np.zeros(3), abs=1.0e-14)
        assert np.sum(tensor**2) == pytest.approx(1.0)
    assert np.sum(plus * cross) == pytest.approx(0.0, abs=1.0e-14)


def test_axisymmetric_connection_residual_has_exact_norm() -> None:
    direction = np.array([0.0, 0.0, 1.0])
    difference = 0.7
    background = axisymmetric_carrier_background(
        direction, perpendicular=-0.2, parallel=-0.2 + difference
    )
    for tensor in tt_polarization_basis(direction):
        residual = linearized_spatial_connection_residual(
            background, tensor, direction
        )
        assert np.sum(residual**2) == pytest.approx(0.5 * difference**2)
        assert np.einsum("l,lij->ij", direction, residual) == pytest.approx(
            np.zeros((3, 3)), abs=1.0e-14
        )


def test_isotropic_carrier_leaves_two_luminal_tensor_modes() -> None:
    speed = 3.0 / 11.0
    roots = tensor_carrier_characteristic_roots(
        0.4 * np.eye(3),
        np.array([0.0, 0.0, 1.0]),
        carrier_speed_squared=speed,
    )
    assert roots[:6] == pytest.approx(np.full(6, speed))
    assert roots[-2:] == pytest.approx(np.ones(2))


def test_axisymmetric_anisotropy_widens_exact_tt_cone() -> None:
    speed = 3.0 / 11.0
    difference = 0.3
    background = np.diag([-0.1, -0.1, -0.1 + difference])
    roots = tensor_carrier_characteristic_roots(
        background,
        np.array([0.0, 0.0, 1.0]),
        carrier_speed_squared=speed,
    )
    expected = axisymmetric_tensor_speed_squared(
        carrier_speed_squared=speed, anisotropy=difference
    )
    assert roots[:6] == pytest.approx(np.full(6, speed))
    assert roots[-2:] == pytest.approx(np.full(2, expected))
    assert expected > 1.0


def test_stable_speed_excess_resolves_small_cone_violation() -> None:
    speed_squared = 1.0 + (3.0 / 11.0) * 1.0e-12
    excess = stable_relative_speed_excess(speed_squared)
    assert excess == pytest.approx(1.363636363636e-13, rel=1.0e-3)
    assert excess > 1.0e-15


def test_tensor_cone_audit_retires_v10d_without_opening_data() -> None:
    report = audit_v10d_tensor_cone(
        carrier_speed_squared=3.0 / 11.0,
        speed_tolerance=1.0e-15,
        demonstration_anisotropy=1.0e-6,
        scan_maximum_anisotropy=1.0,
        scan_samples=101,
    )
    assert report["gates"]["isotropic_carrier_preserves_luminal_TT_cone"]
    assert report["gates"]["axisymmetric_numerical_roots_match_exact_formula"]
    assert not report["gates"]["anisotropic_carrier_preserves_metric_TT_cone"]
    assert not report["all_tensor_cone_gates_pass"]
    assert not report["observational_data_accessed"]
    assert report["demonstration"][
        "maximum_anisotropy_consistent_with_tolerance"
    ] == pytest.approx(np.sqrt((22.0 / 3.0) * 1.0e-15), rel=1.0e-12)


def test_invalid_tensor_cone_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        tensor_carrier_characteristic_roots(
            np.zeros((2, 2)),
            np.array([0.0, 0.0, 1.0]),
            carrier_speed_squared=3.0 / 11.0,
        )
    with pytest.raises(ValueError):
        audit_v10d_tensor_cone(
            carrier_speed_squared=3.0 / 11.0,
            speed_tolerance=0.0,
            demonstration_anisotropy=1.0e-6,
            scan_maximum_anisotropy=1.0,
            scan_samples=101,
        )
