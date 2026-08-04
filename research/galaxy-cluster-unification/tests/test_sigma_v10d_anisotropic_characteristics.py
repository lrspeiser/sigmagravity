from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_v10d_anisotropic_characteristics import (
    audit_v10d_anisotropic_characteristics,
    boosted_one_dimensional_speed,
    divergence_gram,
    quadratic_characteristic_roots,
    static_schur_complement,
)


def test_divergence_gram_has_half_half_one_spectrum() -> None:
    result = divergence_gram(np.array([1.0, 2.0, -3.0]))
    assert np.linalg.eigvalsh(result) == pytest.approx([0.5, 0.5, 1.0])


def test_identity_background_reproduces_longitudinal_and_transverse_roots() -> None:
    roots = quadratic_characteristic_roots(
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        base_spatial_stiffness=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    assert np.max(np.abs(roots.imag)) < 1.0e-12
    assert roots.real == pytest.approx(
        [0.2045454545, 0.232009, 0.232009, 0.881627, 0.881627, 1.0],
        abs=1.0e-6,
    )


def test_noncommuting_anisotropic_static_and_dynamic_blocks_are_healthy() -> None:
    kinetic = np.array([[3.0, 0.7, -0.2], [0.7, 1.8, 0.4], [-0.2, 0.4, 2.2]])
    direction = np.array([1.0, -2.0, 0.5])
    schur = static_schur_complement(
        kinetic,
        direction,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    roots = quadratic_characteristic_roots(
        kinetic,
        direction,
        base_spatial_stiffness=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
    )
    assert np.min(np.linalg.eigvalsh(schur)) > 0.0
    assert np.max(np.abs(roots.imag)) < 1.0e-12
    assert np.min(roots.real) > 0.0
    assert np.max(roots.real) < 1.0


def test_subluminal_rest_speed_remains_subluminal_under_boost() -> None:
    for wave in np.linspace(-1.0, 1.0, 101):
        for boost in np.linspace(-0.99, 0.99, 101):
            assert abs(boosted_one_dimensional_speed(float(wave), float(boost))) <= 1.0 + 1e-12


def test_anisotropic_characteristic_audit_passes_only_source_block() -> None:
    report = audit_v10d_anisotropic_characteristics(
        k_b=1.0,
        beta=np.sqrt(2.0 / 11.0),
        base_spatial_stiffness=3.0 / 4.0,
        carrier_speed_squared=3.0 / 11.0,
        normalized_mixing_squared=2.0 / 11.0,
        random_samples=200,
    )
    assert all(report["gates"].values())
    assert report["all_anisotropic_source_block_gates_pass"] is True
    assert report["unresolved"]["full_metric_aether_scalar_carrier_principal_symbol"] is False
