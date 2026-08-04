from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_degenerate_action import (
    bounded_even_activation,
    k_mouflage_static_parallel_speed_squared,
    luminal_class_ia_coefficients,
    newton_yukawa_acceleration_ratio,
    newton_yukawa_circular_speed_ratio,
    newton_yukawa_log_acceleration_slope,
    normalized_dhost_residuals,
    v5c_trial_coefficients,
)


def test_bounded_even_activation_is_smooth_and_signed_safe() -> None:
    value = np.linspace(-10.0, 10.0, 2001)
    activation = bounded_even_activation(value)
    assert np.all(np.isfinite(activation))
    assert np.all(activation >= 0.0)
    assert np.all(activation < 0.385)
    assert np.allclose(activation, bounded_even_activation(-value))
    assert bounded_even_activation(0.0) == 0.0


def test_luminal_class_ia_identities_vanish() -> None:
    rng = np.random.default_rng(7502)
    kinetic = rng.normal(size=5000)
    coupling = np.exp(rng.normal(scale=0.5, size=5000))
    derivative = rng.normal(scale=0.3, size=5000)
    a3 = rng.normal(scale=0.4, size=5000)
    coefficients = luminal_class_ia_coefficients(
        kinetic, coupling, derivative, a3
    )
    residuals = normalized_dhost_residuals(
        kinetic, coupling, derivative, coefficients
    )
    for residual in residuals.values():
        assert np.max(np.abs(residual)) < 2.0e-15


def test_v5c_trial_row_is_finite_through_signed_zero() -> None:
    ratio = np.concatenate(
        (
            -np.geomspace(1.0e6, 1.0e-12, 1000),
            [0.0],
            np.geomspace(1.0e-12, 1.0e6, 1000),
        )
    )
    coefficients = v5c_trial_coefficients(ratio, 2.0)
    for coefficient in coefficients.values():
        assert np.all(np.isfinite(coefficient))
    assert coefficients["A1"][1000] == 0.0
    assert coefficients["A2"][1000] == 0.0
    assert coefficients["A3"][1000] == 0.0
    assert coefficients["A4"][1000] == 0.0
    assert coefficients["A5"][1000] == 0.0
    for coefficient in coefficients.values():
        assert np.max(np.abs(coefficient)) < 1.0


def test_static_derivative_screen_is_superluminal_under_strict_gate() -> None:
    magnitude = np.geomspace(1.0e-6, 1.0e6, 1000)
    kinetic = -magnitude
    # Representative screening law P_X=1+(-X)^p with p=1.
    first = 1.0 + magnitude
    second = -np.ones_like(magnitude)
    speed_squared = k_mouflage_static_parallel_speed_squared(
        kinetic, first, second
    )
    assert np.all(speed_squared > 1.0)
    assert speed_squared[-1] == pytest.approx(3.0, rel=2.0e-6)


def test_invalid_degenerate_action_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        luminal_class_ia_coefficients(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        k_mouflage_static_parallel_speed_squared(1.0, 1.0, 0.0)


def test_attractive_yukawa_exterior_is_never_flatter_than_inverse_square() -> None:
    ratio = np.geomspace(1.0e-8, 1.0e8, 4000)
    for strength in (0.0, 0.01, 1.0, 100.0, 1.0e6):
        force = newton_yukawa_acceleration_ratio(ratio, strength)
        slope = newton_yukawa_log_acceleration_slope(ratio, strength)
        assert np.all(force >= 1.0)
        assert np.all(slope <= -2.0)


def test_attractive_yukawa_speed_declines_over_every_exterior_decade() -> None:
    inner = np.geomspace(1.0e-8, 1.0e8, 4000)
    for strength in (0.0, 1.0, 100.0):
        ratio = newton_yukawa_circular_speed_ratio(10.0, inner, strength)
        assert np.all(ratio <= 1.0 / np.sqrt(10.0) + 1.0e-14)


def test_invalid_yukawa_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        newton_yukawa_acceleration_ratio(-1.0, 1.0)
    with pytest.raises(ValueError):
        newton_yukawa_acceleration_ratio(1.0, -1.0)
    with pytest.raises(ValueError):
        newton_yukawa_circular_speed_ratio(1.0, 1.0, 1.0)
