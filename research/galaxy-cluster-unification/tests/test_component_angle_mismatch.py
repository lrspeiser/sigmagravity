from __future__ import annotations

import numpy as np
import pytest

from voidscreen.geometric_transport import ThinSheetField, component_angle_mismatch


def field(x, y):
    x_values = np.full((17, 17), float(x))
    y_values = np.full((17, 17), float(y))
    magnitude = np.hypot(x_values, y_values)
    return ThinSheetField(np.zeros_like(x_values), x_values, y_values, magnitude)


def test_all_modes_vanish_for_aligned_components():
    first, second = field(1.0, 0.0), field(3.0, 0.0)
    for mode in ("quadratic_cancellation", "linear_chord_mix", "oriented_cross_mix"):
        assert np.max(component_angle_mismatch(first, second, mode=mode)) == 0.0


def test_linear_chord_is_first_order_and_bounded():
    angle = 1e-3
    mismatch = component_angle_mismatch(
        field(1.0, 0.0),
        field(np.cos(angle), np.sin(angle)),
        mode="linear_chord_mix",
    )
    assert np.allclose(mismatch, angle / 2.0, rtol=1e-6)
    opposite = component_angle_mismatch(
        field(1.0, 0.0), field(-1.0, 0.0), mode="linear_chord_mix"
    )
    assert np.allclose(opposite, 1.0)


def test_mixing_gate_vanishes_when_one_component_disappears():
    zero = field(0.0, 0.0)
    for mode in ("linear_chord_mix", "oriented_cross_mix"):
        assert np.max(component_angle_mismatch(field(1.0, 0.0), zero, mode=mode)) == 0.0


def test_cross_mix_matches_equal_weight_right_angle_limit():
    mismatch = component_angle_mismatch(
        field(1.0, 0.0), field(0.0, 1.0), mode="oriented_cross_mix"
    )
    assert np.allclose(mismatch, 0.5)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        component_angle_mismatch(field(1.0, 0.0), field(0.0, 1.0), mode="unknown")
