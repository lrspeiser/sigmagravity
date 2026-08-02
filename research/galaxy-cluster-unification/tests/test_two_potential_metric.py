from __future__ import annotations

import numpy as np

from voidscreen.two_potential_metric import build_two_potential_metric, rar_acceleration


def test_rar_has_deep_limit_and_exponential_high_acceleration_screening() -> None:
    a0 = 1.2e-10
    deep = 1e-8 * a0
    assert abs(rar_acceleration(deep, a0) / np.sqrt(deep * a0) - 1.0) < 1e-4
    high = np.array([0.04, 274.0])
    assert np.array_equal(rar_acceleration(high, a0), high)


def test_two_potential_metric_reconstructs_weyl_field_exactly() -> None:
    axis = np.linspace(-2.0, 2.0, 17)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    time = -(x * x + 2.0 * y * y + 3.0 * z * z)
    weyl = time + 0.2 * x * y
    metric = build_two_potential_metric(time, weyl, float(axis[1] - axis[0]))
    assert metric.weyl_identity_relative_rms < 1e-15
    assert np.allclose(0.5 * (metric.time_potential + metric.spatial_potential), weyl)


def test_zero_slip_is_exact_when_time_and_weyl_potentials_agree() -> None:
    axis = np.linspace(-1.0, 1.0, 9)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    potential = -(x * x + y * y + z * z)
    metric = build_two_potential_metric(potential, potential, float(axis[1] - axis[0]))
    assert np.array_equal(metric.spatial_potential, potential)
    assert metric.weyl_identity_relative_rms == 0.0
