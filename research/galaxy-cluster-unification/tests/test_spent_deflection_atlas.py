import numpy as np

from voidscreen.spent_deflection_atlas import (
    leave_one_out_constant,
    leave_one_out_log_linear,
    loglog_interpolate,
    vector_alignment,
    vector_rms,
)


def test_vector_helpers_identify_scaled_parallel_fields():
    x = np.array([1.0, 0.0, -1.0])
    y = np.array([0.0, 2.0, 0.0])
    assert np.isclose(vector_alignment(x, y, 3.0 * x, 3.0 * y), 1.0)
    assert np.isclose(vector_rms(3.0 * x, 3.0 * y), 3.0 * vector_rms(x, y))


def test_loglog_interpolation_preserves_power_law():
    anchors = np.array([1.0, 10.0, 100.0])
    values = anchors**-2
    target = np.array([2.0, 20.0])
    assert np.allclose(loglog_interpolate(target, anchors, values), target**-2)


def test_leave_one_out_linear_predicts_exact_relation():
    x = np.arange(5.0)
    y = 1.5 + 2.25 * x
    predicted, rmse = leave_one_out_log_linear(x, y)
    assert np.allclose(predicted, y)
    assert rmse < 1e-12


def test_leave_one_out_constant_uses_only_training_rows():
    predicted, rmse = leave_one_out_constant(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(predicted, [2.5, 2.0, 1.5])
    assert rmse > 0.0
