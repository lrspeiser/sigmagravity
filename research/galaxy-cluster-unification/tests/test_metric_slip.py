import numpy as np

from voidscreen.metric_slip import (
    extra_force_lensing_ratio,
    metric_slip_eta,
    metric_slip_lensing_acceleration,
)


def test_zero_slip_reproduces_dynamical_acceleration():
    gbar = np.asarray([1.0e-10, 1.0e-12])
    gdyn = np.asarray([1.2e-10, 5.0e-12])
    assert np.allclose(metric_slip_lensing_acceleration(gbar, gdyn, 0.0), gdyn)
    assert np.allclose(metric_slip_eta(gbar, gdyn, 0.0), 1.0)


def test_positive_slip_boosts_only_the_extra_force():
    gbar = np.asarray([1.0])
    gdyn = np.asarray([3.0])
    lensing = metric_slip_lensing_acceleration(gbar, gdyn, 2.0)
    assert np.allclose(lensing, 5.0)
    assert extra_force_lensing_ratio(2.0) == 2.0
    assert np.allclose(metric_slip_eta(gbar, gdyn, 2.0), 7.0 / 3.0)


def test_screened_limit_is_gr_independent_of_slip():
    for slip in [-1.0, 0.0, 8.0]:
        assert np.allclose(metric_slip_lensing_acceleration([2.0], [2.0], slip), 2.0)
        assert np.allclose(metric_slip_eta([2.0], [2.0], slip), 1.0)
