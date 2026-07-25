"""Unit checks for the fixed SPARC revision sensitivity script."""

from __future__ import annotations

import math

import numpy as np

import run_sparc_scale_length_sensitivity as analysis


def test_catalog_and_sample_sizes() -> None:
    assert len(analysis.load_metadata()) == 175
    assert len(analysis.load_curves()) == 171


def test_window_limits_and_scale_value() -> None:
    values = analysis.window(np.asarray([1e-12, 3.0, 1e12]), 3.0)
    assert values[0] < 1e-10
    assert math.isclose(values[1], 1.0 / (1.0 + 1.0 / (2.0 * math.pi)))
    assert values[2] > 1.0 - 1e-10


def test_zero_dispersion_limit_is_acceleration_only() -> None:
    radius = np.asarray([1.0, 2.0, 5.0])
    velocity_bar = np.asarray([40.0, 60.0, 80.0])
    radius_m = radius * analysis.KPC_M
    g_bar = (velocity_bar * 1000.0) ** 2 / radius_m
    direct = velocity_bar * np.sqrt(1.0 + analysis.A0 * analysis.h_function(g_bar))
    assert np.allclose(
        direct,
        analysis.predict_acceleration_only(radius, velocity_bar),
        atol=1e-12,
    )


def test_primary_sample_matches_locked_inventory() -> None:
    frame = analysis.evaluate(analysis.load_curves(), 0.30)
    assert len(frame) == 164
    assert int(frame["n_points"].sum()) == 2745
    assert np.isfinite(frame.select_dtypes(include=[float, int]).to_numpy()).all()


def test_predictions_do_not_accept_observed_velocity() -> None:
    radius = np.asarray([0.5, 1.0, 3.0])
    velocity_bar = np.asarray([30.0, 50.0, 70.0])
    first = analysis.predict_window(radius, velocity_bar, 2.0)
    second = analysis.predict_window(radius, velocity_bar, 2.0)
    assert np.array_equal(first, second)
