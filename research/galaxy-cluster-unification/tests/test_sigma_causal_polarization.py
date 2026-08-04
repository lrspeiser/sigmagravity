from __future__ import annotations

import numpy as np
import pytest

from voidscreen.sigma_causal_polarization import (
    bounded_disformal_fraction,
    local_transport_eigenvalues,
    maximum_characteristic_speed,
    minimum_static_operator_eigenvalue,
    transition_bandpass,
)


def test_transition_bandpass_has_frozen_limits_and_peak() -> None:
    ratio = np.array([0.0, 1.0e-5, 1.0, 1.0e5])
    source = transition_bandpass(ratio)
    assert source[0] == 0.0
    assert source[1] == pytest.approx(1.0e-20, rel=1.0e-12)
    assert source[2] == 0.25
    assert source[3] == pytest.approx(1.0e-20, rel=1.0e-12)


def test_disformal_fraction_is_bounded_for_all_field_strengths() -> None:
    ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    for alpha in (0.0, 0.1, 1.0, 10.0, 1.0e6):
        fraction = bounded_disformal_fraction(ratio, alpha)
        assert np.all(fraction >= 0.0)
        assert np.all(fraction < 1.0)
        assert np.max(fraction) <= alpha / (1.0 + alpha) + 1.0e-15


def test_local_scalar_cones_are_lorentzian_and_not_superluminal() -> None:
    ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    for orientation in ("spacelike", "timelike"):
        eigenvalues = local_transport_eigenvalues(
            ratio, 3.0, orientation=orientation
        )
        assert np.all(eigenvalues["time_magnitude"] > 0.0)
        assert np.all(eigenvalues["parallel_spatial"] > 0.0)
        assert np.all(eigenvalues["transverse_spatial"] > 0.0)
        speed = maximum_characteristic_speed(ratio, 3.0, orientation=orientation)
        assert np.all(speed > 0.0)
        assert np.all(speed <= 1.0)


def test_static_massive_branch_is_unique() -> None:
    ratio = np.geomspace(1.0e-12, 1.0e12, 1000)
    eigenvalue = minimum_static_operator_eigenvalue(ratio, 20.0)
    assert np.all(eigenvalue == 1.0)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        transition_bandpass([-1.0])
    with pytest.raises(ValueError):
        bounded_disformal_fraction([1.0], -0.1)
    with pytest.raises(ValueError):
        local_transport_eigenvalues([1.0], 1.0, orientation="null")
