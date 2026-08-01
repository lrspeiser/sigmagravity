import numpy as np
import pytest

from voidscreen.vector_completion import (
    bounded_completion,
    predict_completion_acceleration,
    tidal_curvature_proxy,
)


def test_tidal_proxy_is_g_over_radius():
    result = tidal_curvature_proxy([3.085677581491367e-10], [1.0])
    assert result[0] == pytest.approx(1.0e-29)


def test_completion_is_solar_at_high_curvature_and_bounded_at_low_curvature():
    result = bounded_completion(
        [1.0e-12, 1.0e-40],
        solar_completion=0.2,
        tidal_transition_s2=1.0e-30,
        transition_power=2.0,
    )
    assert result["completion_fraction"][0] == pytest.approx(0.2)
    assert result["enhancement_relative_to_local_G"][0] == pytest.approx(1.0)
    assert result["completion_fraction"][1] == pytest.approx(1.0)
    assert result["enhancement_relative_to_local_G"][1] == pytest.approx(5.0)
    assert np.all(result["completion_fraction"] <= 1.0)


def test_coherence_can_only_remove_vectors_from_completion():
    available = bounded_completion(
        [1.0e-40],
        solar_completion=0.25,
        tidal_transition_s2=1.0e-30,
        transition_power=2.0,
        coherence=[0.0],
        coherence_power=2.0,
    )
    lost = bounded_completion(
        [1.0e-40],
        solar_completion=0.25,
        tidal_transition_s2=1.0e-30,
        transition_power=2.0,
        coherence=[1.0],
        coherence_power=2.0,
    )
    assert available["completion_fraction"][0] == pytest.approx(1.0)
    assert lost["completion_fraction"][0] == pytest.approx(0.25)


def test_predicted_acceleration_never_exceeds_proposed_true_maximum():
    gbar = np.asarray([1.0e-12, 1.0e-10])
    result = predict_completion_acceleration(gbar, [100.0, 1.0], [0.1, -30.0, 2.0])
    assert np.all(result["predicted_acceleration_m_s2"] <= gbar / 0.1)
    assert np.all(result["predicted_acceleration_m_s2"] >= gbar)
