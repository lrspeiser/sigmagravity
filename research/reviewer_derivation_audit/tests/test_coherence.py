import numpy as np
import pytest

from sigma_sprint.coherence import (
    assert_no_outcome_leakage,
    coherence_from_moments,
    phase_space_coherence,
)


def circular_samples(counterrotating=False):
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    velocities = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
    if counterrotating:
        velocities[::2] *= -1
    return positions, velocities


def test_moment_coherence_bounds():
    assert coherence_from_moments([10, 0, 0], np.zeros((3, 3))) == 1.0
    assert coherence_from_moments([0, 0, 0], np.eye(3)) == 0.0
    assert np.isclose(coherence_from_moments([1, 0, 0], np.eye(3)), 0.25)


def test_counterrotation_suppresses_phase_space_coherence():
    ordered = phase_space_coherence(*circular_samples(False), axis=[0, 0, 1])
    counter = phase_space_coherence(*circular_samples(True), axis=[0, 0, 1])
    assert ordered > 0.999
    assert counter < 1e-12


def test_outcome_leakage_is_rejected():
    assert_no_outcome_leakage(["mean_v_phi", "sigma_phi"])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_outcome_leakage(["predicted_velocity", "sigma_phi"])
