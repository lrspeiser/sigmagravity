import numpy as np
import pytest

from voidscreen.tidal_tensor_response import (
    normalized_squared_tidal,
    response_tensor,
    solar_gate,
)


def test_spherical_tidal_direction_has_expected_eigenvalues():
    tidal = np.diag([2.0, -1.0, -1.0])
    direction = normalized_squared_tidal(tidal)
    assert np.allclose(np.diag(direction), [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
    assert np.isclose(np.trace(direction), 1.0)


def test_response_stays_positive_definite():
    tidal = np.array(
        [[1.2, 0.4, -0.2], [0.4, -0.7, 0.1], [-0.2, 0.1, -0.5]]
    )
    response = response_tensor(
        tidal, 0.0, kappa=0.99, a0_m_s2=1.2e-10
    )
    assert np.min(np.linalg.eigvalsh(response)) > 0.0


def test_high_acceleration_screening_is_quadratic():
    gate = solar_gate(np.array([1.2e-10, 1.2e-5]), a0_m_s2=1.2e-10)
    assert np.isclose(gate[0], 0.5)
    assert gate[1] < 1.1e-10


def test_kappa_one_is_rejected():
    with pytest.raises(ValueError):
        response_tensor(
            np.eye(3), 0.0, kappa=1.0, a0_m_s2=1.2e-10
        )


def test_exponential_mapping_is_positive_for_large_kappa():
    response = response_tensor(
        np.diag([2.0, -1.0, -1.0]),
        0.0,
        kappa=8.0,
        a0_m_s2=1.2e-10,
        mapping="exponential",
    )
    assert np.min(np.linalg.eigvalsh(response)) > 0.0
    assert np.min(np.linalg.eigvalsh(response)) < 0.01
