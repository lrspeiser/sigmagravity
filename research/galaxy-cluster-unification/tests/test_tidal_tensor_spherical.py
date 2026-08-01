import numpy as np
import pytest

from voidscreen.tidal_tensor_spherical import acceleration_gate, spherical_boost


def test_gate_transitions_at_a0():
    gate = acceleration_gate(
        np.array([1.2e-12, 1.2e-10, 1.2e-8]),
        a0_m_s2=1.2e-10,
        power=2.0,
    )
    assert gate[0] > 0.99
    assert np.isclose(gate[1], 0.5)
    assert gate[2] < 1.1e-4


def test_exponential_mapping_is_positive_and_unbounded():
    boost = spherical_boost(
        np.array([1e-20]),
        kappa=4.0,
        family="exponential",
        gate_power=1.0,
        a0_m_s2=1.2e-10,
    )
    assert boost[0] > 10.0


def test_reciprocal_mapping_has_linear_low_acceleration_limit():
    boost = spherical_boost(
        np.array([1e-20]),
        kappa=3.0,
        family="reciprocal",
        gate_power=1.0,
        a0_m_s2=1.2e-10,
    )
    assert np.isclose(boost[0], 3.0, rtol=1e-8)


def test_linear_mapping_rejects_kappa_one():
    with pytest.raises(ValueError):
        spherical_boost(
            np.array([1e-12]),
            kappa=1.0,
            family="linear",
            gate_power=1.0,
            a0_m_s2=1.2e-10,
        )
