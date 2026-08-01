import numpy as np
import pytest

from voidscreen.member_routing import normalized_member_weights


def test_identity_and_total_budget_are_preserved():
    mass = np.array([1.0, 2.0, 8.0])
    identity = normalized_member_weights(mass)
    changed = normalized_member_weights(
        mass, mass_power=1.5, radial_dressing=np.array([2.0, 1.0, 0.5])
    )
    assert np.allclose(identity, mass)
    assert np.isclose(changed.sum(), mass.sum())


def test_larger_mass_power_concentrates_the_routing_weight():
    mass = np.array([1.0, 2.0, 8.0])
    shallow = normalized_member_weights(mass, mass_power=0.5)
    steep = normalized_member_weights(mass, mass_power=1.5)
    assert steep[-1] / steep[0] > shallow[-1] / shallow[0]


def test_invalid_routing_inputs_fail_loudly():
    with pytest.raises(ValueError):
        normalized_member_weights([1.0, -0.1])
    with pytest.raises(ValueError):
        normalized_member_weights([0.0, 0.0])
    with pytest.raises(ValueError):
        normalized_member_weights([1.0, 2.0], radial_dressing=[1.0])
