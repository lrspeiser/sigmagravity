from __future__ import annotations

import numpy as np
import pytest

from voidscreen.constitutive import (
    required_response,
    simple_mu_acceleration,
    standard_mu_acceleration,
)
from voidscreen.unified import rar_acceleration


def test_constitutive_inverses_round_trip() -> None:
    gbar = np.geomspace(1e-14, 1e-8, 200)
    observed = 1.7 * np.sqrt(gbar * 1.2e-10) + gbar
    target = required_response(gbar, observed)
    assert target["inverse_valid"].all()

    rar = rar_acceleration(gbar, target["rar_a_eff_m_s2"])
    simple = simple_mu_acceleration(gbar, target["simple_a_x_m_s2"])
    standard = standard_mu_acceleration(gbar, target["standard_a_x_m_s2"])
    np.testing.assert_allclose(rar, observed, rtol=1e-12)
    np.testing.assert_allclose(simple, observed, rtol=1e-12)
    np.testing.assert_allclose(standard, observed, rtol=1e-12)


def test_non_excess_point_is_retained_as_invalid() -> None:
    target = required_response(np.asarray([2.0, 1.0]), np.asarray([1.0, 2.0]))
    assert not target["inverse_valid"][0]
    assert np.isnan(target["rar_a_eff_m_s2"][0])
    assert np.isnan(target["simple_a_x_m_s2"][0])
    assert np.isnan(target["standard_a_x_m_s2"][0])
    assert target["inverse_valid"][1]


@pytest.mark.parametrize(
    ("gbar", "observed"),
    [([1.0], [0.0]), ([0.0], [1.0]), ([1.0], [np.nan])],
)
def test_required_response_rejects_nonphysical_inputs(gbar, observed) -> None:
    with pytest.raises(ValueError):
        required_response(np.asarray(gbar), np.asarray(observed))
