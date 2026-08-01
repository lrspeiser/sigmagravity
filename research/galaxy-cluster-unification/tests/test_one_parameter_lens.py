import numpy as np
import pytest

from voidscreen.one_parameter_lens import predict_one_parameter_acceleration


def test_mass_isothermal_tail_has_one_over_r_extra_acceleration() -> None:
    radius = np.array([50.0, 100.0, 200.0, 400.0])
    gbar = 2.0e-10 * np.square(200.0 / radius)
    predicted = predict_one_parameter_acceleration(
        "mass_isothermal_tail",
        gbar,
        radius,
        3.0,
        gbar_at_reference_m_s2=2.0e-10,
    )
    extra = predicted - gbar
    np.testing.assert_allclose(extra * radius, np.full(4, extra[0] * radius[0]))


def test_scaled_rar_zero_is_exact_baryons_and_constant_boost_is_linear() -> None:
    gbar = np.array([1.0e-12, 1.0e-10, 1.0e-8])
    radius = np.array([1.0, 10.0, 100.0])
    np.testing.assert_allclose(
        predict_one_parameter_acceleration("scaled_rar_extra", gbar, radius, 0.0),
        gbar,
    )
    np.testing.assert_allclose(
        predict_one_parameter_acceleration("constant_boost", gbar, radius, 7.0),
        7.0 * gbar,
    )


def test_screened_tail_switches_off_at_high_acceleration() -> None:
    radius = np.array([200.0, 200.0])
    gbar = np.array([1.2e-14, 1.2e-4])
    predicted = predict_one_parameter_acceleration(
        "screened_mass_isothermal_tail",
        gbar,
        radius,
        9.0,
        gbar_at_reference_m_s2=1.2e-10,
    )
    extra = predicted - gbar
    assert extra[0] == pytest.approx(9.0 * 1.2e-10, rel=2.0e-4)
    assert extra[1] == pytest.approx(9.0 * 1.2e-10 * 1.0e-6, rel=2.0e-6)


def test_transition_power_has_newtonian_and_deep_mond_limits() -> None:
    gbar = np.array([1.0e-16, 1.0e-5])
    predicted = predict_one_parameter_acceleration(
        "rar_transition_power", gbar, np.array([1.0, 1.0]), 2.0
    )
    assert predicted[0] == pytest.approx(np.sqrt(1.2e-10 * gbar[0]), rel=1.0e-3)
    assert predicted[1] == pytest.approx(gbar[1], rel=1.0e-9)


def test_invalid_family_and_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        predict_one_parameter_acceleration("unknown", [1.0], [1.0], 1.0)
    with pytest.raises(ValueError):
        predict_one_parameter_acceleration("constant_boost", [1.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        predict_one_parameter_acceleration(
            "mass_isothermal_tail", [1.0], [1.0], 1.0
        )
