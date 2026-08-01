import numpy as np
import pytest

from voidscreen.solar_system_tail import (
    AU_M,
    PARSEC_M,
    analytic_unscreened_precession_mas_per_century,
    extra_tail_acceleration_m_s2,
    fractional_extra_force,
    secular_perihelion_precession_mas_per_century,
)


REFERENCE_RADIUS_M = 200_000.0 * PARSEC_M


def test_unscreened_fraction_is_lambda_r_over_reference_radius() -> None:
    radius = np.array([0.4, 1.0, 10.0]) * AU_M
    fraction = fractional_extra_force(
        radius,
        parameter=9.0,
        reference_radius_m=REFERENCE_RADIUS_M,
        screened=False,
    )
    np.testing.assert_allclose(fraction, 9.0 * radius / REFERENCE_RADIUS_M)


def test_screen_reduces_the_solar_tail_by_many_orders() -> None:
    unscreened = extra_tail_acceleration_m_s2(
        AU_M,
        parameter=9.0,
        reference_radius_m=REFERENCE_RADIUS_M,
        screened=False,
    )
    screened = extra_tail_acceleration_m_s2(
        AU_M,
        parameter=9.0,
        reference_radius_m=REFERENCE_RADIUS_M,
        screened=True,
    )
    assert 0.0 < screened / unscreened < 3.0e-8


def test_numerical_gauss_average_matches_unscreened_closed_form() -> None:
    values = {
        "semimajor_axis_m": 0.38709893 * AU_M,
        "eccentricity": 0.205630,
        "orbital_period_days": 87.9691,
        "parameter": 9.0,
        "reference_radius_m": REFERENCE_RADIUS_M,
    }
    numerical = secular_perihelion_precession_mas_per_century(
        **values, screened=False
    )
    analytic = analytic_unscreened_precession_mas_per_century(**values)
    assert numerical == pytest.approx(analytic, rel=2.0e-10)


def test_screened_mercury_precession_is_inside_conservative_margin() -> None:
    prediction = secular_perihelion_precession_mas_per_century(
        semimajor_axis_m=0.38709893 * AU_M,
        eccentricity=0.205630,
        orbital_period_days=87.9691,
        parameter=30.0,
        reference_radius_m=REFERENCE_RADIUS_M,
        screened=True,
    )
    assert abs(prediction) < 3.1
