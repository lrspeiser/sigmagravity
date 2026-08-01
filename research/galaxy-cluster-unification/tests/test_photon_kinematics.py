import numpy as np

from voidscreen.photon_kinematics import (
    circular_channel_velocities,
    circular_speed_from_channels,
    galactocentric_geometry,
    lsr_to_heliocentric_velocity,
    solar_galactocentric_velocity,
)


def test_channel_inversion_recovers_circular_speed():
    geometry = galactocentric_geometry(
        np.deg2rad([35.0, 120.0]),
        np.deg2rad([0.5, -1.0]),
        [4.0, 3.0],
        solar_radius_kpc=8.15,
    )
    solar = solar_galactocentric_velocity(236.0, [10.6, 10.7, 7.6])
    expected = np.asarray([225.0, 240.0])
    transverse, radial = circular_channel_velocities(
        geometry, expected, solar_velocity_km_s=solar
    )
    theta_pm, theta_rv = circular_speed_from_channels(
        geometry,
        transverse_longitude_velocity_km_s=transverse,
        heliocentric_radial_velocity_km_s=radial,
        solar_velocity_km_s=solar,
    )
    assert np.allclose(theta_pm, expected)
    assert np.allclose(theta_rv, expected)


def test_lsr_round_trip_sign_at_galactic_longitude_zero():
    heliocentric = lsr_to_heliocentric_velocity(
        [20.0],
        [0.0],
        [0.0],
        [10.3, 15.3, 7.7],
    )
    assert np.allclose(heliocentric, 9.7)


def test_solar_velocity_axis_convention():
    velocity = solar_galactocentric_velocity(236.0, [10.6, 10.7, 7.6])
    assert np.allclose(velocity, [-10.6, 246.7, 7.6])
