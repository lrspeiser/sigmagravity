import numpy as np

from voidscreen.photon_path_scaling import (
    baryonic_speed,
    path_feature,
    rar_speed,
    weighted_fit,
)


def test_baryonic_speed_uses_signed_gas_square():
    result = baryonic_speed(
        np.array([-3.0, 3.0]),
        np.array([5.0, 5.0]),
        np.array([0.0, 0.0]),
        disk_mass_to_light=1.0,
        bulge_mass_to_light=1.0,
    )
    assert np.allclose(result, [4.0, np.sqrt(34.0)])


def test_rar_speed_exceeds_baryonic_speed_at_low_acceleration():
    speed, acceleration = rar_speed(
        np.array([10.0]),
        np.array([30.0]),
        acceleration_scale_m_s2=1.2e-10,
    )
    assert acceleration[0] < 1.2e-10
    assert speed[0] > 30.0


def test_saturating_path_is_bounded():
    feature = path_feature(
        np.array([0.1, 1.0, 100.0]), kind="saturating", scale=1.0
    )
    assert np.all((feature > 0.0) & (feature <= 1.0))
    assert np.all(np.diff(feature) > 0.0)


def test_weighted_fit_recovers_path_amplitude():
    feature = np.array([-1.0, 0.0, 1.0, 2.0])
    values, _ = weighted_fit(feature, 4.0 + 3.0 * feature, np.ones(4))
    assert np.allclose(values, [4.0, 3.0])
