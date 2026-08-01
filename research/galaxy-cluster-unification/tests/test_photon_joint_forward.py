import numpy as np
import pandas as pd

from voidscreen.photon_joint_forward import (
    channel_design,
    fit_weighted_linear,
    photon_features,
)


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system": ["a", "b", "c", "d"],
            "radius_kpc": [5.0, 7.0, 9.0, 12.0],
            "l_deg": [20.0, 80.0, 170.0, 260.0],
            "b_deg": [0.1, -0.2, 0.3, -0.1],
            "longitude_projection": [0.8, -0.6, 0.5, -0.9],
            "radial_projection": [0.5, 0.7, -0.8, -0.6],
            "v_longitude_mc_median_km_s": [1.0, 2.0, 3.0, 4.0],
            "v_helio_radial_mc_median_km_s": [5.0, 6.0, 7.0, 8.0],
            "v_longitude_mc_sigma_km_s": [2.0] * 4,
            "v_helio_radial_mc_sigma_km_s": [3.0] * 4,
        }
    )


def test_constant_photon_feature_is_one():
    feature, names = photon_features(
        np.array([4.0, 8.0]),
        model="frequency_constant",
        r0_kpc=8.0,
        theta_reference_km_s=230.0,
        a_star_m_s2=1.2e-10,
    )
    assert names == ["photon_A_km_s"]
    assert np.all(feature == 1.0)


def test_photon_feature_only_enters_radial_channel():
    built = channel_design(
        synthetic_frame(),
        rotation_order=1,
        photon_model="frequency_constant",
        r0_kpc=8.15,
        theta_reference_km_s=236.0,
        a_star_m_s2=1.2e-10,
        fixed_solar_z_km_s=7.6,
        velocity_error_floor_km_s=1.0,
    )
    design = built["design"]
    assert np.all(design[:4, -1] == 0.0)
    assert np.allclose(design[4:, -1], synthetic_frame()["radial_projection"])


def test_weighted_linear_recovers_exact_coefficients():
    design = np.array(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    )
    truth = np.array([4.0, -1.5])
    fit = fit_weighted_linear(design, design @ truth, np.ones(4))
    assert np.allclose(fit["values"], truth)
    assert fit["chi2"] < 1e-20
