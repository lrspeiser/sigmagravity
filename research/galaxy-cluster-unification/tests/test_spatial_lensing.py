import numpy as np

from voidscreen.spatial_lensing import MemberRedistributionField


def test_member_contrast_removes_circular_mean_and_far_field_mass():
    impact = np.geomspace(0.05, 500.0, 500)
    field = MemberRedistributionField.build(
        np.array([-8.0, 4.0, 15.0]),
        np.array([2.0, -5.0, 3.0]),
        np.array([0.2, 0.5, 0.3]),
        total_mass_msun=2.0e13,
        lens_angular_diameter_distance_mpc=1000.0,
        softening_arcsec=1.0,
        impact_arcsec=impact,
        azimuth_samples=512,
    )
    phi = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
    radii = np.array([2.0, 10.0, 40.0, 200.0])
    x = radii[:, None] * np.cos(phi)[None, :]
    y = radii[:, None] * np.sin(phi)[None, :]
    alpha_x, alpha_y = field.contrast_alpha_arcsec(x, y, distance_ratio=0.6)
    radial = np.mean(
        alpha_x * np.cos(phi)[None, :] + alpha_y * np.sin(phi)[None, :], axis=1
    )
    assert field.total_mass_msun == 2.0e13
    assert np.max(np.abs(radial)) < 2.0e-3


def test_member_discrete_deflection_scales_with_distance_ratio():
    field = MemberRedistributionField.build(
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        total_mass_msun=1.0e12,
        lens_angular_diameter_distance_mpc=800.0,
        softening_arcsec=0.5,
        impact_arcsec=np.geomspace(0.05, 100.0, 200),
        azimuth_samples=128,
    )
    alpha1 = field.discrete_alpha_arcsec(10.0, 0.0, distance_ratio=0.2)[0]
    alpha2 = field.discrete_alpha_arcsec(10.0, 0.0, distance_ratio=0.7)[0]
    assert np.isclose(alpha2 / alpha1, 3.5)
