import numpy as np

from sigma_sprint.auxiliary_field import euler_residual_spherical


def test_action_euler_equation_for_quadratic_field():
    radius = np.linspace(1.0, 20.0, 200)
    field = radius**2
    ell = 2.0
    beta = 3.0
    source = (field - 6.0 * ell**2) / beta
    residual = euler_residual_spherical(radius, field, source, ell, beta)
    assert np.max(np.abs(residual[3:-3])) < 1e-9
