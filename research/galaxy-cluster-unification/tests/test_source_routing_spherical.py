from __future__ import annotations

import numpy as np

from voidscreen.source_routing_spherical import source_conserving_spherical_response


def test_spherical_routing_is_positive_and_conserves_outer_flux():
    radius = np.geomspace(0.05, 200.0, 2048)
    mass = radius**3 / np.power(radius * radius + 1.0, 1.5)
    gbar = mass / np.square(radius)
    potential = 1.0 / np.sqrt(radius * radius + 1.0)
    depth = potential / 100.0**2
    path = potential / np.maximum(radius * gbar, 1e-20)
    response = source_conserving_spherical_response(
        radius,
        gbar,
        depth,
        path,
        a0_m_s2=0.03,
        transition_depth=1e-4,
        transition_power=4.0,
        extra_spatial_channels=2.0,
        path_power=0.5,
    )
    assert np.all(response.routed_acceleration_m_s2 > 0.0)
    assert response.positive_generator_strength_m3_s2 > 0.0
    assert response.net_added_flux_fraction < 1e-12
    assert np.isclose(
        response.routed_acceleration_m_s2[-1],
        response.base_acceleration_m_s2[-1],
        rtol=1e-12,
    )
