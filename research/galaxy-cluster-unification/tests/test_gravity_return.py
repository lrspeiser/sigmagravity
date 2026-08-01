import numpy as np

from voidscreen.gravity_return import (
    jensen_shannon_divergence,
    normalized_ring_kernel,
    normalized_directional_ring_kernel,
    routed_arrival_map,
    semicircle_arc_geometry,
    source_origin_probabilities,
    transition_radius_arcsec,
)


def test_return_kernel_and_prediction_conserve_weight():
    axis = np.linspace(-50.0, 50.0, 101)
    baryon = np.zeros((101, 101))
    baryon[50, 50] = 2.0
    kernel = normalized_ring_kernel(axis, return_radius_arcsec=20.0, width_arcsec=3.0)
    prediction, arrival = routed_arrival_map(baryon, kernel, routed_fraction=0.4)
    assert np.isclose(kernel.sum(), 1.0)
    assert np.isclose(arrival.sum(), 1.0)
    assert np.isclose(prediction.sum(), 1.0)
    assert prediction[50, 50] > arrival[50, 50]


def test_backtracking_prefers_source_on_return_annulus():
    probability = source_origin_probabilities(
        [0.0, 20.0, 80.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 100.0],
        destination_x_arcsec=40.0,
        destination_y_arcsec=0.0,
        return_radius_arcsec=20.0,
        width_arcsec=2.0,
    )
    assert probability[1] > probability[0]
    assert probability[1] > probability[2]
    assert np.isclose(probability.sum(), 1.0)


def test_directional_ring_is_normalized_and_axis_sensitive():
    axis = np.linspace(-40.0, 40.0, 81)
    kernel = normalized_directional_ring_kernel(
        axis,
        return_radius_arcsec=20.0,
        width_arcsec=3.0,
        major_axis_deg=0.0,
        directional_concentration=2.0,
    )
    assert np.isclose(kernel.sum(), 1.0)
    center = len(axis) // 2
    assert kernel[center, center + 20] > kernel[center + 20, center]


def test_scale_and_diagnostics_are_well_behaved():
    radius = transition_radius_arcsec(5.0e13, a0_m_s2=2.4e-10, scale_kpc_per_arcsec=5.5)
    assert 20.0 < radius < 50.0
    assert jensen_shannon_divergence([1.0, 0.0], [1.0, 0.0]) == 0.0
    geometry = semicircle_arc_geometry(20.0, 5.5)
    assert geometry["maximum_hidden_height_kpc"] == 55.0
