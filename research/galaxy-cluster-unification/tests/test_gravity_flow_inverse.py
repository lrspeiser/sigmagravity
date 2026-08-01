import numpy as np

from voidscreen.gravity_flow_inverse import (
    coarsen_destination,
    local_projection_excess,
    map_similarity,
    off_plane_arc_length,
    rasterize_transport_paths,
    solve_transport,
    transport_diagnostics,
)


def test_local_projection_excess_removes_local_shape():
    aperture = np.ones((2, 2), dtype=bool)
    baryon = np.array([[3.0, 0.0], [0.0, 1.0]])
    target = np.array([[2.0, 1.0], [1.0, 2.0]])
    excess, fit = local_projection_excess(target, baryon, aperture)
    assert np.isclose(excess.sum(), 1.0)
    assert fit["fitted_local_projection"] > 0.0
    assert excess[0, 0] < target[0, 0] / target.sum()


def test_coarsen_and_transport_preserve_marginals():
    axis = np.arange(4, dtype=float)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    image = np.zeros((4, 4), dtype=float)
    image[0, 0] = 1.0
    image[2, 2] = 1.0
    destinations, destination_weights, _ = coarsen_destination(
        image, x_grid, y_grid, factor=2, radius_kpc=10.0
    )
    sources = np.array([[0.5, 0.5], [2.5, 2.5]])
    source_weights = np.array([0.4, 0.6])
    plan, _ = solve_transport(
        sources,
        source_weights,
        destinations,
        destination_weights,
        entropy_length_kpc=1.0,
    )
    assert np.allclose(plan.sum(axis=1), source_weights, atol=1e-8)
    assert np.allclose(plan.sum(axis=0), destination_weights, atol=1e-8)
    stats = transport_diagnostics(
        plan,
        sources,
        source_weights,
        destinations,
        destination_weights,
        baryonic_center=np.array([1.5, 1.5]),
    )
    assert stats["source_marginal_max_error"] < 1e-8
    assert stats["target_marginal_max_error"] < 1e-8


def test_path_raster_and_off_plane_arc_family():
    plan = np.array([[1.0]])
    source = np.array([[0.0, 0.0]])
    destination = np.array([[2.0, 0.0]])
    axis = np.arange(-1.0, 4.0)
    image = rasterize_transport_paths(
        plan, source, destination, axis, samples_per_path=5
    )
    assert np.isclose(image.sum(), 1.0)
    distance = np.array([2.0])
    assert np.allclose(off_plane_arc_length(distance, 0.0), distance)
    assert off_plane_arc_length(distance, 0.5)[0] > distance[0]
    assert map_similarity(image, image)["jensen_shannon"] == 0.0
    assert np.isclose(map_similarity(image, image)["pearson"], 1.0)
