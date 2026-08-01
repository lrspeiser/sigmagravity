from pathlib import Path

import numpy as np

from voidscreen.halo_backtrack import (
    cluster_halo_columns,
    coarsen_source_map,
    component_samples,
    halo_assignment,
    posterior_component_destinations,
    solve_capacity_transport,
    thin_bayes_chain,
)
from voidscreen.route_template import (
    baryonic_network_transitions,
    conservative_network_route_template,
)
from voidscreen.stellar_morphology_lensing import (
    StellarMorphologyDeflectionField,
    blend_morphology_deflection_fields,
)


def test_chain_parser_excludes_shear_and_thins(tmp_path: Path):
    path = tmp_path / "bayes.dat"
    path.write_text(
        "#Nsample\n#ln(Lhood)\n#O1 : x (arcsec)\n#O1 : y (arcsec)\n"
        "#O1 : emass\n#O1 : theta (deg)\n#O1 : rc (arcsec)\n"
        "#O1 : sigma (km/s)\n#O2 : gamma\n#O2 : theta (deg)\n#Chi2\n"
        + "\n".join(
            f"1 {-i} {i} {-i} 0.2 30 5 800 0.1 20 {i}" for i in range(10)
        )
        + "\n",
        encoding="ascii",
    )
    headers, samples, rows = thin_bayes_chain(path, 4)
    assert rows == 10
    assert samples.shape == (4, 11)
    assert set(cluster_halo_columns(headers)) == {1}
    components = component_samples(headers, samples, 5.0)
    assert np.allclose(components[1]["x_kpc"], -samples[:, 2] * 5.0)


def test_posterior_destination_conserves_weight_and_identity():
    components = {
        1: {
            "x_kpc": np.array([0.0, 1.0]),
            "y_kpc": np.array([0.0, 0.0]),
            "core_kpc": np.array([5.0, 5.0]),
            "sigma_km_s": np.array([1000.0, 1000.0]),
            "emass": np.zeros(2),
            "theta_deg": np.zeros(2),
        },
        2: {
            "x_kpc": np.array([50.0, 51.0]),
            "y_kpc": np.array([0.0, 0.0]),
            "core_kpc": np.array([5.0, 5.0]),
            "sigma_km_s": np.array([500.0, 500.0]),
            "emass": np.zeros(2),
            "theta_deg": np.zeros(2),
        },
    }
    axis = np.arange(-100.0, 101.0, 10.0)
    position, weight, identity, maps = posterior_component_destinations(
        components,
        axis,
        width_mode="fixed",
        width_kpc=20.0,
        weight_mode="sigma2",
        maximum_radius_kpc=100.0,
        minimum_relative_density=1e-4,
    )
    assert len(position) == len(weight) == len(identity)
    assert np.isclose(weight.sum(), 1.0)
    assert set(identity) == {1, 2}
    assert np.isclose(maps[1].sum() / maps[2].sum(), 4.0)


def test_source_map_and_halo_assignment():
    image = np.zeros((8, 8))
    image[1, 1] = 1.0
    image[6, 6] = 3.0
    position, weight = coarsen_source_map(
        image,
        np.arange(-4.0, 4.0),
        2.0,
        factor=2,
        maximum_radius_kpc=20.0,
        retained_weight=1.0,
    )
    assert len(position) == 2
    assert np.isclose(weight.sum(), 1.0)
    conditional, marginal = halo_assignment(
        np.array([[0.2, 0.3], [0.1, 0.4]]), np.array([1, 2])
    )
    assert np.allclose(conditional.sum(axis=1), 1.0)
    assert np.allclose(marginal, [0.3, 0.7])


def test_capacity_transport_can_leave_distant_source_unused():
    source = np.array([[0.0, 0.0], [100.0, 0.0]])
    source_weight = np.array([0.5, 0.5])
    destination = np.array([[0.0, 0.0]])
    plan, audit = solve_capacity_transport(
        source,
        source_weight,
        destination,
        np.array([1.0]),
        capacity_multiplier=2.0,
        entropy_length_kpc=5.0,
    )
    assert np.isclose(plan.sum(), 1.0)
    assert plan[0, 0] > 0.999
    assert plan[1, 0] < 0.001
    assert audit["target_marginal_max_error"] < 1e-10
    assert audit["maximum_source_capacity_excess"] < 1e-10


def test_baryonic_network_kernel_is_normalized_and_excludes_self():
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]])
    weights = np.array([0.5, 0.3, 0.2])
    transition = baryonic_network_transitions(
        positions,
        weights,
        target_weight_power=1.0,
        distance_power=2.0,
        softening=5.0,
        link_scale=50.0,
    )
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(np.diag(transition), 0.0)
    assert transition[0, 1] > transition[0, 2]


def test_conservative_network_route_preserves_total_weight():
    axis = np.arange(-50.0, 51.0, 2.0)
    image, audit = conservative_network_route_template(
        axis,
        np.array([[-10.0, 0.0], [10.0, 0.0], [0.0, 15.0]]),
        np.array([0.4, 0.4, 0.2]),
        routing_fraction=0.4,
        target_weight_power=1.0,
        distance_power=2.0,
        softening=5.0,
        link_scale=30.0,
        hop_fraction=1.0,
        smoothing=3.0,
        top_k=1,
    )
    assert np.isclose(image.sum(), 1.0)
    assert audit["normalization_error"] < 1e-12
    assert audit["branch_count"] == 3
    assert audit["mean_route_length"] > 0.0


def test_morphology_field_blend_is_linear():
    axis = np.arange(-2.0, 3.0)
    radius = np.array([0.0, 10.0])
    zero = np.zeros((5, 5))
    left = StellarMorphologyDeflectionField(
        axis, np.ones((5, 5)), 2.0 * np.ones((5, 5)), radius, np.zeros(2), zero, zero, {}
    )
    right = StellarMorphologyDeflectionField(
        axis, 5.0 * np.ones((5, 5)), 6.0 * np.ones((5, 5)), radius, np.zeros(2), zero, zero, {}
    )
    blended = blend_morphology_deflection_fields(left, right, 0.25)
    alpha_x, alpha_y = blended.alpha_arcsec(
        np.array([0.0]), np.array([0.0]), distance_ratio=1.0
    )
    assert np.allclose(alpha_x, 2.0)
    assert np.allclose(alpha_y, 3.0)
