from __future__ import annotations

import numpy as np

from voidscreen.synthetic_universe import (
    GalaxySeed,
    RadialBaryonProfile,
    TRANSPORT_PARAMETER_NAMES,
    fixed_rar_acceleration,
    generate_galaxy_scene,
    predict_acceleration,
    radial_particle_acceleration_m_s2,
    simple_mond_acceleration,
    sobol_galaxy_population,
    stable_hash_partition,
    transport_acceleration_from_features,
)


def profile() -> RadialBaryonProfile:
    return RadialBaryonProfile(
        "fixture",
        np.array([0.5, 1.0, 2.0, 4.0, 8.0]),
        np.array([8.0e-10, 5.0e-10, 2.8e-10, 1.2e-10, 4.0e-11]),
    )


def test_profile_context_is_baryonic_and_finite():
    item = profile()
    context = item.context()
    assert item.total_mass_msun > 0.0
    assert 0.0 < item.r50_kpc <= item.r80_kpc
    assert context["mean_surface_density_msun_pc2"] > 0.0
    assert context["reference_gbar_m_s2"] > 0.0


def test_seeded_particle_scene_is_exactly_reproducible_and_mass_conserving():
    item = profile()
    seed = GalaxySeed(
        "fixture",
        item,
        disk_mass_msun=4.0e10,
        bulge_mass_msun=1.0e10,
        gas_mass_msun=8.0e9,
        disk_scale_kpc=2.5,
        bulge_scale_kpc=0.5,
        gas_scale_kpc=5.0,
        spiral_strength=0.2,
        random_seed=42,
    )
    first = generate_galaxy_scene(seed, 2048)
    second = generate_galaxy_scene(seed, 2048)
    np.testing.assert_array_equal(first.positions_kpc, second.positions_kpc)
    np.testing.assert_array_equal(first.masses_msun, second.masses_msun)
    assert len(first.positions_kpc) == 2048
    assert np.isclose(first.total_mass_msun, 5.8e10)
    acceleration = radial_particle_acceleration_m_s2(
        first, np.array([1.0, 2.0, 4.0]), softening_kpc=0.08
    )
    assert np.all(np.isfinite(acceleration))
    assert np.all(acceleration > 0.0)


def test_known_gravity_laws_have_expected_order_in_low_acceleration_regime():
    gbar = np.array([1.0e-12, 1.0e-10, 1.0e-8])
    rar = fixed_rar_acceleration(gbar)
    mond = simple_mond_acceleration(gbar)
    assert np.all(rar >= gbar)
    assert np.all(mond >= gbar)
    assert np.allclose(rar[-1], gbar[-1], rtol=0.01)
    assert np.allclose(mond[-1], gbar[-1], rtol=0.02)


def test_transport_is_universal_finite_and_solar_screened():
    parameters = np.array([-10.0, 2.0, 1.0, 2.0, 1.0, 1.7, 4.0])
    assert len(parameters) == len(TRANSPORT_PARAMETER_NAMES)
    gbar = np.array([1.0e-12, 1.0e-10, 1.0e-5])
    predicted = transport_acceleration_from_features(
        gbar,
        np.array([10.0, 10.0, 1.0e-8]),
        surface_density_msun_pc2=np.array([5.0, 100.0, 1.0e9]),
        r80_kpc=np.array([5.0, 20.0, 1.0e-8]),
        reference_gbar_m_s2=np.array([1.0e-13, 1.0e-13, 1.0e-5]),
        parameters=parameters,
    )
    assert np.all(np.isfinite(predicted))
    assert np.all(predicted >= gbar)
    assert abs(predicted[-1] / gbar[-1] - 1.0) < 1.0e-4


def test_plugin_prediction_matches_direct_transport_call():
    item = profile()
    parameters = np.array([-10.0, 2.0, 1.0, 2.0, 1.0, 1.7, 4.0])
    from_plugin = predict_acceleration("transport", item, parameters=parameters)
    context = item.context()
    direct = transport_acceleration_from_features(
        item.gbar_m_s2,
        item.radius_kpc,
        surface_density_msun_pc2=context["mean_surface_density_msun_pc2"],
        r80_kpc=context["r80_kpc"],
        reference_gbar_m_s2=context["reference_gbar_m_s2"],
        parameters=parameters,
    )
    np.testing.assert_allclose(from_plugin, direct)


def test_sobol_population_and_hash_split_are_reproducible():
    first = sobol_galaxy_population(1024, seed=7)
    second = sobol_galaxy_population(1024, seed=7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert np.all(np.isfinite(first[key]))
    labels = [f"G{index:03d}" for index in range(100)]
    split_a = stable_hash_partition(
        labels, salt="fixture", train_fraction=0.6, development_fraction=0.2
    )
    split_b = stable_hash_partition(
        labels, salt="fixture", train_fraction=0.6, development_fraction=0.2
    )
    assert split_a == split_b
    assert set(split_a.values()) == {"train", "development", "holdout"}
