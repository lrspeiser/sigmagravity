import numpy as np

from voidscreen.reopened_hybrids import (
    apply_channel_gate_memory_to_response,
    apply_radial_diffusion_to_response,
    apply_radial_memory_to_response,
    mercury_precession_mas_per_century,
    no_flux_log_radius_diffusion,
    radial_memory_blend,
    screened_hybrid_profile_response,
    screened_hybrid_response,
    solar_system_diagnostics,
    tidal_shape_property,
)


PARAMETERS = np.array([0.25, -24.0, 1.0, 2.0])


def test_interaction_eta_connects_additive_to_product_excess():
    gbar = np.array([1.0e-11, 3.0e-11])
    density = np.array([1.0e-26, 3.0e-25])
    radius = np.array([10.0, 100.0])
    additive = screened_hybrid_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {"combination": "interaction", "interaction_eta": 0.0},
    )
    product = screened_hybrid_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {"combination": "interaction", "interaction_eta": 1.0},
    )
    expected_difference = (
        additive["acceleration_screen"]
        * additive["rg_excess"]
        * additive["sigma_excess"]
    )
    assert np.allclose(
        product["fractional_excess"] - additive["fractional_excess"],
        expected_difference,
    )


def test_saturation_caps_pre_screen_combined_excess():
    response = screened_hybrid_response(
        [1.0e-13],
        [1.0e-30],
        [100.0],
        [0.03, -24.0, 1.0, 20.0],
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "saturation_ceiling": 3.0,
        },
    )
    assert response["raw_combined_excess"][0] > 3.0
    assert response["fractional_excess"][0] <= 3.0


def test_sigma_channel_can_saturate_without_capping_rg_channel():
    response = screened_hybrid_response(
        [1.0e-13],
        [1.0e-30],
        [100.0],
        [0.03, -24.0, 1.0, 20.0],
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "sigma_saturation_ceiling": 1.0,
        },
    )
    assert response["sigma_excess"][0] > 1.0
    assert response["effective_sigma_excess"][0] <= 1.0
    assert response["effective_rg_excess"][0] == response["rg_excess"][0]
    assert response["rg_saturation_fraction"][0] == 1.0


def test_rg_channel_can_saturate_without_capping_sigma_channel():
    response = screened_hybrid_response(
        [1.0e-13],
        [1.0e-30],
        [100.0],
        [0.03, -24.0, 1.0, 20.0],
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "rg_saturation_ceiling": 2.0,
        },
    )
    assert response["rg_excess"][0] > 2.0
    assert response["effective_rg_excess"][0] <= 2.0
    assert response["effective_sigma_excess"][0] == response["sigma_excess"][0]
    assert response["sigma_saturation_fraction"][0] == 1.0


def test_mass_gate_saturates_sigma_at_galaxy_mass_and_rg_at_cluster_mass():
    radius = np.array([10.0, 1000.0])
    equivalent_mass_msun = np.array([1.0e10, 1.0e14])
    radius_m = radius * 3.085677581491367e19
    gbar = (
        6.67430e-11
        * equivalent_mass_msun
        * 1.988409870698051e30
        / np.square(radius_m)
    )
    response = screened_hybrid_response(
        gbar,
        [1.0e-24, 1.0e-27],
        radius,
        [0.03, -24.0, 1.0, 20.0],
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "rg_saturation_ceiling": 2.75,
            "sigma_saturation_ceiling": 6.5,
            "channel_gate_property": "equivalent_mass",
            "channel_gate_log10_pivot": 11.5,
            "channel_gate_sharpness": 4.0,
            "channel_gate_cluster_high": True,
        },
    )
    gate = response["channel_gate_cluster_weight"]
    assert gate[0] < 0.01
    assert gate[1] > 0.99
    assert response["sigma_saturation_fraction"][0] < 1.0
    assert response["rg_saturation_fraction"][1] < 1.0
    assert response["rg_saturation_fraction"][0] > 0.99


def test_density_ratio_gate_can_define_cluster_as_low_property():
    response = screened_hybrid_response(
        [1.0e-11, 1.0e-11],
        [1.0e-28, 1.0e-22],
        [10.0, 10.0],
        PARAMETERS,
        {
            "channel_gate_property": "local_to_mean_density",
            "channel_gate_log10_pivot": 0.0,
            "channel_gate_sharpness": 3.0,
            "channel_gate_cluster_high": False,
        },
    )
    gate = response["channel_gate_cluster_weight"]
    assert gate[0] > gate[1]


def test_channel_cap_roles_can_be_selected_independently():
    response = screened_hybrid_response(
        [1.0e-13, 1.0e-13],
        [1.0e-28, 1.0e-22],
        [10.0, 10.0],
        [0.03, -24.0, 1.0, 20.0],
        {
            "rg_saturation_ceiling": 2.75,
            "sigma_saturation_ceiling": 6.5,
            "channel_gate_property": "local_to_mean_density",
            "channel_gate_log10_pivot": 0.0,
            "channel_gate_sharpness": 3.0,
            "channel_gate_cluster_high": False,
            "rg_cap_cluster_weight": False,
            "sigma_cap_cluster_weight": True,
        },
    )
    rg_weight = response["rg_channel_cap_weight"]
    sigma_weight = response["sigma_channel_cap_weight"]
    assert rg_weight[0] < rg_weight[1]
    assert sigma_weight[0] > sigma_weight[1]
    assert np.allclose(rg_weight + sigma_weight, 1.0)


def test_tidal_shape_gate_accepts_axisymmetric_property_override():
    eigenvalues = np.array(
        [
            [-2.0, 1.0, 1.0],
            [-0.2, 0.9, 3.0],
        ]
    )
    dominance = tidal_shape_property(eigenvalues, "tidal_l1_dominance")
    response = screened_hybrid_response(
        [1.0e-12, 1.0e-12],
        [1.0e-25, 1.0e-25],
        [10.0, 10.0],
        PARAMETERS,
        {
            "channel_gate_property": "tidal_l1_dominance",
            "channel_gate_pivot": 0.55,
            "channel_gate_sharpness": 20.0,
            "channel_gate_cluster_high": False,
        },
        channel_gate_property_values=dominance,
    )
    assert dominance[0] < dominance[1]
    assert (
        response["channel_gate_cluster_weight"][0]
        > response["channel_gate_cluster_weight"][1]
    )


def test_extended_tidal_shape_invariants_match_point_mass_tensor():
    eigenvalues = np.array([[-2.0, 1.0, 1.0]])
    assert np.allclose(
        tidal_shape_property(eigenvalues, "tidal_traceless_fraction"),
        [1.0],
    )
    assert np.allclose(
        tidal_shape_property(eigenvalues, "tidal_trace_fraction"),
        [0.0],
    )
    assert np.allclose(
        tidal_shape_property(eigenvalues, "tidal_positive_fraction"),
        [0.5],
    )
    assert np.allclose(
        tidal_shape_property(eigenvalues, "tidal_radial_abs_fraction"),
        [0.5],
    )
    assert np.allclose(
        tidal_shape_property(eigenvalues, "tidal_signed_determinant_shape"),
        [-2.0 / np.sqrt(216.0)],
    )


def test_signed_tidal_gate_accepts_negative_coordinate():
    signed_determinant = np.array([-0.16, -0.04])
    response = screened_hybrid_response(
        [1.0e-12, 1.0e-12],
        [1.0e-25, 1.0e-25],
        [10.0, 10.0],
        PARAMETERS,
        {
            "channel_gate_property": "tidal_signed_determinant_shape",
            "channel_gate_pivot": -0.10,
            "channel_gate_sharpness": 20.0,
            "channel_gate_cluster_high": False,
        },
        channel_gate_property_values=signed_determinant,
    )
    assert (
        response["channel_gate_cluster_weight"][0]
        > response["channel_gate_cluster_weight"][1]
    )


def test_channel_gate_memory_strength_zero_is_exactly_local():
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    settings = {
        "combination": "interaction",
        "interaction_eta": 1.0,
        "rg_saturation_ceiling": 2.75,
        "channel_gate_property": "tidal_middle_to_max",
        "channel_gate_pivot": 0.685,
        "channel_gate_sharpness": 5.0,
        "channel_gate_cluster_high": True,
        "rg_cap_cluster_weight": True,
        "channel_gate_memory_strength": 0.0,
        "channel_gate_memory_log_scale": 0.35,
    }
    local = screened_hybrid_response(
        np.full(4, 1.0e-12),
        np.full(4, 1.0e-25),
        radius,
        PARAMETERS,
        settings,
        channel_gate_property_values=np.array([0.3, 0.5, 0.7, 0.9]),
    )
    remembered = apply_channel_gate_memory_to_response(
        local, radius, settings
    )
    assert np.array_equal(
        remembered["channel_gate_cluster_weight"],
        local["channel_gate_cluster_weight"],
    )
    assert np.allclose(remembered["enhancement"], local["enhancement"])


def test_channel_gate_memory_carries_inner_geometry_outward():
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    settings = {
        "combination": "interaction",
        "interaction_eta": 1.0,
        "rg_saturation_ceiling": 2.75,
        "channel_gate_property": "tidal_middle_to_max",
        "channel_gate_pivot": 0.5,
        "channel_gate_sharpness": 20.0,
        "channel_gate_cluster_high": True,
        "rg_cap_cluster_weight": True,
        "channel_gate_memory_strength": 1.0,
        "channel_gate_memory_log_scale": 1.0,
    }
    local = screened_hybrid_response(
        np.full(4, 1.0e-13),
        np.full(4, 1.0e-28),
        radius,
        [0.03, -24.0, 1.0, 10.0],
        settings,
        channel_gate_property_values=np.array([0.9, 0.7, 0.3, 0.1]),
    )
    remembered = apply_channel_gate_memory_to_response(
        local, radius, settings
    )
    local_weight = local["channel_gate_cluster_weight"]
    memory_weight = remembered["channel_gate_cluster_weight"]
    assert memory_weight[0] == local_weight[0]
    assert memory_weight[-1] > local_weight[-1]
    assert remembered["effective_rg_excess"][-1] < local["effective_rg_excess"][-1]


def test_profile_response_applies_gate_memory_without_force_memory():
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    base = {
        "combination": "interaction",
        "interaction_eta": 1.0,
        "rg_saturation_ceiling": 2.75,
        "channel_gate_property": "tidal_middle_to_max",
        "channel_gate_pivot": 0.7,
        "channel_gate_sharpness": 8.0,
        "channel_gate_cluster_high": True,
        "channel_gate_memory_log_scale": 0.5,
        "radial_memory_strength": 0.0,
    }
    property_values = np.array([0.9, 0.8, 0.4, 0.2])
    local = screened_hybrid_profile_response(
        np.full(4, 1.0e-13),
        np.full(4, 1.0e-28),
        radius,
        [0.03, -24.0, 1.0, 10.0],
        {**base, "channel_gate_memory_strength": 0.0},
        channel_gate_property_values=property_values,
    )
    remembered = screened_hybrid_profile_response(
        np.full(4, 1.0e-13),
        np.full(4, 1.0e-28),
        radius,
        [0.03, -24.0, 1.0, 10.0],
        {**base, "channel_gate_memory_strength": 1.0},
        channel_gate_property_values=property_values,
    )
    assert not np.allclose(
        remembered["enhancement"], local["enhancement"], rtol=1.0e-6
    )
    assert np.all(
        remembered["radial_memory_blended_source"]
        == remembered["radial_memory_local_source"]
    )


def test_channel_gate_memory_requires_gate_and_valid_controls():
    local = screened_hybrid_response(
        [1.0e-12, 1.0e-13],
        [1.0e-25, 1.0e-26],
        [1.0, 2.0],
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_channel_gate_memory_to_response(
            local,
            [1.0, 2.0],
            {"channel_gate_memory_strength": 0.5},
        )
    with np.testing.assert_raises(ValueError):
        apply_channel_gate_memory_to_response(
            local,
            [1.0, 2.0],
            {"channel_gate_memory_strength": 1.1},
        )


def test_radial_memory_strength_zero_is_exactly_local():
    radius = np.array([1.0, 2.0, 4.0])
    local = np.array([0.2, 0.8, 2.0])
    blended, memory = radial_memory_blend(
        radius,
        local,
        strength=0.0,
        log_scale=1.0,
    )
    assert np.array_equal(blended, local)
    assert np.array_equal(memory, local)


def test_radial_memory_direction_changes_increasing_profile():
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    local = np.array([0.0, 1.0, 2.0, 3.0])
    outward, _ = radial_memory_blend(
        radius,
        local,
        strength=1.0,
        log_scale=1.0,
    )
    inward, _ = radial_memory_blend(
        radius,
        local,
        strength=1.0,
        log_scale=1.0,
        outer_to_inner=True,
    )
    assert np.all(outward[1:] < local[1:])
    assert np.all(inward[:-1] > local[:-1])
    assert outward[0] == local[0]
    assert inward[-1] == local[-1]


def test_disabled_cap_gate_preserves_global_caps_with_coordinate_present():
    radius = np.geomspace(1.0, 32.0, 8)
    gbar = 1.0e-10 / np.power(radius, 1.4)
    density = 1.0e-23 / np.square(radius)
    common = {
        "combination": "interaction",
        "interaction_eta": 1.0,
        "screen_power": 1.5,
        "rg_saturation_ceiling": 2.0,
        "sigma_saturation_ceiling": 1.5,
        "radial_memory_strength": 0.6,
        "radial_memory_log_scale": 0.8,
    }
    global_response = screened_hybrid_profile_response(
        gbar, density, radius, PARAMETERS, common
    )
    gated_response = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **common,
            "channel_gate_property": "tidal_middle_to_max",
            "channel_gate_pivot": 0.5,
            "channel_gate_sharpness": 8.0,
            "rg_cap_gate_enabled": False,
            "sigma_cap_gate_enabled": False,
        },
        channel_gate_property_values=np.linspace(0.1, 0.9, len(radius)),
    )
    np.testing.assert_array_equal(
        gated_response["rg_channel_cap_weight"], np.ones_like(radius)
    )
    np.testing.assert_array_equal(
        gated_response["sigma_channel_cap_weight"], np.ones_like(radius)
    )
    np.testing.assert_allclose(
        gated_response["enhancement"], global_response["enhancement"]
    )


def test_channel_and_complement_can_gate_radial_memory_strength():
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    gbar = np.array([8.0e-11, 4.0e-11, 2.0e-11, 1.0e-11])
    density = np.full_like(radius, 1.0e-25)
    common = {
        "channel_gate_property": "tidal_middle_to_max",
        "channel_gate_pivot": 0.5,
        "channel_gate_sharpness": 10.0,
        "channel_gate_cluster_high": True,
        "rg_cap_gate_enabled": False,
        "sigma_cap_gate_enabled": False,
        "radial_memory_strength": 0.8,
        "radial_memory_log_scale": 1.0,
    }
    coordinates = np.array([0.9, 0.7, 0.3, 0.1])
    channel = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**common, "radial_memory_gate_mode": "channel"},
        channel_gate_property_values=coordinates,
    )
    complement = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**common, "radial_memory_gate_mode": "complement"},
        channel_gate_property_values=coordinates,
    )
    np.testing.assert_allclose(
        channel["radial_memory_gate_weight"]
        + complement["radial_memory_gate_weight"],
        1.0,
    )
    for response in (channel, complement):
        expected = response["radial_memory_local_source"] + response[
            "radial_memory_effective_strength"
        ] * (
            response["radial_memory_average"]
            - response["radial_memory_local_source"]
        )
        np.testing.assert_allclose(
            response["radial_memory_blended_source"], expected
        )
    assert channel["radial_memory_effective_strength"][0] > complement[
        "radial_memory_effective_strength"
    ][0]
    assert channel["radial_memory_effective_strength"][-1] < complement[
        "radial_memory_effective_strength"
    ][-1]


def test_gated_radial_memory_requires_valid_mode_and_channel_property():
    radius = np.array([1.0, 2.0])
    local = screened_hybrid_response(
        [1.0e-10, 5.0e-11],
        [1.0e-24, 5.0e-25],
        radius,
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {
                "radial_memory_strength": 1.0,
                "radial_memory_gate_mode": "channel",
            },
        )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {
                "radial_memory_strength": 1.0,
                "radial_memory_gate_mode": "unknown",
            },
        )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {"radial_memory_strength": 1.1},
        )


def test_profile_response_memory_preserves_constant_excess_shape():
    radius = np.geomspace(1.0, 100.0, 12)
    gbar = np.full_like(radius, 1.0e-11)
    density = np.full_like(radius, 1.0e-25)
    local = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "screen_power": 1.5,
            "radial_memory_strength": 0.0,
        },
    )
    memory = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "screen_power": 1.5,
            "radial_memory_strength": 1.0,
            "radial_memory_log_scale": 2.0,
        },
    )
    np.testing.assert_allclose(
        memory["enhancement"], local["enhancement"]
    )


def test_zero_transport_powers_reproduce_legacy_fractional_memory():
    radius = np.geomspace(1.0, 100.0, 12)
    gbar = 1.0e-10 / radius
    density = 1.0e-23 / np.square(radius)
    legacy = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "radial_memory_strength": 0.7,
            "radial_memory_log_scale": 1.3,
        },
    )
    explicit = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "radial_memory_strength": 0.7,
            "radial_memory_log_scale": 1.3,
            "radial_memory_gbar_power": 0.0,
            "radial_memory_radius_power": 0.0,
            "radial_memory_channel_code": 0,
        },
    )
    np.testing.assert_array_equal(
        explicit["enhancement"], legacy["enhancement"]
    )


def test_acceleration_transport_differs_from_fractional_transport():
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    gbar = np.array([8.0e-11, 4.0e-11, 2.0e-11, 1.0e-11])
    density = np.full_like(radius, 1.0e-25)
    local = screened_hybrid_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {"screen_power": 1.5},
    )
    fractional = apply_radial_memory_to_response(
        local,
        radius,
        {
            "radial_memory_strength": 1.0,
            "radial_memory_log_scale": 10.0,
            "radial_memory_gbar_power": 0.0,
        },
    )
    acceleration = apply_radial_memory_to_response(
        local,
        radius,
        {
            "radial_memory_strength": 1.0,
            "radial_memory_log_scale": 10.0,
            "radial_memory_gbar_power": 1.0,
        },
    )
    assert acceleration["fractional_excess"][-1] > fractional[
        "fractional_excess"
    ][-1]
    assert acceleration["radial_memory_source_factor"][0] > acceleration[
        "radial_memory_source_factor"
    ][-1]


def test_channel_memory_can_move_rg_without_moving_sigma():
    radius = np.geomspace(1.0, 100.0, 12)
    gbar = 1.0e-10 / radius
    density = 1.0e-22 / np.power(radius, 2.5)
    local = screened_hybrid_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "rg_saturation_ceiling": 2.0,
            "sigma_saturation_ceiling": 1.5,
        },
    )
    rg_memory = apply_radial_memory_to_response(
        local,
        radius,
        {
            "combination": "interaction",
            "interaction_eta": 1.0,
            "rg_saturation_ceiling": 2.0,
            "sigma_saturation_ceiling": 1.5,
            "radial_memory_strength": 1.0,
            "radial_memory_log_scale": 2.0,
            "radial_memory_channel_code": 1,
        },
    )
    np.testing.assert_array_equal(
        rg_memory["memory_effective_sigma_excess"],
        local["effective_sigma_excess"],
    )
    assert np.any(
        rg_memory["memory_effective_rg_excess"]
        != local["effective_rg_excess"]
    )


def test_channel_memory_rejects_combined_pre_screen_flag():
    radius = np.array([1.0, 2.0])
    local = screened_hybrid_response(
        [1.0e-11, 5.0e-12],
        [1.0e-25, 5.0e-26],
        radius,
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {
                "radial_memory_strength": 1.0,
                "radial_memory_channel_code": 1,
                "radial_memory_pre_screen": True,
            },
        )


def test_zero_slope_gate_strength_is_exactly_legacy_carrier():
    radius = np.geomspace(1.0, 100.0, 12)
    gbar = 1.0e-10 / np.power(radius, 1.4)
    density = 1.0e-23 / np.square(radius)
    legacy = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "radial_memory_strength": 1.0,
            "radial_memory_log_scale": 2.0,
            "radial_memory_gbar_power": -1.0,
            "radial_memory_radius_power": -0.5,
        },
    )
    gated_off = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            "radial_memory_strength": 1.0,
            "radial_memory_log_scale": 2.0,
            "radial_memory_gbar_power": -1.0,
            "radial_memory_radius_power": -0.5,
            "radial_memory_slope_gate_strength": 0.0,
            "radial_memory_slope_gate_pivot": 100.0,
            "radial_memory_slope_gate_sharpness": -10.0,
            "radial_memory_steep_gbar_power": 5.0,
            "radial_memory_steep_radius_power": 5.0,
        },
    )
    np.testing.assert_array_equal(
        gated_off["radial_memory_source_factor"],
        legacy["radial_memory_source_factor"],
    )
    np.testing.assert_array_equal(
        gated_off["enhancement"], legacy["enhancement"]
    )


def test_inactive_slope_gate_does_not_add_profile_validation():
    radius = np.array([1.0, 1.0])
    local = screened_hybrid_response(
        [1.0e-10, 5.0e-11],
        [1.0e-24, 5.0e-25],
        radius,
        PARAMETERS,
        {},
    )
    response = apply_radial_memory_to_response(
        local,
        radius,
        {
            "radial_memory_strength": 0.0,
            "radial_memory_slope_gate_strength": 0.0,
        },
    )
    np.testing.assert_array_equal(response["enhancement"], local["enhancement"])


def test_slope_gate_responds_more_to_steep_baryonic_profile():
    radius = np.geomspace(1.0, 100.0, 20)
    density = 1.0e-24 / np.square(radius)
    settings = {
        "radial_memory_strength": 1.0,
        "radial_memory_log_scale": 2.0,
        "radial_memory_gbar_power": -1.0,
        "radial_memory_radius_power": -0.5,
        "radial_memory_slope_gate_strength": 1.0,
        "radial_memory_slope_gate_pivot": -1.0,
        "radial_memory_slope_gate_sharpness": 4.0,
        "radial_memory_steep_gbar_power": -0.5,
        "radial_memory_steep_radius_power": 1.5,
    }
    shallow = screened_hybrid_profile_response(
        1.0e-10 / np.power(radius, 0.25),
        density,
        radius,
        PARAMETERS,
        settings,
    )
    steep = screened_hybrid_profile_response(
        1.0e-10 / np.square(radius),
        density,
        radius,
        PARAMETERS,
        settings,
    )
    assert np.allclose(
        shallow["radial_memory_local_log_gbar_slope"], -0.25
    )
    assert np.allclose(
        steep["radial_memory_local_log_gbar_slope"], -2.0
    )
    assert np.all(
        steep["radial_memory_slope_gate_weight"]
        > shallow["radial_memory_slope_gate_weight"]
    )
    assert np.all(
        steep["radial_memory_effective_radius_power"]
        > shallow["radial_memory_effective_radius_power"]
    )


def test_slope_gate_rejects_invalid_strength_and_active_sharpness():
    radius = np.array([1.0, 2.0])
    local = screened_hybrid_response(
        [1.0e-10, 5.0e-11],
        [1.0e-24, 5.0e-25],
        radius,
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {"radial_memory_slope_gate_strength": 1.1},
        )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {
                "radial_memory_slope_gate_strength": 1.0,
                "radial_memory_slope_gate_sharpness": 0.0,
            },
        )


def test_profile_slope_gate_uses_one_measured_power_law_slope():
    radius = np.geomspace(1.0, 100.0, 20)
    response = screened_hybrid_profile_response(
        1.0e-10 / np.power(radius, 1.6),
        1.0e-24 / np.square(radius),
        radius,
        PARAMETERS,
        {
            "radial_memory_strength": 1.0,
            "radial_memory_gbar_power": -1.0,
            "radial_memory_radius_power": -0.5,
            "radial_memory_slope_gate_strength": 1.0,
            "radial_memory_slope_gate_mode": 1,
            "radial_memory_slope_gate_pivot": -1.0,
            "radial_memory_slope_gate_sharpness": 4.0,
            "radial_memory_steep_gbar_power": -0.5,
            "radial_memory_steep_radius_power": 1.5,
        },
    )
    np.testing.assert_allclose(
        response["radial_memory_slope_gate_coordinate"], -1.6
    )
    assert np.ptp(response["radial_memory_slope_gate_weight"]) == 0.0
    assert np.all(response["radial_memory_slope_gate_mode"] == 1.0)


def test_profile_and_pointwise_exponent_gates_match_on_exact_power_law():
    radius = np.geomspace(1.0, 100.0, 20)
    gbar = 1.0e-10 / np.power(radius, 1.6)
    density = 1.0e-24 / np.square(radius)
    settings = {
        "radial_memory_strength": 1.0,
        "radial_memory_log_scale": 2.0,
        "radial_memory_gbar_power": -1.0,
        "radial_memory_radius_power": -0.5,
        "radial_memory_slope_gate_strength": 1.0,
        "radial_memory_slope_gate_pivot": -1.0,
        "radial_memory_slope_gate_sharpness": 4.0,
        "radial_memory_steep_gbar_power": -0.5,
        "radial_memory_steep_radius_power": 1.5,
    }
    pointwise = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**settings, "radial_memory_slope_gate_mode": 0},
    )
    profile = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**settings, "radial_memory_slope_gate_mode": 1},
    )
    np.testing.assert_allclose(
        profile["radial_memory_slope_gate_weight"],
        pointwise["radial_memory_slope_gate_weight"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        profile["enhancement"], pointwise["enhancement"], rtol=1.0e-12
    )


def test_profile_response_blend_is_bounded_by_completed_endpoints():
    radius = np.geomspace(1.0, 100.0, 20)
    gbar = 1.0e-10 / np.power(radius, 1.6)
    density = 1.0e-24 / np.square(radius)
    common = {
        "radial_memory_strength": 1.0,
        "radial_memory_log_scale": 2.0,
    }
    base = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **common,
            "radial_memory_gbar_power": -1.0,
            "radial_memory_radius_power": -0.5,
        },
    )
    steep = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **common,
            "radial_memory_gbar_power": -0.5,
            "radial_memory_radius_power": 1.5,
        },
    )
    blended = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **common,
            "radial_memory_gbar_power": -1.0,
            "radial_memory_radius_power": -0.5,
            "radial_memory_slope_gate_strength": 1.0,
            "radial_memory_slope_gate_mode": 2,
            "radial_memory_slope_gate_pivot": -1.0,
            "radial_memory_slope_gate_sharpness": 4.0,
            "radial_memory_steep_gbar_power": -0.5,
            "radial_memory_steep_radius_power": 1.5,
        },
    )
    lower = np.minimum(base["fractional_excess"], steep["fractional_excess"])
    upper = np.maximum(base["fractional_excess"], steep["fractional_excess"])
    assert np.all(blended["fractional_excess"] >= lower - 1.0e-12)
    assert np.all(blended["fractional_excess"] <= upper + 1.0e-12)
    assert np.ptp(blended["radial_memory_slope_gate_weight"]) == 0.0


def test_pointwise_response_blend_is_bounded_and_differs_from_exponent_gate():
    radius = np.geomspace(1.0, 100.0, 24)
    gbar = (
        1.0e-10
        / np.power(radius, 1.3)
        * np.exp(0.35 * np.sin(3.0 * np.log(radius)))
    )
    density = 1.0e-24 / np.square(radius)
    common = {
        "radial_memory_strength": 1.0,
        "radial_memory_log_scale": 2.0,
    }
    base = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **common,
            "radial_memory_gbar_power": -1.0,
            "radial_memory_radius_power": -0.5,
        },
    )
    steep = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **common,
            "radial_memory_gbar_power": -0.5,
            "radial_memory_radius_power": 1.5,
        },
    )
    gated_settings = {
        **common,
        "radial_memory_gbar_power": -1.0,
        "radial_memory_radius_power": -0.5,
        "radial_memory_slope_gate_strength": 1.0,
        "radial_memory_slope_gate_pivot": -1.0,
        "radial_memory_slope_gate_sharpness": 8.0,
        "radial_memory_steep_gbar_power": -0.5,
        "radial_memory_steep_radius_power": 1.5,
    }
    exponent_gate = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**gated_settings, "radial_memory_slope_gate_mode": 0},
    )
    response_gate = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**gated_settings, "radial_memory_slope_gate_mode": 3},
    )
    lower = np.minimum(base["fractional_excess"], steep["fractional_excess"])
    upper = np.maximum(base["fractional_excess"], steep["fractional_excess"])
    assert np.all(np.isfinite(response_gate["enhancement"]))
    assert np.all(response_gate["fractional_excess"] >= lower - 1.0e-12)
    assert np.all(response_gate["fractional_excess"] <= upper + 1.0e-12)
    assert not np.allclose(
        response_gate["fractional_excess"], exponent_gate["fractional_excess"]
    )


def test_active_slope_gate_rejects_invalid_mode_but_inactive_ignores_it():
    radius = np.array([1.0, 2.0])
    local = screened_hybrid_response(
        [1.0e-10, 5.0e-11],
        [1.0e-24, 5.0e-25],
        radius,
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {
                "radial_memory_slope_gate_strength": 1.0,
                "radial_memory_slope_gate_mode": 5.0,
            },
        )
    inactive = apply_radial_memory_to_response(
        local,
        radius,
        {
            "radial_memory_slope_gate_strength": 0.0,
            "radial_memory_slope_gate_mode": 99.0,
        },
    )
    np.testing.assert_array_equal(inactive["enhancement"], local["enhancement"])


def test_all_slope_gate_modes_handle_one_point_profiles():
    radius = np.array([10.0])
    for mode in range(5):
        response = screened_hybrid_profile_response(
            [1.0e-11],
            [1.0e-25],
            radius,
            PARAMETERS,
            {
                "radial_memory_strength": 1.0,
                "radial_memory_slope_gate_strength": 1.0,
                "radial_memory_slope_gate_mode": mode,
                "radial_memory_steep_gbar_power": -0.5,
                "radial_memory_steep_radius_power": 1.5,
            },
        )
        assert np.isfinite(response["enhancement"][0])
        assert response["radial_memory_slope_gate_coordinate"][0] == 0.0


def test_smoothed_local_response_matches_point_response_on_exact_power_law():
    radius = np.geomspace(1.0, 100.0, 30)
    gbar = 1.0e-10 / np.power(radius, 1.6)
    density = 1.0e-24 / np.square(radius)
    settings = {
        "radial_memory_strength": 1.0,
        "radial_memory_log_scale": 0.8,
        "radial_memory_gbar_power": -1.0,
        "radial_memory_radius_power": -0.5,
        "radial_memory_slope_gate_strength": 1.0,
        "radial_memory_slope_gate_pivot": 0.0,
        "radial_memory_slope_gate_sharpness": 4.0,
        "radial_memory_steep_gbar_power": -0.5,
        "radial_memory_steep_radius_power": 1.5,
    }
    point = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {**settings, "radial_memory_slope_gate_mode": 3},
    )
    smoothed = screened_hybrid_profile_response(
        gbar,
        density,
        radius,
        PARAMETERS,
        {
            **settings,
            "radial_memory_slope_gate_mode": 4,
            "radial_memory_slope_smoothing_log_scale": 0.5,
        },
    )
    np.testing.assert_allclose(
        smoothed["radial_memory_slope_gate_coordinate"], -1.6, atol=1.0e-12
    )
    np.testing.assert_allclose(
        smoothed["fractional_excess"], point["fractional_excess"], rtol=1.0e-12
    )
    assert np.all(
        smoothed["radial_memory_slope_smoothing_log_scale"] == 0.5
    )


def test_smoothed_local_slope_reduces_point_derivative_roughness():
    radius = np.geomspace(1.0, 100.0, 80)
    gbar = (
        1.0e-10
        / np.power(radius, 1.4)
        * np.exp(0.12 * np.sin(18.0 * np.log(radius)))
    )
    response = screened_hybrid_profile_response(
        gbar,
        1.0e-24 / np.square(radius),
        radius,
        PARAMETERS,
        {
            "radial_memory_strength": 1.0,
            "radial_memory_slope_gate_strength": 1.0,
            "radial_memory_slope_gate_mode": 4,
            "radial_memory_slope_smoothing_log_scale": 0.5,
            "radial_memory_steep_gbar_power": -0.5,
            "radial_memory_steep_radius_power": 1.5,
        },
    )
    local = response["radial_memory_local_log_gbar_slope"]
    smoothed = response["radial_memory_slope_gate_coordinate"]
    assert np.std(np.diff(smoothed)) < 0.1 * np.std(np.diff(local))
    assert abs(float(np.median(smoothed)) + 1.4) < 0.05


def test_smoothed_local_slope_scale_validation_is_mode_specific():
    radius = np.array([1.0, 2.0, 4.0])
    local = screened_hybrid_response(
        [1.0e-10, 5.0e-11, 2.5e-11],
        [1.0e-24, 5.0e-25, 2.5e-25],
        radius,
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_radial_memory_to_response(
            local,
            radius,
            {
                "radial_memory_slope_gate_strength": 1.0,
                "radial_memory_slope_gate_mode": 4,
                "radial_memory_slope_smoothing_log_scale": 0.0,
            },
        )
    inactive = apply_radial_memory_to_response(
        local,
        radius,
        {
            "radial_memory_slope_gate_strength": 0.0,
            "radial_memory_slope_gate_mode": 4,
            "radial_memory_slope_smoothing_log_scale": 0.0,
        },
    )
    np.testing.assert_array_equal(inactive["enhancement"], local["enhancement"])


def test_no_flux_diffusion_preserves_constant_and_carrier_integral():
    radius = np.array([1.0, 1.7, 4.0, 11.0, 30.0])
    constant = np.full(len(radius), 3.5)
    unchanged, widths = no_flux_log_radius_diffusion(
        radius, constant, log_scale=0.7
    )
    np.testing.assert_allclose(unchanged, constant, rtol=2.0e-14, atol=2.0e-14)

    source = np.array([0.0, 1.0, 4.0, 0.5, 0.0])
    diffused, widths = no_flux_log_radius_diffusion(
        radius, source, log_scale=0.7
    )
    assert np.all(diffused >= 0.0)
    assert np.count_nonzero(diffused > 1.0e-12) > np.count_nonzero(source)
    np.testing.assert_allclose(
        np.dot(widths, diffused),
        np.dot(widths, source),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_profile_diffusion_zero_strength_is_exact_and_one_point_is_identity():
    radius = np.geomspace(1.0, 30.0, 12)
    local = screened_hybrid_profile_response(
        1.0e-10 / np.square(radius),
        1.0e-24 / np.square(radius),
        radius,
        PARAMETERS,
        {},
    )
    explicit_zero = screened_hybrid_profile_response(
        1.0e-10 / np.square(radius),
        1.0e-24 / np.square(radius),
        radius,
        PARAMETERS,
        {
            "radial_diffusion_strength": 0.0,
            "radial_diffusion_log_scale": 0.7,
            "radial_diffusion_gbar_power": 1.0,
            "radial_diffusion_radius_power": 1.0,
        },
    )
    np.testing.assert_array_equal(
        explicit_zero["enhancement"], local["enhancement"]
    )

    point = screened_hybrid_profile_response(
        [1.0e-11],
        [1.0e-25],
        [10.0],
        PARAMETERS,
        {"radial_diffusion_strength": 1.0},
    )
    np.testing.assert_array_equal(
        point["fractional_excess"], point["pre_diffusion_fractional_excess"]
    )


def test_profile_diffusion_carrier_choice_changes_redistribution_not_integral():
    radius = np.geomspace(1.0, 100.0, 24)
    gbar = 1.0e-10 / np.power(radius, 1.7)
    density = 1.0e-24 / np.square(radius)
    responses = []
    for gbar_power, radius_power in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)):
        response = screened_hybrid_profile_response(
            gbar,
            density,
            radius,
            PARAMETERS,
            {
                "radial_diffusion_strength": 1.0,
                "radial_diffusion_log_scale": 0.7,
                "radial_diffusion_gbar_power": gbar_power,
                "radial_diffusion_radius_power": radius_power,
            },
        )
        widths = response["radial_diffusion_log_radius_cell_width"]
        np.testing.assert_allclose(
            np.dot(widths, response["radial_diffusion_blended_source"]),
            np.dot(widths, response["radial_diffusion_local_source"]),
            rtol=2.0e-13,
        )
        responses.append(response["fractional_excess"])
    assert not np.allclose(responses[0], responses[1])
    assert not np.allclose(responses[1], responses[2])


def test_profile_diffusion_validation_is_strict():
    local = screened_hybrid_response(
        [1.0e-10, 5.0e-11],
        [1.0e-24, 5.0e-25],
        [1.0, 2.0],
        PARAMETERS,
        {},
    )
    with np.testing.assert_raises(ValueError):
        apply_radial_diffusion_to_response(
            local,
            [1.0, 2.0],
            {"radial_diffusion_strength": 1.1},
        )
    with np.testing.assert_raises(ValueError):
        apply_radial_diffusion_to_response(
            local,
            [1.0, 2.0],
            {
                "radial_diffusion_strength": 1.0,
                "radial_diffusion_log_scale": 0.0,
            },
        )
    with np.testing.assert_raises(ValueError):
        no_flux_log_radius_diffusion(
            [1.0, 1.0], [1.0, 2.0], log_scale=0.5
        )


def test_screen_power_controls_solar_suppression():
    weak = solar_system_diagnostics(
        PARAMETERS,
        {"combination": "interaction", "screen_power": 0.7},
        cassini_fractional_limit=2.3e-5,
    )
    strong = solar_system_diagnostics(
        PARAMETERS,
        {"combination": "interaction", "screen_power": 1.5},
        cassini_fractional_limit=2.3e-5,
    )
    assert (
        strong["maximum_fractional_change_limb_to_Saturn"]
        < weak["maximum_fractional_change_limb_to_Saturn"]
    )


def test_mercury_precession_is_finite():
    value = mercury_precession_mas_per_century(
        PARAMETERS,
        {"combination": "interaction", "screen_power": 1.5},
        quadrature_points=4096,
    )
    assert np.isfinite(value)


def test_default_channel_gate_topology_remains_monotonic():
    coordinates = np.array([0.2, 0.5, 0.8])
    common = {
        "channel_gate_property": "tidal_middle_to_max",
        "channel_gate_pivot": 0.5,
        "channel_gate_sharpness": 5.0,
    }
    implicit = screened_hybrid_response(
        np.full(3, 1.0e-10),
        np.full(3, 1.0e-24),
        np.arange(1.0, 4.0),
        PARAMETERS,
        common,
        channel_gate_property_values=coordinates,
    )
    explicit = screened_hybrid_response(
        np.full(3, 1.0e-10),
        np.full(3, 1.0e-24),
        np.arange(1.0, 4.0),
        PARAMETERS,
        {**common, "channel_gate_topology": "monotonic"},
        channel_gate_property_values=coordinates,
    )
    np.testing.assert_array_equal(
        implicit["channel_gate_cluster_weight"],
        explicit["channel_gate_cluster_weight"],
    )


def test_band_and_tails_channel_gates_are_exact_complements():
    coordinates = np.array([0.1, 0.45, 0.65, 0.8, 1.0])
    common = {
        "channel_gate_property": "tidal_middle_to_max",
        "channel_gate_lower_pivot": 0.45,
        "channel_gate_upper_pivot": 0.8,
        "channel_gate_sharpness": 10.0,
    }
    responses = {}
    for topology in ("band", "tails"):
        responses[topology] = screened_hybrid_response(
            np.full(len(coordinates), 1.0e-10),
            np.full(len(coordinates), 1.0e-24),
            np.arange(1.0, len(coordinates) + 1.0),
            PARAMETERS,
            {**common, "channel_gate_topology": topology},
            channel_gate_property_values=coordinates,
        )["channel_gate_cluster_weight"]
    np.testing.assert_allclose(
        responses["band"] + responses["tails"], 1.0, atol=1.0e-15
    )
    assert responses["band"][2] > responses["band"][0]
    assert responses["band"][2] > responses["band"][-1]
    assert responses["tails"][2] < responses["tails"][0]
    assert responses["tails"][2] < responses["tails"][-1]


def test_constant_channel_gate_controls_are_exact():
    coordinates = np.array([0.2, 0.5, 0.8])
    for weight in (0.0, 0.4, 1.0):
        response = screened_hybrid_response(
            np.full(3, 1.0e-10),
            np.full(3, 1.0e-24),
            np.arange(1.0, 4.0),
            PARAMETERS,
            {
                "channel_gate_property": "tidal_middle_to_max",
                "channel_gate_topology": "constant",
                "channel_gate_constant_weight": weight,
            },
            channel_gate_property_values=coordinates,
        )
        np.testing.assert_array_equal(
            response["channel_gate_cluster_weight"], weight
        )


def test_nonmonotonic_channel_gate_validates_topology_and_pivots():
    arguments = (
        [1.0e-10],
        [1.0e-24],
        [1.0],
        PARAMETERS,
    )
    with np.testing.assert_raises(ValueError):
        screened_hybrid_response(
            *arguments,
            {
                "channel_gate_property": "tidal_middle_to_max",
                "channel_gate_topology": "band",
                "channel_gate_lower_pivot": 0.8,
                "channel_gate_upper_pivot": 0.45,
            },
            channel_gate_property_values=[0.6],
        )
    with np.testing.assert_raises(ValueError):
        screened_hybrid_response(
            *arguments,
            {
                "channel_gate_property": "tidal_middle_to_max",
                "channel_gate_topology": "constant",
                "channel_gate_constant_weight": 1.1,
            },
            channel_gate_property_values=[0.6],
        )
    with np.testing.assert_raises(ValueError):
        screened_hybrid_response(
            *arguments,
            {
                "channel_gate_property": "tidal_middle_to_max",
                "channel_gate_topology": "unknown",
            },
            channel_gate_property_values=[0.6],
        )
