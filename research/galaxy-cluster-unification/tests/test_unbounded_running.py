import numpy as np

from voidscreen.unbounded_running import (
    RUNNING_MODELS,
    VARIABLE_EXPONENT_DENSITY_MODELS,
    equivalent_enclosed_baryonic_mass_msun,
    mean_equivalent_baryonic_density_g_cm3,
    point_mass_scale_diagnostics,
    predict_running_acceleration,
    running_enhancement,
    solar_system_diagnostics,
)

PARAMETERS = {
    "curvature_log": [-30.0, 1.0, 1.0],
    "curvature_loglog": [-30.0, 1.0, 1.0],
    "curvature_rootlog": [-30.0, 1.0, 0.5],
    "curvature_power": [-30.0, 1.0, 0.5],
    "curvature_stretched_power": [-30.0, 1.0, 0.5, 0.8],
    "curvature_mixed_power": [-30.0, 1.0, 0.2, 0.5],
    "curvature_additive_power": [-30.0, 1.0, 0.2, 0.5],
    "curvature_decelerating_power": [-30.0, 1.0, 0.5, 0.5],
    "curvature_variable_mass_power": [-30.0, 1.0, 0.5, 10.0, 0.5],
    "curvature_variable_density_power": [-30.0, 1.0, 0.5, -24.0, 0.5],
    "curvature_variable_shape_power": [-30.0, 1.0, 0.5, -1.0, 0.5],
    "path_log_running": [-30.0, 1.0, 0.0, 1.0, 1.0],
    "path_power_running": [-30.0, 1.0, 0.0, 1.0, 0.5],
    "tensor_alignment_log": [-30.0, 1.0, 1.0, 1.0],
    "tensor_dominance_log": [-30.0, 1.0, 1.0, 1.0],
    "tensor_alignment_power": [-30.0, 1.0, 0.5, 1.0],
    "tensor_dominance_power": [-30.0, 1.0, 0.5, 1.0],
}


def test_all_predeclared_running_models_are_covered():
    assert set(PARAMETERS) == RUNNING_MODELS


def test_running_laws_are_newtonian_at_high_curvature_and_grow_without_ceiling():
    for model, parameters in PARAMETERS.items():
        eigenvalues = np.asarray([[-2.0e-15, 1.0e-15, 1.0e-15], [-2.0e-40, 1.0e-40, 1.0e-40]])
        result = running_enhancement(
            [1.0e-3, 1.0e-15],
            [1.0e-9, 1.0e6],
            model,
            parameters,
            tidal_eigenvalues_s2=eigenvalues if model.startswith("tensor_") else None,
            local_density_g_cm3=(
                [1.0e-20, 1.0e-30]
                if model in VARIABLE_EXPONENT_DENSITY_MODELS
                else None
            ),
        )
        assert result["enhancement_relative_to_local_G"][0] >= 1.0
        assert result["enhancement_relative_to_local_G"][1] > result[
            "enhancement_relative_to_local_G"
        ][0]


def test_predicted_acceleration_is_never_below_baryonic_newtonian_value():
    result = predict_running_acceleration(
        [1.0e-10, 1.0e-12],
        [1.0, 100.0],
        "curvature_log",
        PARAMETERS["curvature_log"],
    )
    assert np.all(result["predicted_acceleration_m_s2"] >= [1.0e-10, 1.0e-12])


def test_solar_and_large_scale_diagnostics_are_finite():
    solar = solar_system_diagnostics(
        "curvature_log", PARAMETERS["curvature_log"], cassini_limit=2.3e-5
    )
    scale = point_mass_scale_diagnostics(
        "curvature_log", PARAMETERS["curvature_log"]
    )
    assert solar["PPN_gamma_minus_one"] == 0.0
    assert np.isfinite(solar["maximum_fractional_change_limb_to_Saturn"])
    assert scale["enhancement_by_radius_kpc"]["1e+06"] > scale[
        "enhancement_by_radius_kpc"
    ]["1"]


def test_solar_diagnostic_preserves_changes_below_float_spacing_at_one():
    solar = solar_system_diagnostics(
        "curvature_power",
        [-28.30851641358452, 2.0, 0.12124817480126633],
        cassini_limit=2.3e-5,
    )
    assert 0.0 < solar["Earth_orbit_fractional_change"] < 1.0e-25
    assert 0.0 < solar["Saturn_orbit_fractional_change"] < 1.0e-20
    assert solar["Cassini_pass"]


def test_variable_exponent_reduces_to_constant_power_when_beta_is_zero():
    gbar = np.asarray([1.0e-10, 1.0e-12])
    radius = np.asarray([1.0, 100.0])
    constant = running_enhancement(
        gbar, radius, "curvature_power", [-30.0, 2.0, 0.2]
    )
    variable = running_enhancement(
        gbar,
        radius,
        "curvature_variable_density_power",
        [-30.0, 2.0, 0.0, -24.0, 0.2],
        local_density_g_cm3=[1.0e-20, 1.0e-30],
    )
    assert np.allclose(
        variable["enhancement_relative_to_local_G"],
        constant["enhancement_relative_to_local_G"],
    )
    assert np.all(variable["effective_exponent"] == 2.0)


def test_mass_and_shape_proxies_have_expected_spherical_mean_relation():
    gbar = np.asarray([1.0e-10])
    radius = np.asarray([10.0])
    mass = equivalent_enclosed_baryonic_mass_msun(gbar, radius)
    density = mean_equivalent_baryonic_density_g_cm3(gbar, radius)
    assert mass[0] > 0.0
    assert density[0] > 0.0
