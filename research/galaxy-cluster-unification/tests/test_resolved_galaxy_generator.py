from __future__ import annotations

import json

import numpy as np

from voidscreen.resolved_galaxy_generator import (
    extract_galaxy_parameters,
    lift_surface_density_to_volume,
    package_content_hash,
    render_galaxy,
    roundtrip_metrics,
    sample_vertical_realization,
)


def synthetic_components() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-6.0, 6.0, 65)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx - 0.25, yy + 0.15)
    phi = np.arctan2(yy + 0.15, xx - 0.25)
    gas = 4.0e6 * np.exp(-radius / 2.3) * (1.0 + 0.18 * np.cos(phi))
    gas += 1.3e6 * np.exp(-0.5 * ((xx + 1.7) ** 2 + (yy - 0.8) ** 2) / 0.45**2)
    stars = 8.0e6 * np.exp(-radius / 1.4) * (1.0 + 0.12 * np.cos(2.0 * phi))
    stars += 2.0e6 * np.exp(-0.5 * ((xx - 0.8) ** 2 + (yy + 1.2) ** 2) / 0.3**2)
    return axis, gas, stars


def test_extraction_is_deterministic_json_and_contains_no_gravity_fit() -> None:
    axis, gas, stars = synthetic_components()
    first = extract_galaxy_parameters(
        "SYNTHETIC", axis, gas, stars, residual_feature_count=12
    )
    second = extract_galaxy_parameters(
        "SYNTHETIC", axis, gas, stars, residual_feature_count=12
    )
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
    assert first["contentSha256"] == package_content_hash(first)
    assert first["gravityParameters"] == {}
    assert first["velocityTargetsUsed"] is False
    assert first["verticalStructure"]["status"] == "assumed_prior_not_measured"


def test_known_map_roundtrip_preserves_mass_and_structure() -> None:
    axis, gas, stars = synthetic_components()
    package = extract_galaxy_parameters(
        "SYNTHETIC", axis, gas, stars, residual_feature_count=24
    )
    generated = render_galaxy(package, axis)
    gas_metrics = roundtrip_metrics(gas, generated["gas"], axis)
    stellar_metrics = roundtrip_metrics(stars, generated["stars"], axis)
    assert gas_metrics["mass_relative_error"] < 1e-12
    assert stellar_metrics["mass_relative_error"] < 1e-12
    assert gas_metrics["normalized_l2"] < 0.16
    assert stellar_metrics["normalized_l2"] < 0.18
    assert gas_metrics["pixel_correlation"] > 0.98
    assert stellar_metrics["pixel_correlation"] > 0.98


def test_generation_controls_change_shape_without_hidden_per_object_fit() -> None:
    axis, gas, stars = synthetic_components()
    package = extract_galaxy_parameters(
        "SYNTHETIC", axis, gas, stars, residual_feature_count=8
    )
    ordinary = render_galaxy(package, axis)
    changed = render_galaxy(
        package,
        axis,
        component_controls={
            "gas": {
                "mass_scale": 1.5,
                "radial_scale": 0.8,
                "fourier_scale": 0.2,
                "residual_scale": 0.0,
                "rotation_deg": 35.0,
            }
        },
    )
    spacing = axis[1] - axis[0]
    assert np.isclose(
        np.sum(changed["gas"]) * spacing**2,
        1.5 * np.sum(ordinary["gas"]) * spacing**2,
    )
    assert not np.allclose(changed["gas"], ordinary["gas"])
    assert np.allclose(changed["stars"], ordinary["stars"])


def test_distinct_3d_priors_project_to_the_identical_2d_map() -> None:
    axis, gas, _ = synthetic_components()
    z_axis = np.linspace(-3.0, 3.0, 41)
    thin = lift_surface_density_to_volume(
        gas, axis, z_axis, scale_height_kpc=0.15, profile="exponential"
    )
    thick = lift_surface_density_to_volume(
        gas, axis, z_axis, scale_height_kpc=0.75, profile="sech_squared"
    )
    dz = z_axis[1] - z_axis[0]
    assert not np.allclose(thin, thick)
    assert np.allclose(np.sum(thin, axis=2) * dz, gas, rtol=1e-12, atol=1e-8)
    assert np.allclose(np.sum(thick, axis=2) * dz, gas, rtol=1e-12, atol=1e-8)


def test_seeded_vertical_realizations_are_replayable_and_declared_as_priors() -> None:
    axis, gas, _ = synthetic_components()
    z_axis = np.linspace(-3.0, 3.0, 33)
    first, first_metadata = sample_vertical_realization(
        gas,
        axis,
        z_axis,
        r80_kpc=3.5,
        component="gas",
        rng=np.random.default_rng(20260802),
    )
    second, second_metadata = sample_vertical_realization(
        gas,
        axis,
        z_axis,
        r80_kpc=3.5,
        component="gas",
        rng=np.random.default_rng(20260802),
    )
    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert first_metadata["status"] == "assumed_prior_not_measured"
