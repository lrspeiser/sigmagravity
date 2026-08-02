from __future__ import annotations

import numpy as np

from voidscreen.sky_lensing import (
    GridSkyDeflectionField,
    LinearCombinationSkyDeflectionField,
    assign_observed_roots,
    critical_curve_points,
    find_lens_roots,
    lens_invariants,
    photon_deflection_sky,
    profiled_source,
)


class SingularIsothermalSphere:
    half_extent_arcsec = 3.0

    def __init__(self, einstein_radius_arcsec: float = 1.0):
        self.einstein_radius_arcsec = einstein_radius_arcsec

    def alpha(self, east_arcsec, north_arcsec, source_redshift):
        del source_redshift
        east = np.asarray(east_arcsec, dtype=float)
        north = np.asarray(north_arcsec, dtype=float)
        radius = np.hypot(east, north)
        return (
            self.einstein_radius_arcsec
            * np.divide(east, radius, out=np.full_like(east, np.nan), where=radius > 0.0),
            self.einstein_radius_arcsec
            * np.divide(
                north, radius, out=np.full_like(north, np.nan), where=radius > 0.0
            ),
        )


class AffineLens:
    half_extent_arcsec = 10.0

    def alpha(self, east_arcsec, north_arcsec, source_redshift):
        del source_redshift
        east = np.asarray(east_arcsec, dtype=float)
        north = np.asarray(north_arcsec, dtype=float)
        return 0.30 * east + 0.10 * north, 0.10 * east + 0.20 * north


def test_grid_field_enforces_north_rows_east_columns_without_axis_swap():
    north = np.linspace(-3.0, 3.0, 13)
    east = np.linspace(-2.0, 2.0, 11)
    east_grid, north_grid = np.meshgrid(east, north, indexing="xy")
    field = GridSkyDeflectionField(
        north_axis_arcsec=north,
        east_axis_arcsec=east,
        alpha_east_ratio_one_arcsec=2.0 * east_grid + 3.0 * north_grid,
        alpha_north_ratio_one_arcsec=5.0 * east_grid - 7.0 * north_grid,
        distance_ratio=lambda _redshift: 0.5,
    )
    alpha_east, alpha_north = field.alpha(
        np.asarray([0.4, -1.2]), np.asarray([-1.1, 2.3]), 4.0
    )
    assert np.allclose(alpha_east, 0.5 * np.asarray([-2.5, 4.5]))
    assert np.allclose(alpha_north, 0.5 * np.asarray([9.7, -22.1]))


def test_affine_lens_invariants_match_exact_convergence_and_shear():
    invariants = lens_invariants(AffineLens(), 1.0, -2.0, 2.0, step_arcsec=1.0e-4)
    assert np.allclose(invariants.convergence, 0.25, atol=1.0e-10)
    assert np.allclose(invariants.shear_1, 0.05, atol=1.0e-10)
    assert np.allclose(invariants.shear_2, 0.10, atol=1.0e-10)
    assert np.allclose(invariants.rotation, 0.0, atol=1.0e-10)
    assert np.allclose(invariants.determinant, 0.55, atol=1.0e-10)


def test_linear_combination_field_combines_vectors_and_extent():
    first = AffineLens()
    second = AffineLens()
    second.half_extent_arcsec = 7.0
    field = LinearCombinationSkyDeflectionField((first, second), (-2.0, 3.5))
    expected_east, expected_north = first.alpha(2.0, -1.0, 4.0)
    alpha_east, alpha_north = field.alpha(2.0, -1.0, 4.0)
    assert field.half_extent_arcsec == 7.0
    assert np.allclose(alpha_east, 1.5 * expected_east)
    assert np.allclose(alpha_north, 1.5 * expected_north)


def test_sis_global_roots_are_stable_with_grid_density():
    field = SingularIsothermalSphere()
    expected = np.asarray([[-0.8, 0.0], [1.2, 0.0]])
    for grid_points in (81, 161, 241):
        result = find_lens_roots(
            field,
            np.asarray([0.2, 0.0]),
            2.0,
            bound_arcsec=2.5,
            grid_points=grid_points,
            deduplication_tolerance_arcsec=0.05,
        )
        assert len(result.roots_arcsec) == 2
        assert np.allclose(result.roots_arcsec, expected, atol=2.0e-5)
        assert np.max(result.closure_arcsec) < 1.0e-8


def test_sis_critical_curve_recovers_einstein_radius():
    points = critical_curve_points(
        SingularIsothermalSphere(), 2.0, bound_arcsec=2.0, grid_points=301
    )
    radii = np.hypot(points[:, 0], points[:, 1])
    ring = radii[radii > 0.25]
    assert len(ring) > 100
    assert abs(float(np.median(ring)) - 1.0) < 0.02


def test_source_profile_and_assignment_recover_sis_images():
    field = SingularIsothermalSphere()
    observed = np.asarray([[-0.8, 0.0], [1.2, 0.0]])
    source = profiled_source(field, observed, 2.0)
    roots = find_lens_roots(
        field,
        source,
        2.0,
        bound_arcsec=2.5,
        observed_starts_arcsec=observed,
    )
    assignment = assign_observed_roots(observed, roots.roots_arcsec)
    assert np.allclose(source, [0.2, 0.0])
    assert assignment.complete
    assert assignment.matched_images == 2
    assert assignment.rms_arcsec < 1.0e-8


def test_sky_photon_wrapper_maps_row_and_column_components_explicitly():
    shape = (7, 9, 11)
    acceleration_north = np.full(shape, 2.0)
    acceleration_east = np.full(shape, 3.0)
    acceleration_los = np.zeros(shape)
    result = photon_deflection_sky(
        (acceleration_north, acceleration_east, acceleration_los),
        0.5,
        light_speed=1.0,
    )
    expected_east = -2.0 * 3.0 * 0.5 * (shape[2] - 1)
    expected_north = -2.0 * 2.0 * 0.5 * (shape[2] - 1)
    assert np.allclose(result.alpha_east_radian, expected_east)
    assert np.allclose(result.alpha_north_radian, expected_north)
