from __future__ import annotations

import numpy as np
import pytest

from voidscreen.observation_adapters import evaluate_observation_targets
from voidscreen.sky_lensing import C_M_S, RAD_TO_ARCSEC


def _model() -> dict:
    return {
        "observables": [
            {
                "id": "photon_acceleration",
                "target": "photons",
                "rank": "vector",
                "unit": "m/s^2",
            }
        ]
    }


def _sis_fixture(
    *,
    einstein_radius_arcsec: float = 1.0,
    reference_ratio: float = 0.7,
    cells: tuple[int, int, int] = (101, 101, 9),
    angular_spacing_arcsec: float = 0.05,
) -> tuple[dict, dict[str, np.ndarray]]:
    lens_distance = 1.0e20
    physical_spacing = lens_distance * angular_spacing_arcsec / RAD_TO_ARCSEC
    spacing = [physical_spacing] * 3
    origin = [-0.5 * (count - 1) * physical_spacing for count in cells]
    north_axis = np.arange(cells[0]) * angular_spacing_arcsec - 0.5 * (
        cells[0] - 1
    ) * angular_spacing_arcsec
    east_axis = np.arange(cells[1]) * angular_spacing_arcsec - 0.5 * (
        cells[1] - 1
    ) * angular_spacing_arcsec
    east, north = np.meshgrid(east_axis, north_axis, indexing="xy")
    radius = np.hypot(east, north)
    alpha_east_arcsec = np.divide(
        einstein_radius_arcsec * east,
        radius,
        out=np.zeros_like(east),
        where=radius > 0,
    )
    alpha_north_arcsec = np.divide(
        einstein_radius_arcsec * north,
        radius,
        out=np.zeros_like(north),
        where=radius > 0,
    )
    path_length = (cells[2] - 1) * physical_spacing
    acceleration_scale = -(C_M_S**2) / (2.0 * reference_ratio * path_length)
    alpha_east_rad = alpha_east_arcsec / RAD_TO_ARCSEC
    alpha_north_rad = alpha_north_arcsec / RAD_TO_ARCSEC
    observables = {
        "photon_acceleration__axis0": np.repeat(
            (acceleration_scale * alpha_north_rad)[..., None], cells[2], axis=2
        ),
        "photon_acceleration__axis1": np.repeat(
            (acceleration_scale * alpha_east_rad)[..., None], cells[2], axis=2
        ),
        "photon_acceleration__axis2": np.zeros(cells),
    }
    geometry = {
        "coordinateSystem": "cartesian_3d",
        "dimensions": 3,
        "spacing": spacing,
        "origin": origin,
    }
    return geometry, observables


def _target(*, families: list[dict] | None = None, **changes) -> dict:
    return {
        "schemaVersion": "sigma-observation-target/1",
        "id": "raw-images",
        "kind": "multiple_image_systems",
        "observable": "photon_acceleration",
        "northAxis": 0,
        "eastAxis": 1,
        "lineOfSightAxis": 2,
        "lensAngularDiameterDistanceM": 1.0e20,
        "skyCenterM": [0.0, 0.0, 0.0],
        "rootSearchBoundArcsec": 2.4,
        "rootGridPoints": 81,
        "supplementalGridPoints": [81],
        "closureToleranceArcsec": 1.0e-4,
        "deduplicationToleranceArcsec": 0.05,
        "jacobianStepArcsec": 0.02,
        "includeResidualMinima": True,
        "families": families
        or [
            {
                "id": "source-a",
                "distanceRatio": 0.7,
                "observedImagesArcsec": [[-0.8, 0.0], [1.2, 0.0]],
                "positionUncertaintiesArcsec": [0.05, 0.05],
            }
        ],
        "provenance": {"kind": "analytic SIS fixture"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        **changes,
    }


def _axisymmetric_sis_fixture() -> tuple[dict, dict[str, np.ndarray], dict]:
    lens_distance = 1.0e20
    reference_ratio = 0.7
    angular_spacing_arcsec = 0.05
    physical_spacing = (
        lens_distance * angular_spacing_arcsec / RAD_TO_ARCSEC
    )
    shape = (51, 65)
    path_length = (shape[1] - 1) * physical_spacing
    acceleration_scale = (
        (1.0 / RAD_TO_ARCSEC)
        * C_M_S**2
        / (2.0 * reference_ratio * path_length)
    )
    radial = np.full(shape, -acceleration_scale)
    radial[0, :] = 0.0
    origin = [0.0, -0.5 * (shape[1] - 1) * physical_spacing]
    geometry = {
        "coordinateSystem": "axisymmetric_cylindrical",
        "dimensions": 2,
        "spacing": [physical_spacing, physical_spacing],
        "origin": origin,
        "axisOrder": ["r", "z"],
    }
    observables = {
        "photon_acceleration__axis0": radial,
        "photon_acceleration__axis1": np.zeros(shape),
    }
    target = _target(
        rootSearchBoundArcsec=1.4,
        skyCenterM=[0.0, 0.0, 0.0],
        axisymmetricInclinationDeg=0.0,
        skyShape=[101, 101],
        lineOfSightSamples=65,
        gridOriginM=origin,
    )
    for key in ("northAxis", "eastAxis", "lineOfSightAxis"):
        target.pop(key)
    return geometry, observables, target


def test_sis_raw_images_profile_source_find_roots_and_score() -> None:
    geometry, observables = _sis_fixture()
    roots: dict[str, np.ndarray] = {}
    auxiliary: dict[str, list[dict]] = {}
    evaluation, rows = evaluate_observation_targets(
        _model(),
        observables,
        geometry,
        [_target()],
        root_outputs=roots,
        auxiliary_rows=auxiliary,
    )
    result = evaluation["targets"][0]
    assert result["state"] == "scored"
    assert result["fittedObservationNuisanceParameters"] == 2
    assert result["gravityParametersAdded"] == 0
    np.testing.assert_allclose(result["families"][0]["profiledSourceArcsec"], [0.2, 0], atol=1e-12)
    assert result["families"][0]["matchedImages"] == 2
    # A rasterized singular center can create one explicitly disclosed,
    # extremely demagnified interpolation root; it is not silently deleted.
    assert result["families"][0]["predictedRoots"] >= 2
    assert result["score"]["channels"]["image_position_arcsec"]["imagePlaneRmsArcsec"] < 1e-3
    assert evaluation["channelAggregates"]["image_position_arcsec"]["rmse"] < 1e-3
    assert len(rows) == 2
    assert all(row["assignment_state"] == "matched" for row in rows)
    assert len(auxiliary["multiple_image_families"]) == 1
    assert sorted(roots) == [
        "target_000__family_000__absolute_magnifications",
        "target_000__family_000__closures_arcsec",
        "target_000__family_000__roots_arcsec",
    ]


def test_axisymmetric_sis_raw_images_use_one_archived_ratio_one_projection() -> None:
    geometry, observables, target = _axisymmetric_sis_fixture()
    roots: dict[str, np.ndarray] = {}
    maps: dict[str, np.ndarray] = {}
    evaluation, rows = evaluate_observation_targets(
        _model(),
        observables,
        geometry,
        [target],
        map_outputs=maps,
        root_outputs=roots,
    )
    result = evaluation["targets"][0]
    family = result["families"][0]
    projection = result["ratioOneProjection"]
    assert result["state"] == "scored"
    assert result["coordinateSystem"] == "axisymmetric_cylindrical"
    assert result["samplingMode"] == "axisymmetric_cylindrical_ray_integral"
    assert result["gravityParametersAdded"] == 0
    np.testing.assert_allclose(family["profiledSourceArcsec"], [0.2, 0.0], atol=1e-12)
    assert family["matchedImages"] == 2
    assert family["score"]["imagePlaneRmsArcsec"] < 1.0e-3
    assert all(row["assignment_state"] == "matched" for row in rows)
    assert projection["finiteRootSupport"]["verifiedFiniteNodes"] > 3_000
    assert projection["finiteRootSupport"]["outsideSupportInterpolationFill"] == (
        "zero_only_outside_verified_root_support"
    )
    alpha_key = "target_000__ratio_one_projection__alpha_east_arcsec"
    assert alpha_key in maps
    assert np.any(~np.isfinite(maps[alpha_key]))
    assert len(roots) == 3


def test_axisymmetric_raw_images_reject_search_crossing_unsupported_rays() -> None:
    geometry, observables, target = _axisymmetric_sis_fixture()
    target["rootSearchBoundArcsec"] = 2.0
    with pytest.raises(ValueError, match="finite photon support"):
        evaluate_observation_targets(_model(), observables, geometry, [target])


def test_axisymmetric_raw_images_reject_ambiguous_axes_and_origin() -> None:
    geometry, observables, target = _axisymmetric_sis_fixture()
    target["northAxis"] = 0
    with pytest.raises(ValueError, match="does not accept Cartesian"):
        evaluate_observation_targets(_model(), observables, geometry, [target])
    target.pop("northAxis")
    target["gridOriginM"] = [0.0, geometry["origin"][1] + 1.0]
    with pytest.raises(ValueError, match="origin must match"):
        evaluate_observation_targets(_model(), observables, geometry, [target])


def test_missing_multiplicity_has_no_finite_aggregate_score() -> None:
    geometry, observables = _sis_fixture(einstein_radius_arcsec=0.1)
    evaluation, rows = evaluate_observation_targets(
        _model(), observables, geometry, [_target()]
    )
    result = evaluation["targets"][0]
    channel = result["score"]["channels"]["image_position_arcsec"]
    assert result["state"] == "incomplete_topology"
    assert result["incompleteFamilyCount"] == 1
    assert channel["rmse"] is None
    assert channel["chiSquare"] is None
    assert "image_position_arcsec" not in evaluation["channelAggregates"]
    assert evaluation["scoredTargetCount"] == 0
    assert any(row["assignment_state"] == "unmatched" for row in rows)


def test_distance_ratio_changes_the_einstein_radius_without_new_parameters() -> None:
    geometry, observables = _sis_fixture()
    families = [
        {
            "id": "ratio-one",
            "distanceRatio": 0.7,
            "observedImagesArcsec": [[-0.8, 0.0], [1.2, 0.0]],
            "positionUncertaintiesArcsec": [0.05, 0.05],
        },
        {
            "id": "ratio-half",
            "distanceRatio": 0.35,
            "observedImagesArcsec": [[-0.4, 0.0], [0.6, 0.0]],
            "positionUncertaintiesArcsec": [0.05, 0.05],
        },
    ]
    evaluation, _rows = evaluate_observation_targets(
        _model(), observables, geometry, [_target(families=families)]
    )
    result = evaluation["targets"][0]
    assert result["state"] == "scored"
    assert result["fittedObservationNuisanceParameters"] == 4
    assert result["gravityParametersAdded"] == 0
    assert result["score"]["channels"]["image_position_arcsec"]["imagePlaneRmsArcsec"] < 1e-3


def test_storage_axis_permutation_preserves_named_sky_roots() -> None:
    geometry, observables = _sis_fixture(cells=(81, 91, 11))
    canonical, _rows = evaluate_observation_targets(
        _model(), observables, geometry, [_target(rootSearchBoundArcsec=1.9)]
    )
    declared_axes = (2, 0, 1)
    inverse = tuple(np.argsort(declared_axes))
    physical_components = [
        observables[f"photon_acceleration__axis{axis}"] for axis in range(3)
    ]
    stored_components = [None, None, None]
    for physical_axis, stored_axis in enumerate(declared_axes):
        stored_components[stored_axis] = np.transpose(
            physical_components[physical_axis], axes=inverse
        )
    stored_spacing = [0.0, 0.0, 0.0]
    stored_origin = [0.0, 0.0, 0.0]
    for physical_axis, stored_axis in enumerate(declared_axes):
        stored_spacing[stored_axis] = geometry["spacing"][physical_axis]
        stored_origin[stored_axis] = geometry["origin"][physical_axis]
    permuted, _rows = evaluate_observation_targets(
        _model(),
        {
            f"photon_acceleration__axis{axis}": values
            for axis, values in enumerate(stored_components)
        },
        {
            **geometry,
            "spacing": stored_spacing,
            "origin": stored_origin,
        },
        [
            _target(
                northAxis=declared_axes[0],
                eastAxis=declared_axes[1],
                lineOfSightAxis=declared_axes[2],
                rootSearchBoundArcsec=1.9,
            )
        ],
    )
    canonical_family = canonical["targets"][0]["families"][0]
    permuted_family = permuted["targets"][0]["families"][0]
    np.testing.assert_allclose(
        permuted_family["profiledSourceArcsec"],
        canonical_family["profiledSourceArcsec"],
        atol=1e-12,
    )
    assert permuted_family["matchedImages"] == canonical_family["matchedImages"]
    assert (
        permuted_family["score"]["imagePlaneRmsArcsec"]
        == canonical_family["score"]["imagePlaneRmsArcsec"]
    )
