"""Run the frozen P0734 typed photon-lensing acceptance protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_job import _write_deterministic_npz
from voidscreen.observation_adapters import evaluate_observation_targets
from voidscreen.sky_lensing import C_M_S

CONFIG = ROOT / "configs" / "p0734_typed_photon_lensing_adapter.json"
RESULT = ROOT / "results" / "p0734_typed_photon_lensing_adapter" / "report.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model(target: str = "photons") -> dict:
    return {
        "observables": [
            {
                "id": "photon_acceleration",
                "target": target,
                "rank": "vector",
                "unit": "m/s^2",
            }
        ]
    }


def geometry(shape, spacing) -> dict:
    return {
        "coordinateSystem": "cartesian_3d",
        "dimensions": 3,
        "spacing": list(spacing),
        "origin": [
            -0.5 * (count - 1) * step
            for count, step in zip(shape, spacing, strict=True)
        ],
    }


def target(**changes) -> dict:
    return {
        "schemaVersion": "sigma-observation-target/1",
        "id": "p0734-acceptance",
        "kind": "photon_lensing_map",
        "observable": "photon_acceleration",
        "northAxis": 0,
        "eastAxis": 1,
        "lineOfSightAxis": 2,
        "distanceRatio": 0.7,
        "lensAngularDiameterDistanceM": 1.0e20,
        "minimumValidPixels": 25,
        "provenance": {"kind": "P0734 analytic acceptance"},
        "license": {"id": "CC0-1.0", "redistributionAllowed": True},
        **changes,
    }


def evaluate(observables, shape, spacing, specification, arrays=None):
    maps: dict[str, np.ndarray] = {}
    evaluation, _rows = evaluate_observation_targets(
        model(),
        observables,
        geometry(shape, spacing),
        [specification],
        arrays=arrays,
        map_outputs=maps,
    )
    return evaluation, maps


def uniform_metrics() -> dict:
    shape = (9, 11, 13)
    spacing = (2.0, 3.0, 5.0)
    values = {
        "photon_acceleration__axis0": np.full(shape, 4.0),
        "photon_acceleration__axis1": np.full(shape, -3.0),
        "photon_acceleration__axis2": np.zeros(shape),
    }
    _evaluation, baseline = evaluate(values, shape, spacing, target())
    path = (shape[2] - 1) * spacing[2]
    expected_north = -2.0 * 0.7 * path * 4.0 / C_M_S**2
    expected_east = -2.0 * 0.7 * path * -3.0 / C_M_S**2
    relative_error = max(
        float(
            np.max(
                np.abs(
                    baseline["target_000__alpha_north_radian"] / expected_north
                    - 1.0
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    baseline["target_000__alpha_east_radian"] / expected_east
                    - 1.0
                )
            )
        ),
    )
    _evaluation, doubled_ratio = evaluate(
        values, shape, spacing, target(distanceRatio=1.4)
    )
    ratio_error = float(
        np.max(
            np.abs(
                doubled_ratio["target_000__alpha_east_radian"]
                / baseline["target_000__alpha_east_radian"]
                - 2.0
            )
        )
        / 2.0
    )
    long_shape = (9, 11, 25)
    long_values = {
        "photon_acceleration__axis0": np.full(long_shape, 4.0),
        "photon_acceleration__axis1": np.full(long_shape, -3.0),
        "photon_acceleration__axis2": np.zeros(long_shape),
    }
    _evaluation, doubled_path = evaluate(
        long_values, long_shape, spacing, target()
    )
    path_error = float(
        np.max(
            np.abs(
                doubled_path["target_000__alpha_east_radian"]
                / baseline["target_000__alpha_east_radian"]
                - 2.0
            )
        )
        / 2.0
    )
    return {
        "uniformFieldRelativeError": relative_error,
        "distanceRatioScalingRelativeError": ratio_error,
        "pathLengthScalingRelativeError": path_error,
        "signConvention": "positive transverse acceleration produces negative deflection",
    }


def affine_metrics() -> tuple[dict, dict[str, np.ndarray]]:
    shape = (17, 19, 21)
    spacing = (2.0e17, 3.0e17, 4.0e17)
    lens_distance = 2.0e22
    distance_ratio = 0.65
    coordinates = [
        origin + np.arange(count) * step
        for origin, count, step in zip(
            geometry(shape, spacing)["origin"], shape, spacing, strict=True
        )
    ]
    north_m, east_m, _los_m = np.meshgrid(*coordinates, indexing="ij")
    north_angle = north_m / lens_distance
    east_angle = east_m / lens_distance
    alpha_east = 0.04 * east_angle + 0.01 * north_angle
    alpha_north = 0.01 * east_angle + 0.02 * north_angle
    path = (shape[2] - 1) * spacing[2]
    field_scale = -(C_M_S**2) / (2.0 * distance_ratio * path)
    values = {
        "photon_acceleration__axis0": field_scale * alpha_north,
        "photon_acceleration__axis1": field_scale * alpha_east,
        "photon_acceleration__axis2": np.zeros(shape),
    }
    _evaluation, maps = evaluate(
        values,
        shape,
        spacing,
        target(
            distanceRatio=distance_ratio,
            lensAngularDiameterDistanceM=lens_distance,
        ),
    )
    expected = {
        "convergence": 0.03,
        "shear_1": 0.01,
        "shear_2": 0.01,
        "rotation": 0.0,
        "jacobian_determinant": 0.96 * 0.98 - 0.01**2,
    }
    errors = {
        name: float(np.max(np.abs(maps[f"target_000__{name}"] - value)))
        for name, value in expected.items()
    }
    return (
        {
            "maximumInvariantAbsoluteError": max(errors.values()),
            "gradientFieldRotationRms": float(
                np.sqrt(np.mean(np.square(maps["target_000__rotation"])))
            ),
            "perMapAbsoluteErrors": errors,
        },
        maps,
    )


def point_mass_metrics() -> dict:
    shape = (65, 65, 257)
    spacing = (1.0, 1.0, 1.0)
    coordinates = [
        -0.5 * (count - 1) + np.arange(count) for count in shape
    ]
    north, east, line_of_sight = np.meshgrid(*coordinates, indexing="ij")
    radius_squared = north**2 + east**2 + line_of_sight**2
    radius = np.sqrt(np.maximum(radius_squared, 0.25))
    gravity_mass = 2.0e24
    scale = -gravity_mass / radius**3
    values = {
        "photon_acceleration__axis0": scale * north,
        "photon_acceleration__axis1": scale * east,
        "photon_acceleration__axis2": scale * line_of_sight,
    }
    _evaluation, maps = evaluate(
        values,
        shape,
        spacing,
        target(distanceRatio=1.0, lensAngularDiameterDistanceM=1.0e9),
    )
    offsets = np.arange(4, 33)
    center = shape[0] // 2
    predicted = np.abs(
        maps["target_000__alpha_east_radian"][center, center + offsets]
    )
    expected = 4.0 * gravity_mass / (C_M_S**2 * offsets)
    relative = np.abs(predicted / expected - 1.0)
    return {
        "impactParameterMinimumCells": int(offsets.min()),
        "impactParameterMaximumCells": int(offsets.max()),
        "medianRelativeError": float(np.median(relative)),
        "p95RelativeError": float(np.quantile(relative, 0.95)),
        "finiteLineOfSightHalfLengthCells": 128,
    }


def scoring_metrics(affine_maps: dict[str, np.ndarray]) -> dict:
    shape = affine_maps["target_000__alpha_east_arcsec"].shape
    cube_shape = (shape[0], shape[1], 21)
    spacing = (2.0e17, 3.0e17, 4.0e17)
    lens_distance = 2.0e22
    distance_ratio = 0.65
    coordinates = [
        origin + np.arange(count) * step
        for origin, count, step in zip(
            geometry(cube_shape, spacing)["origin"], cube_shape, spacing, strict=True
        )
    ]
    north_m, east_m, _los_m = np.meshgrid(*coordinates, indexing="ij")
    north_angle = north_m / lens_distance
    east_angle = east_m / lens_distance
    alpha_east = 0.04 * east_angle + 0.01 * north_angle
    alpha_north = 0.01 * east_angle + 0.02 * north_angle
    path = (cube_shape[2] - 1) * spacing[2]
    field_scale = -(C_M_S**2) / (2.0 * distance_ratio * path)
    values = {
        "photon_acceleration__axis0": field_scale * alpha_north,
        "photon_acceleration__axis1": field_scale * alpha_east,
        "photon_acceleration__axis2": np.zeros(cube_shape),
    }
    arrays = {
        "alpha_e": affine_maps["target_000__alpha_east_arcsec"],
        "alpha_n": affine_maps["target_000__alpha_north_arcsec"],
        "alpha_sigma": np.full(shape, 0.05),
        "g1": affine_maps["target_000__reduced_shear_1"],
        "g2": affine_maps["target_000__reduced_shear_2"],
        "g_sigma": np.full(shape, 0.01),
    }
    evaluation, _maps = evaluate(
        values,
        cube_shape,
        spacing,
        target(
            distanceRatio=distance_ratio,
            lensAngularDiameterDistanceM=lens_distance,
            observedAlphaEastArcsecArrayKey="alpha_e",
            observedAlphaNorthArcsecArrayKey="alpha_n",
            deflectionUncertaintyArcsecArrayKey="alpha_sigma",
            observedReducedShear1ArrayKey="g1",
            observedReducedShear2ArrayKey="g2",
            reducedShearUncertaintyArrayKey="g_sigma",
        ),
        arrays=arrays,
    )
    channels = evaluation["channelAggregates"]
    return {
        "channelNames": sorted(channels),
        "deflectionRmseArcsec": channels["deflection_arcsec"]["rmse"],
        "reducedShearRmse": channels["reduced_shear_dimensionless"]["rmse"],
        "legacyVelocityRmse": evaluation["rmseMPerS"],
    }


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gates = config["numericalGates"]
    uniform = uniform_metrics()
    affine, affine_maps = affine_metrics()
    point_mass = point_mass_metrics()
    scoring = scoring_metrics(affine_maps)
    with tempfile.TemporaryDirectory(prefix="p0734-npz-") as raw:
        temporary = Path(raw)
        first = temporary / "first.npz"
        second = temporary / "second.npz"
        _write_deterministic_npz(first, affine_maps)
        _write_deterministic_npz(second, affine_maps)
        deterministic_archive = first.read_bytes() == second.read_bytes()

    photon_tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_photon_lensing_adapter.py",
            "tests/test_observation_evaluation_job.py::test_decoupled_photon_maps_byte_match_integrated_field_job",
            "-q",
        ],
        ROOT,
    )
    hosted_tests = run(["npm.cmd", "test"], ROOT / "hosted-simulator")
    build = run(["npm.cmd", "run", "build"], ROOT / "hosted-simulator")
    hosted_count = 0
    for line in hosted_tests.stdout.splitlines():
        if line.startswith("# tests "):
            hosted_count = int(line.rsplit(" ", maxsplit=1)[1])

    gate_results = {
        "uniform_field_normalization": uniform["uniformFieldRelativeError"]
        <= gates["uniformFieldRelativeErrorMaximum"],
        "linear_path_length_scaling": uniform["pathLengthScalingRelativeError"]
        <= gates["linearPathLengthRatioRelativeErrorMaximum"],
        "linear_distance_ratio_scaling": uniform[
            "distanceRatioScalingRelativeError"
        ]
        <= gates["linearDistanceRatioRelativeErrorMaximum"],
        "affine_lensing_invariants": affine["maximumInvariantAbsoluteError"]
        <= gates["affineInvariantAbsoluteErrorMaximum"],
        "point_mass_median_normalization": point_mass["medianRelativeError"]
        <= gates["pointMassMedianRelativeErrorMaximum"],
        "point_mass_p95_normalization": point_mass["p95RelativeError"]
        <= gates["pointMassP95RelativeErrorMaximum"],
        "gradient_field_rotation": affine["gradientFieldRotationRms"]
        <= gates["gradientFieldRotationRmsMaximum"],
        "exact_deflection_scoring": scoring["deflectionRmseArcsec"]
        <= gates["exactSyntheticDeflectionRmseArcsecMaximum"],
        "exact_reduced_shear_scoring": scoring["reducedShearRmse"]
        <= gates["exactSyntheticReducedShearRmseMaximum"],
        "channel_separation": scoring["channelNames"]
        == ["deflection_arcsec", "reduced_shear_dimensionless"]
        and scoring["legacyVelocityRmse"] is None,
        "deterministic_map_archive": deterministic_archive,
        "integrated_and_decoupled_parity": photon_tests.returncode == 0,
        "hosted_preflight_and_batch_acceptance": hosted_tests.returncode == 0
        and hosted_count >= 68,
        "static_build": build.returncode == 0,
        "no_per_object_gravity_parameters": gates[
            "maximumPerObjectGravityParameters"
        ]
        == 0,
        "observation_adds_no_gravity_parameters": gates[
            "maximumGravityParametersAddedByObservationEvaluation"
        ]
        == 0,
    }
    failed = [name for name, passed in gate_results.items() if not passed]
    source_paths = [
        "src/voidscreen/photon_lensing_adapter.py",
        "src/voidscreen/observation_adapters.py",
        "src/voidscreen/field_job.py",
        "src/voidscreen/observation_evaluation_job.py",
        "src/voidscreen/sky_lensing.py",
        "hosted-simulator/lib/observation-target.mjs",
        "hosted-simulator/lib/field-job-preflight.mjs",
        "hosted-simulator/lib/observation-evaluation-preflight.mjs",
        "hosted-simulator/lib/batch-preflight.mjs",
        "hosted-simulator/lib/local-batch-service.mjs",
        "hosted-simulator/schemas/observation-target-v1.schema.json",
        "tests/test_photon_lensing_adapter.py",
        "tests/test_observation_evaluation_job.py",
        "scripts/run_p0734_typed_photon_lensing_adapter.py",
    ]
    report = {
        "schemaVersion": "sigma-p0734-typed-photon-lensing-adapter-result/1",
        "stage": "P0734",
        "status": "pass" if not failed else "fail",
        "configSha256": file_sha256(CONFIG),
        "parentCommit": config["parent"]["commit"],
        "analyticAcceptance": {
            "uniformField": uniform,
            "affineField": affine,
            "pointMass": point_mass,
            "syntheticScoring": scoring,
        },
        "executionAcceptance": {
            "photonPytestReturnCode": photon_tests.returncode,
            "hostedTestReturnCode": hosted_tests.returncode,
            "hostedTestCount": hosted_count,
            "staticBuildReturnCode": build.returncode,
            "deterministicMapArchive": deterministic_archive,
            "integratedAndDecoupledMapParity": photon_tests.returncode == 0,
        },
        "gateResults": gate_results,
        "failedGates": failed,
        "sourceSha256": {
            relative: file_sha256(ROOT / relative) for relative in source_paths
        },
        "claimBoundary": config["claimBoundary"],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if photon_tests.returncode:
        print(photon_tests.stdout, file=sys.stderr)
        print(photon_tests.stderr, file=sys.stderr)
    if hosted_tests.returncode:
        print(hosted_tests.stdout, file=sys.stderr)
        print(hosted_tests.stderr, file=sys.stderr)
    if build.returncode:
        print(build.stdout, file=sys.stderr)
        print(build.stderr, file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
