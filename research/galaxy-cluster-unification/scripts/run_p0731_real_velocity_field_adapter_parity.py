#!/usr/bin/env python3
"""Commission the generic velocity-map adapter on real LITTLE THINGS data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0712_external_galaxy_velocity_field_validation import (
    CHANNEL_WIDTH_KM_S,
    SYSTEMIC_KM_S,
    beam_convolve_velocity,
    image2,
    velocity_unit_scale,
)

from voidscreen.field_job import (
    file_sha256,
    load_array_bundle,
    write_array_bundle,
)
from voidscreen.galaxy_maps import (
    aips_clean_beam_degrees,
    sky_pixels_to_disk_coordinates,
)
from voidscreen.observation_adapters import evaluate_observation_targets

DEFAULT_CONFIG = ROOT / "configs/p0731_real_velocity_field_adapter_parity.json"
DEFAULT_OUTPUT = ROOT / "results/p0731_real_velocity_field_adapter_parity"
KPC_M = 3.085677581491367e19


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def beam_kernel(header: fits.Header) -> np.ndarray:
    bmaj_deg, bmin_deg, bpa_deg = aips_clean_beam_degrees(header)
    pixel_x_deg = abs(float(header["CDELT1"]))
    pixel_y_deg = abs(float(header["CDELT2"]))
    major_sigma = math.sqrt(
        (bmaj_deg / pixel_x_deg / 2.354820045) * (bmaj_deg / pixel_y_deg / 2.354820045)
    )
    minor_sigma = math.sqrt(
        (bmin_deg / pixel_x_deg / 2.354820045) * (bmin_deg / pixel_y_deg / 2.354820045)
    )
    size = int(max(9, 2 * math.ceil(4.0 * major_sigma) + 1))
    if size % 2 == 0:
        size += 1
    return Gaussian2DKernel(
        major_sigma,
        minor_sigma,
        theta=np.deg2rad(90.0 + bpa_deg),
        x_size=size,
        y_size=size,
    ).array


def verify_field_artifacts(directory: Path) -> bool:
    index_path = directory / "artifact_index.json"
    manifest_path = directory / "manifest.json"
    index = read_json(index_path)
    manifest = read_json(manifest_path)
    if file_sha256(index_path) != manifest.get("artifactIndexSha256"):
        return False
    for record in index.get("artifacts", []):
        path = directory / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or file_sha256(path) != record["sha256"]
        ):
            return False
    return True


def field_cache(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    store = ROOT / config["parents"]["fieldJobStore"] / "jobs"
    model_by_name = {item["fieldModelName"]: item for item in config["models"]}
    expected_systems = set(config["systems"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for scientific_path in store.glob("*/artifacts/scientific_result.json"):
        artifacts = scientific_path.parent
        required = [
            artifacts / "job.json",
            artifacts / "model.json",
            artifacts / "input_bundle.json",
            artifacts / "observables.npz",
            artifacts / "artifact_index.json",
            artifacts / "manifest.json",
        ]
        if not all(path.is_file() for path in required):
            continue
        job = read_json(artifacts / "job.json")
        targets = job.get("observationTargets", [])
        if not targets or targets[0].get("kind") != "circular_speed_curve":
            continue
        galaxy = str(targets[0].get("id", "")).split("-published-", 1)[0]
        model = read_json(artifacts / "model.json")
        if galaxy not in expected_systems or model.get("name") not in model_by_name:
            continue
        specification = model_by_name[model["name"]]
        scientific = read_json(scientific_path)
        record_path = scientific_path.parents[1] / "record.json"
        record = read_json(record_path)
        if scientific.get("state") != "succeeded" or not scientific.get("converged"):
            raise RuntimeError(f"cached field did not succeed: {record['id']}")
        if record["preflight"]["modelSha256"] != specification["modelSha256"]:
            raise RuntimeError(f"model hash changed for {galaxy} {specification['id']}")
        key = (galaxy, specification["id"])
        if key in result:
            raise RuntimeError(f"duplicate cached field: {key}")
        result[key] = {
            "directory": artifacts,
            "record": record,
            "model": model,
            "bundle": read_json(artifacts / "input_bundle.json"),
            "scientific": scientific,
            "artifactsValid": verify_field_artifacts(artifacts),
        }
    expected = len(config["systems"]) * len(config["models"])
    if len(result) != expected:
        missing = sorted(
            (galaxy, model["id"])
            for galaxy in config["systems"]
            for model in config["models"]
            if (galaxy, model["id"]) not in result
        )
        raise RuntimeError(
            f"expected {expected} cached fields, found {len(result)}; missing={missing}"
        )
    return result


def observation_products(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    unlock = read_json(ROOT / config["parents"]["unlockManifest"])
    products: dict[str, dict[str, str]] = {}
    for record in unlock["galaxy_moment_products"]:
        products.setdefault(record["system"], {})[record["product"]] = record["filename"]
    return products


def prepare_observation(
    galaxy: str,
    config: dict[str, Any],
    metadata: pd.DataFrame,
    audit: pd.DataFrame,
    products: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, np.ndarray], fits.Header, dict[str, Any]]:
    moment_directory = ROOT / config["parents"]["momentDirectory"] / galaxy
    moment1_path = moment_directory / products[galaxy]["XMOM1"]
    moment2_path = moment_directory / products[galaxy]["XMOM2"]
    baryon_config = read_json(ROOT / "configs/p0636_little_things_baryon_acquisition.json")
    baryon_target = next(item for item in baryon_config["targets"] if item["id"] == galaxy)
    moment0_path = (
        ROOT / config["parents"]["baryonDirectory"] / galaxy / baryon_target["hi_filename"]
    )
    moment1, header1 = image2(moment1_path)
    moment2, header2 = image2(moment2_path)
    moment0, _header0 = image2(moment0_path)
    if moment1.shape != moment2.shape or moment1.shape != moment0.shape:
        raise RuntimeError(f"moment-map shape mismatch for {galaxy}")
    moment1 *= velocity_unit_scale(header1)
    moment2 *= velocity_unit_scale(header2)
    valid_velocity = np.isfinite(moment1)
    yy_all, xx_all = np.nonzero(valid_velocity)
    bmaj, _, _ = aips_clean_beam_degrees(header1)
    margin = math.ceil(4.0 * bmaj / abs(float(header1["CDELT1"])) / 2.35482)
    y0 = max(0, int(yy_all.min()) - margin)
    y1 = min(moment1.shape[0], int(yy_all.max()) + margin + 1)
    x0 = max(0, int(xx_all.min()) - margin)
    x1 = min(moment1.shape[1], int(xx_all.max()) + margin + 1)
    velocity = moment1[y0:y1, x0:x1]
    dispersion = moment2[y0:y1, x0:x1]
    intensity = np.clip(moment0[y0:y1, x0:x1], 0.0, None)
    yy, xx = np.indices(velocity.shape, dtype=float)
    xx += x0
    yy += y0
    meta = metadata.loc[galaxy]
    center = SkyCoord(
        str(meta["photometric_center_ra_j2000"]),
        str(meta["photometric_center_dec_j2000"]),
        unit=(u.hourangle, u.deg),
    )
    major_kpc, minor_kpc = sky_pixels_to_disk_coordinates(
        xx,
        yy,
        WCS(header1).celestial,
        center=center,
        position_angle_deg=float(meta["photometric_pa_deg"]),
        inclination_deg=float(meta["derived_photometric_inclination_deg"]),
        distance_mpc=float(meta["distance_mpc"]),
    )
    radius_kpc = np.hypot(major_kpc, minor_kpc)
    observed_km_s = velocity - SYSTEMIC_KM_S[galaxy]
    score_mask = (
        np.isfinite(observed_km_s)
        & np.isfinite(dispersion)
        & (dispersion >= 0.0)
        & np.isfinite(intensity)
        & (intensity > 0.0)
        & (radius_kpc <= float(audit.loc[galaxy, "hi_r995_kpc"]))
    )
    channel = CHANNEL_WIDTH_KM_S[galaxy]
    uncertainty_km_s = np.sqrt(
        np.square(np.where(np.isfinite(dispersion), dispersion, 0.0)) + (channel / 2.355) ** 2
    )
    initial_weight = np.where(
        score_mask,
        np.where(np.isfinite(intensity), intensity, 0.0)
        / np.maximum(np.square(uncertainty_km_s), 1e-12),
        0.0,
    )
    direction = np.divide(
        major_kpc,
        radius_kpc,
        out=np.zeros_like(major_kpc),
        where=radius_kpc > 0.0,
    )
    covariance = float(
        np.sum(initial_weight * np.where(score_mask, observed_km_s, 0.0) * direction)
    )
    handedness = 1 if covariance >= 0.0 else -1
    arrays = {
        "disk_major_coordinate_m": np.where(np.isfinite(major_kpc), major_kpc * KPC_M, 0.0),
        "disk_minor_coordinate_m": np.where(np.isfinite(minor_kpc), minor_kpc * KPC_M, 0.0),
        "observed_velocity_m_s": np.where(
            np.isfinite(velocity), velocity * 1000.0, SYSTEMIC_KM_S[galaxy] * 1000.0
        ),
        "velocity_uncertainty_m_s": np.where(
            np.isfinite(uncertainty_km_s), uncertainty_km_s * 1000.0, 1e30
        ),
        "hi_intensity_weight": np.where(np.isfinite(intensity), intensity, 0.0),
        "velocity_score_mask": score_mask.astype(float),
        "beam_kernel": beam_kernel(header1),
    }
    target = {
        "schemaVersion": "sigma-observation-target/1",
        "id": f"{galaxy}-real-resolved-velocity-field",
        "kind": "line_of_sight_velocity_field",
        "observable": "massive_tracer_acceleration",
        "centerM": [0.0, 0.0, 0.0],
        "planeAxes": [0, 1],
        "inclinationDeg": float(meta["derived_photometric_inclination_deg"]),
        "handedness": handedness,
        "nonPositiveInwardPolicy": config["observation"]["nonPositiveInwardPolicy"],
        "majorCoordinateArrayKey": "disk_major_coordinate_m",
        "minorCoordinateArrayKey": "disk_minor_coordinate_m",
        "observedVelocityArrayKey": "observed_velocity_m_s",
        "uncertaintyArrayKey": "velocity_uncertainty_m_s",
        "observedVelocityZeroPointMPerS": SYSTEMIC_KM_S[galaxy] * 1000.0,
        "intensityWeightArrayKey": "hi_intensity_weight",
        "scoreMaskArrayKey": "velocity_score_mask",
        "beamKernelArrayKey": "beam_kernel",
        "weighting": config["observation"]["weighting"],
        "minimumValidPixels": int(config["observation"]["minimumValidPixels"]),
        "fittedNuisanceParameters": 0,
        "provenance": {
            "dataset": "LITTLE THINGS",
            "galaxy": galaxy,
            "moment1Sha256": sha256(moment1_path),
            "moment2Sha256": sha256(moment2_path),
            "moment0Sha256": sha256(moment0_path),
            "pipeline": "frozen P0712 observation geometry and weighting",
        },
        "license": {"id": "research-source-license", "redistributionAllowed": False},
    }
    diagnostics = {
        "galaxy": galaxy,
        "map_rows": int(velocity.shape[0]),
        "map_columns": int(velocity.shape[1]),
        "score_mask_pixels": int(score_mask.sum()),
        "inclination_deg": target["inclinationDeg"],
        "handedness": handedness,
        "systemic_velocity_km_s": SYSTEMIC_KM_S[galaxy],
        "channel_width_km_s": channel,
        "moment1_sha256": sha256(moment1_path),
        "moment2_sha256": sha256(moment2_path),
        "moment0_sha256": sha256(moment0_path),
    }
    return target, arrays, header1, diagnostics


def package_observation(
    root: Path,
    galaxy: str,
    arrays: dict[str, np.ndarray],
    target: dict[str, Any],
) -> dict[str, Any]:
    descriptions = {
        "disk_major_coordinate_m": {"unit": "m", "rank": "scalar", "role": "coordinate"},
        "disk_minor_coordinate_m": {"unit": "m", "rank": "scalar", "role": "coordinate"},
        "observed_velocity_m_s": {"unit": "m/s", "rank": "scalar", "role": "observation"},
        "velocity_uncertainty_m_s": {"unit": "m/s", "rank": "scalar", "role": "uncertainty"},
        "hi_intensity_weight": {"unit": "Jy/beam*m/s", "rank": "scalar", "role": "weight"},
        "velocity_score_mask": {"unit": "1", "rank": "scalar", "role": "mask"},
        "beam_kernel": {"unit": "1", "rank": "scalar", "role": "kernel"},
    }
    bundle = write_array_bundle(
        root / galaxy,
        arrays,
        {
            "schemaVersion": "sigma-array-bundle-request/1",
            "geometry": {
                "coordinateSystem": "observed_sky_map_2d",
                "dimensions": 2,
                "lengthUnit": "pixel",
                "referenceFrame": "LITTLE_THINGS_moment_map_crop",
            },
            "arrays": descriptions,
            "provenance": target["provenance"],
            "license": target["license"],
        },
    )
    loaded, loaded_arrays = load_array_bundle(root / galaxy)
    if loaded["bundleSha256"] != bundle["bundleSha256"] or set(loaded_arrays) != set(arrays):
        raise RuntimeError(f"observation bundle verification failed for {galaxy}")
    return bundle


def independent_prediction(
    observables: dict[str, np.ndarray],
    geometry: dict[str, Any],
    target: dict[str, Any],
    arrays: dict[str, np.ndarray],
    header: fits.Header,
) -> tuple[np.ndarray, np.ndarray, float]:
    dimensions = int(geometry["dimensions"])
    spacing = np.asarray(geometry["spacing"], dtype=float)
    origin = np.asarray(geometry["origin"], dtype=float)
    components = [
        observables[f"massive_tracer_acceleration__axis{axis}"] for axis in range(dimensions)
    ]
    axes = [
        origin[index] + np.arange(components[0].shape[index]) * spacing[index]
        for index in range(dimensions)
    ]
    major = arrays["disk_major_coordinate_m"]
    minor = arrays["disk_minor_coordinate_m"]
    positions = np.zeros((major.size, dimensions), dtype=float)
    positions[:, 0] = major.ravel()
    positions[:, 1] = minor.ravel()
    sampled = np.vstack(
        [
            RegularGridInterpolator(tuple(axes), component, bounds_error=False, fill_value=np.nan)(
                positions
            )
            for component in components
        ]
    )
    radius = np.hypot(major, minor)
    radial_x = np.divide(major, radius, out=np.zeros_like(major), where=radius > 0.0)
    radial_y = np.divide(minor, radius, out=np.zeros_like(minor), where=radius > 0.0)
    inward = -(
        sampled[0].reshape(major.shape) * radial_x + sampled[1].reshape(major.shape) * radial_y
    )
    circular = np.sqrt(np.maximum(radius * inward, 0.0))
    intrinsic = (
        target["handedness"] * np.sin(np.deg2rad(target["inclinationDeg"])) * circular * radial_x
    )
    convolved, _diagnostics = beam_convolve_velocity(
        intrinsic,
        arrays["hi_intensity_weight"],
        header,
    )
    observed = arrays["observed_velocity_m_s"] - target["observedVelocityZeroPointMPerS"]
    uncertainty = arrays["velocity_uncertainty_m_s"]
    valid = (
        (arrays["velocity_score_mask"] > 0.0)
        & np.isfinite(convolved)
        & np.isfinite(observed)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
        & np.isfinite(arrays["hi_intensity_weight"])
        & (arrays["hi_intensity_weight"] > 0.0)
    )
    residual = convolved - observed
    weight = np.where(
        valid,
        arrays["hi_intensity_weight"] / np.square(uncertainty),
        0.0,
    )
    weighted_rmse = float(
        np.sqrt(
            np.sum(weight[valid] * np.square(residual[valid]))
            / np.sum(weight[valid])
        )
    )
    return convolved, valid, weighted_rmse


def adapter_prediction(
    model: dict[str, Any],
    observables: dict[str, np.ndarray],
    geometry: dict[str, Any],
    target: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    evaluation, rows = evaluate_observation_targets(
        model,
        observables,
        geometry,
        [target],
        arrays=arrays,
    )
    prediction = np.full(arrays["disk_major_coordinate_m"].shape, np.nan)
    valid = np.zeros(prediction.shape, dtype=bool)
    for row in rows:
        index = (int(row["row_index"]), int(row["column_index"]))
        prediction[index] = float(row["predicted_velocity_m_s"])
        valid[index] = True
    return evaluation["targets"][0], prediction, valid


def render_score_plot(output: Path, summaries: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(
        [row["model"] for row in summaries],
        [row["equal_galaxy_weighted_rmse_km_s"] for row in summaries],
        color=["#7f8c8d", "#2980b9", "#27ae60", "#c0392b"],
    )
    axes[0].set_ylabel("Equal-galaxy weighted RMSE (km/s)")
    axes[0].set_title("Real velocity maps; P0723 fixed fields")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        [row["model"] for row in summaries],
        [row["maximum_prediction_parity_difference_m_s"] for row in summaries],
        color=["#7f8c8d", "#2980b9", "#27ae60", "#c0392b"],
    )
    axes[1].set_ylabel("Maximum adapter/reference difference (m/s)")
    axes[1].set_yscale("log")
    axes[1].set_title("Independent P0712-operation parity")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "score_and_parity_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if config.get("status") != "frozen_before_P0731_adapter_scores":
        raise RuntimeError("P0731 protocol is not frozen")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(ROOT / config["parents"]["photometricMetadata"]).set_index("galaxy")
    audit = pd.read_csv(ROOT / config["parents"]["registeredMapAudit"]).set_index("galaxy")
    products = observation_products(config)
    cached_fields = field_cache(config)
    old_scores = pd.read_csv(
        ROOT
        / "results/p0712_external_galaxy_velocity_field_validation/per_galaxy_velocity_field_scores.csv"
    ).set_index("galaxy")
    old_report = read_json(ROOT / config["parents"]["frozenVelocityResults"])
    rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sigma-p0731-") as temporary_value:
        bundle_root = Path(temporary_value) / "observations"
        bundle_root.mkdir()
        for galaxy in config["systems"]:
            print(f"P0731 observation {galaxy}", flush=True)
            target, arrays, header, diagnostics = prepare_observation(
                galaxy, config, metadata, audit, products
            )
            bundle = package_observation(bundle_root, galaxy, arrays, target)
            bundle_rows.append(
                {
                    **diagnostics,
                    "observation_bundle_sha256": bundle["bundleSha256"],
                    "observation_bundle_arrays": len(bundle["arrays"]),
                    "observation_bundle_hashes_valid": True,
                }
            )
            for specification in config["models"]:
                print(f"P0731 field {galaxy} {specification['id']}", flush=True)
                cached = cached_fields[(galaxy, specification["id"])]
                with np.load(
                    cached["directory"] / "observables.npz", allow_pickle=False
                ) as archive:
                    observables = {key: archive[key].astype(float) for key in archive.files}
                adapter, prediction, adapter_valid = adapter_prediction(
                    cached["model"],
                    observables,
                    cached["bundle"]["geometry"],
                    target,
                    arrays,
                )
                reference, reference_valid, reference_score = independent_prediction(
                    observables,
                    cached["bundle"]["geometry"],
                    target,
                    arrays,
                    header,
                )
                support_equal = bool(np.array_equal(adapter_valid, reference_valid))
                common = adapter_valid & reference_valid
                difference = prediction[common] - reference[common]
                parity_rmse = float(np.sqrt(np.mean(np.square(difference))))
                parity_max = float(np.max(np.abs(difference)))
                adapter_score = float(adapter["score"]["declaredWeightedRmseMPerS"])
                old_name = specification["p0712Comparator"]
                old_score = (
                    float(old_scores.loc[galaxy, f"weighted_RMSE_{old_name}"])
                    if old_name is not None
                    else None
                )
                rows.append(
                    {
                        "galaxy": galaxy,
                        "model": specification["id"],
                        "field_job_id": cached["record"]["id"],
                        "field_job_sha256": cached["scientific"]["jobSha256"],
                        "field_result_sha256": cached["scientific"]["resultSha256"],
                        "field_input_bundle_sha256": cached["bundle"]["bundleSha256"],
                        "observation_bundle_sha256": bundle["bundleSha256"],
                        "field_artifact_hashes_valid": cached["artifactsValid"],
                        "universal_gravity_parameters": cached["scientific"]["parameterAccounting"][
                            "universalCount"
                        ],
                        "per_object_gravity_parameters": cached["scientific"][
                            "parameterAccounting"
                        ]["perObjectCount"],
                        "adapter_valid_pixels": int(adapter_valid.sum()),
                        "independent_valid_pixels": int(reference_valid.sum()),
                        "valid_pixel_support_exact": support_equal,
                        "adapter_declared_weighted_rmse_km_s": adapter_score / 1000.0,
                        "independent_declared_weighted_rmse_km_s": reference_score / 1000.0,
                        "weighted_rmse_parity_difference_m_s": abs(adapter_score - reference_score),
                        "prediction_parity_rmse_m_s": parity_rmse,
                        "prediction_parity_max_abs_m_s": parity_max,
                        "p0712_original_weighted_rmse_km_s": old_score,
                        "p0712_to_p0731_score_change_km_s": (adapter_score / 1000.0 - old_score)
                        if old_score is not None
                        else None,
                    }
                )

    summaries: list[dict[str, Any]] = []
    for specification in config["models"]:
        selected = [row for row in rows if row["model"] == specification["id"]]
        values = np.asarray([row["adapter_declared_weighted_rmse_km_s"] for row in selected])
        old_value = old_report["sample_weighted_RMSE_km_s"].get(specification["p0712Comparator"])
        summaries.append(
            {
                "model": specification["id"],
                "galaxies": len(selected),
                "equal_galaxy_weighted_rmse_km_s": float(np.sqrt(np.mean(np.square(values)))),
                "p0712_original_equal_galaxy_weighted_rmse_km_s": old_value,
                "maximum_prediction_parity_rmse_m_s": max(
                    row["prediction_parity_rmse_m_s"] for row in selected
                ),
                "maximum_prediction_parity_difference_m_s": max(
                    row["prediction_parity_max_abs_m_s"] for row in selected
                ),
                "maximum_weighted_rmse_parity_difference_m_s": max(
                    row["weighted_rmse_parity_difference_m_s"] for row in selected
                ),
                "minimum_valid_pixels": min(row["adapter_valid_pixels"] for row in selected),
                "universal_gravity_parameters": selected[0]["universal_gravity_parameters"],
                "per_object_gravity_parameters": selected[0]["per_object_gravity_parameters"],
            }
        )
    gates = config["engineeringGates"]
    gate_results = {
        "required_systems": len({row["galaxy"] for row in rows}) == int(gates["requiredSystems"]),
        "required_models": len({row["model"] for row in rows}) == int(gates["requiredModels"]),
        "required_evaluations": len(rows) == int(gates["requiredEvaluations"]),
        "minimum_valid_pixels": all(
            row["adapter_valid_pixels"] >= int(gates["minimumValidPixelsPerEvaluation"])
            for row in rows
        ),
        "prediction_parity_rmse": all(
            row["prediction_parity_rmse_m_s"]
            <= float(gates["maximumAdapterToIndependentPredictionRmseMPerS"])
            for row in rows
        ),
        "prediction_parity_maximum": all(
            row["prediction_parity_max_abs_m_s"]
            <= float(gates["maximumAdapterToIndependentPredictionAbsoluteDifferenceMPerS"])
            for row in rows
        ),
        "score_parity": all(
            row["weighted_rmse_parity_difference_m_s"]
            <= float(gates["maximumAdapterToIndependentDeclaredWeightedRmseDifferenceMPerS"])
            for row in rows
        ),
        "exact_valid_pixel_support": all(row["valid_pixel_support_exact"] for row in rows)
        if gates["exactValidPixelSupport"]
        else True,
        "no_per_object_gravity_parameters": all(
            row["per_object_gravity_parameters"] <= int(gates["maximumPerObjectGravityParameters"])
            for row in rows
        ),
        "field_artifact_hashes": all(row["field_artifact_hashes_valid"] for row in rows),
        "observation_bundle_hashes": all(
            row["observation_bundle_hashes_valid"] for row in bundle_rows
        ),
    }
    report = {
        "schemaVersion": "sigma-p0731-real-velocity-field-adapter-parity-result/1",
        "stage": config["stage"],
        "status": "pass" if all(gate_results.values()) else "fail",
        "sampleStatus": config["sampleStatus"],
        "systems": len({row["galaxy"] for row in rows}),
        "models": len(summaries),
        "evaluations": len(rows),
        "modelSummaries": summaries,
        "gateResults": gate_results,
        "failedGates": [name for name, value in gate_results.items() if not value],
        "configSha256": sha256(config_path),
        "adapterSourceSha256": sha256(ROOT / "src/voidscreen/observation_adapters.py"),
        "frozenReferenceSourceSha256": sha256(ROOT / config["parents"]["frozenVelocityPipeline"]),
        "claimBoundary": config["claimBoundary"],
    }
    write_csv(output / "observation_bundle_manifest.csv", bundle_rows)
    write_csv(output / "per_galaxy_model_scores.csv", rows)
    write_csv(output / "model_summary.csv", summaries)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render_score_plot(output, summaries)
    summary_lines = [
        "# P0731 real velocity-field adapter parity",
        "",
        f"- Status: **{report['status'].upper()}**.",
        f"- Real galaxies / fixed models / evaluations: **{report['systems']} / {report['models']} / {report['evaluations']}**.",
        f"- Maximum prediction parity RMSE: **{max(row['prediction_parity_rmse_m_s'] for row in rows):.3g} m/s**.",
        f"- Maximum absolute pixel difference: **{max(row['prediction_parity_max_abs_m_s'] for row in rows):.3g} m/s**.",
        f"- Maximum weighted-score difference: **{max(row['weighted_rmse_parity_difference_m_s'] for row in rows):.3g} m/s**.",
        "- Per-object gravity parameters: **0**.",
        "",
        "These are spent-sample engineering and real-data scores for the P0723 fixed fields, not a new blind theory validation.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
