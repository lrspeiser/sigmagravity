"""Run the preregistered one-shot resolved-galaxy holdout velocity test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p0739_spiral_baryonic_registration as registration  # noqa: E402
from run_p0742_spiral_twin_roundtrip_development import circular_speed_map, canonical_sha256, sha256  # noqa: E402
from run_p0744_development_velocity_field_reveal import (  # noqa: E402
    beam_convolve_velocity,
    error_band,
    field_scores,
    velocity_unit_scale,
    weighted_median,
)
from run_p0747_post_reveal_kinematic_axis_diagnostic import fit_unit_phase  # noqa: E402
from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0752_final_holdout_velocity_field_test.json"
DEFAULT_OUTPUT = ROOT / "results/p0752_final_holdout_velocity_field_test"
P0738_RAW = ROOT / "data/raw/p0738_things_sings_resolved"
SPARC_TABLE = ROOT / "data/raw/sparc/table1.dat"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def radial_phase_points(
    observed: np.ndarray,
    source_speed: np.ndarray,
    twin_speed: np.ndarray,
    gas: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    axis: np.ndarray,
    inclination_deg: float,
    unit: np.ndarray,
    beam_kpc: float,
    hi_radius_kpc: float,
    bins: int = 20,
) -> list[dict[str, float]]:
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)
    cos_azimuth = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0.0)
    sin_azimuth = np.divide(yy, radius, out=np.zeros_like(yy), where=radius > 0.0)
    unit_projection = unit[0] * cos_azimuth + unit[1] * sin_azimuth
    projection = math.sin(math.radians(inclination_deg)) * unit_projection
    valid = mask & (np.abs(unit_projection) >= 0.5) & (np.abs(projection) > 0.0)
    observed_rotation = np.divide(
        observed, projection, out=np.full_like(observed, np.nan), where=np.abs(projection) > 0.0
    )
    observed_rotation = np.where(observed_rotation > 0.0, observed_rotation, np.nan)
    edges = np.linspace(beam_kpc, min(hi_radius_kpc, float(axis[-1])), bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        selected = valid & np.isfinite(observed_rotation) & (radius >= edges[index]) & (radius < edges[index + 1])
        if int(selected.sum()) < 8:
            continue
        weights = gas[selected] / np.square(uncertainty[selected])
        rows.append(
            {"radius_kpc": float(np.sum(weights * radius[selected]) / np.sum(weights)),
             "observed_rotation_km_s": weighted_median(observed_rotation[selected], weights),
             "source_prediction_km_s": float(np.sum(weights * source_speed[selected]) / np.sum(weights)),
             "twin_prediction_km_s": float(np.sum(weights * twin_speed[selected]) / np.sum(weights)),
             "pixels": int(selected.sum())}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)

    acquisition = read_json(ROOT / config["parents"]["resolvedAcquisition"])
    if acquisition["manifestSha256"] != config["parents"]["resolvedAcquisitionManifestSha256"]:
        raise ValueError("resolved acquisition manifest hash mismatch")
    expected_hashes = {(row["galaxy"], row["kind"]): row["sha256"] for row in acquisition["files"]}
    registered_result = ROOT / config["parents"]["registeredBaryonsResultPath"]
    twin_result = ROOT / config["parents"]["adaptiveTwinsResultPath"]
    registered_report = read_json(registered_result / "report.json")
    twin_report = read_json(twin_result / "report.json")
    if registered_report["configSha256"] != config["parents"]["registeredBaryonsConfigSha256"]:
        raise ValueError("registered-baryon config hash mismatch")
    if twin_report["configSha256"] != config["parents"]["adaptiveTwinsConfigSha256"]:
        raise ValueError("adaptive-twin config hash mismatch")
    policy_report = read_json(ROOT / config["parents"]["observationPolicyDevelopment"])
    if policy_report["reportSha256"] != config["parents"]["observationPolicyDevelopmentResultSha256"]:
        raise ValueError("observation-policy development hash mismatch")

    audit = pd.read_csv(registered_result / "map_audit.csv").set_index("galaxy")
    twin_catalog = pd.read_csv(twin_result / "selected_parameter_catalog.csv").set_index("galaxy")
    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    coordinates = registration.load_coordinates()
    args.output.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []
    target_arrays_opened = 0
    validation_arrays_opened = 0
    holdout_arrays_opened = 0
    hashes_match = True

    for galaxy in config["systems"]:
        row = audit.loc[galaxy]
        split = str(row.split)
        if split not in set(config["eligibleSplits"]):
            raise ValueError(f"target split is outside preregistration: {galaxy} ({split})")
        source_path = registered_result / "maps" / f"{galaxy}.npz"
        source_hash = next(item["sha256"] for item in registered_report["mapFiles"] if item["galaxy"] == galaxy)
        twin_path = twin_result / "selected" / "generated_maps" / f"{galaxy}.npz"
        if sha256(source_path) != source_hash:
            raise ValueError(f"registered source hash mismatch for {galaxy}")
        if sha256(twin_path) != str(twin_catalog.loc[galaxy].generated_map_sha256):
            raise ValueError(f"adaptive twin hash mismatch for {galaxy}")
        with np.load(source_path) as payload:
            axis = np.asarray(payload["axis_kpc"], dtype=float)
            source = {key: np.asarray(payload[key], dtype=float) for key in ("gas", "total")}
        with np.load(twin_path) as payload:
            twin = {key: np.asarray(payload[key], dtype=float) for key in ("gas", "total")}

        raw = P0738_RAW / galaxy
        moment1_path = next(raw.glob("*MOM1_THINGS.FITS"))
        moment2_path = next(raw.glob("*MOM2_THINGS.FITS"))
        for kind, path in (("moment1", moment1_path), ("moment2", moment2_path)):
            if sha256(path) != expected_hashes[(galaxy, kind)]:
                hashes_match = False
                raise ValueError(f"target hash mismatch for {galaxy} {kind}")
        with fits.open(moment1_path, memmap=True) as hdus:
            moment1 = np.asarray(hdus[0].data, dtype=float).squeeze()
            header1 = hdus[0].header.copy()
        with fits.open(moment2_path, memmap=True) as hdus:
            moment2 = np.asarray(hdus[0].data, dtype=float).squeeze()
            header2 = hdus[0].header.copy()
        target_arrays_opened += 2
        validation_arrays_opened += 2 if split == "validation" else 0
        holdout_arrays_opened += 2 if split == "holdout" else 0
        moment1 *= velocity_unit_scale(header1)
        moment2 *= velocity_unit_scale(header2)
        meta = metadata.loc[galaxy]
        center = coordinates[galaxy]
        x1, y1 = registration.plane_world_pixels(
            axis, center, float(row.photometric_position_angle_deg), float(row.inclination_deg),
            float(row.distance_mpc), registration.celestial_wcs(header1)
        )
        x2, y2 = registration.plane_world_pixels(
            axis, center, float(row.photometric_position_angle_deg), float(row.inclination_deg),
            float(row.distance_mpc), registration.celestial_wcs(header2)
        )
        observed_absolute = registration.sample_plane(moment1, x1, y1)
        dispersion = registration.sample_plane(moment2, x2, y2)
        xx, yy = np.meshgrid(axis, axis)
        radius = np.hypot(xx, yy)
        core = (
            np.isfinite(observed_absolute) & np.isfinite(source["gas"]) & (source["gas"] > 0.0)
            & (radius <= 0.5 * float(meta.effective_radius_kpc))
        )
        systemic = weighted_median(observed_absolute[core], source["gas"][core])
        observed = observed_absolute - systemic
        cos_azimuth = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0.0)
        sin_azimuth = np.divide(yy, radius, out=np.zeros_like(yy), where=radius > 0.0)
        convention_mask = (
            np.isfinite(observed) & (source["gas"] > 0.0) & (radius >= float(row.things_beam_kpc))
            & (radius <= float(meta.HI_radius_kpc))
        )
        covariance = float(
            np.sum(source["gas"][convention_mask] * observed[convention_mask] * cos_azimuth[convention_mask])
        )
        handedness = 1 if covariance >= 0.0 else -1
        uncertainty = np.sqrt(
            np.square(np.where(np.isfinite(dispersion), np.maximum(dispersion, 0.0), np.nan))
            + (5.2 / 2.355) ** 2
        )
        score_mask = (
            np.isfinite(observed) & np.isfinite(dispersion) & (dispersion >= 0.0)
            & (source["gas"] > 0.0) & (radius >= float(row.things_beam_kpc))
            & (radius <= float(meta.HI_radius_kpc))
        )
        unit, fitted_amplitude, explained_fraction = fit_unit_phase(
            observed, cos_azimuth, sin_azimuth, source["gas"], uncertainty, score_mask
        )
        old_unit = np.asarray([handedness, 0.0], dtype=float)
        axis_offset_deg = float(np.degrees(np.arccos(np.clip(np.dot(old_unit, unit), -1.0, 1.0))))
        inclination_projection = math.sin(math.radians(float(row.inclination_deg)))
        image_projection = inclination_projection * handedness * cos_azimuth
        kinematic_projection = inclination_projection * (unit[0] * cos_azimuth + unit[1] * sin_azimuth)

        per_model_maps: dict[str, dict[str, np.ndarray]] = {}
        for model in config["models"]:
            kwargs = {
                "model": model["id"],
                "gravitational_constant": float(model["gravitationalConstantM3KgS2"]),
                "a0": float(model["a0MPerS2"]) if "a0MPerS2" in model else None,
                "padding_factor": float(config["fieldSettings"]["paddingFactor"]),
            }
            source_speed = circular_speed_map(source["total"], axis, **kwargs)
            twin_speed = circular_speed_map(twin["total"], axis, **kwargs)
            predictions = {
                "registered_baryons_photometric_axis": beam_convolve_velocity(
                    source_speed * image_projection, source["gas"], float(row.things_beam_kpc),
                    float(row.spacing_kpc)
                ),
                "registered_baryons_kinematic_axis": beam_convolve_velocity(
                    source_speed * kinematic_projection, source["gas"], float(row.things_beam_kpc),
                    float(row.spacing_kpc)
                ),
                "adaptive_twin_kinematic_axis": beam_convolve_velocity(
                    twin_speed * kinematic_projection, source["gas"], float(row.things_beam_kpc),
                    float(row.spacing_kpc)
                ),
            }
            transport = field_scores(
                predictions["adaptive_twin_kinematic_axis"],
                predictions["registered_baryons_kinematic_axis"], uncertainty, source["gas"], score_mask
            )["gas_weighted_rmse_km_s"]
            for kind, prediction in predictions.items():
                metrics = field_scores(prediction, observed, uncertainty, source["gas"], score_mask)
                score_rows.append(
                    {"galaxy": galaxy, "model": model["id"], "prediction_kind": kind, **metrics,
                     "error_band": error_band(metrics["field_error_ratio"]),
                     "twin_source_transport_rmse_km_s": transport,
                     "gravity_parameters_fitted": 0, "dark_matter_parameters": 0}
                )
            for point in radial_phase_points(
                observed, source_speed, twin_speed, source["gas"], uncertainty, score_mask, axis,
                float(row.inclination_deg), unit, float(row.things_beam_kpc), float(meta.HI_radius_kpc)
            ):
                radial_rows.append({"galaxy": galaxy, "model": model["id"], **point})
            per_model_maps[model["id"]] = predictions

        nuisance_rows.append(
            {"galaxy": galaxy, "systemic_velocity_km_s": systemic, "handedness": handedness,
             "inclination_deg": float(row.inclination_deg),
             "image_position_angle_deg": float(row.photometric_position_angle_deg),
             "kinematic_phase_offset_deg_in_registered_plane": axis_offset_deg,
             "kinematic_unit_x": float(unit[0]), "kinematic_unit_y": float(unit[1]),
             "fitted_first_harmonic_amplitude_km_s": fitted_amplitude,
             "first_harmonic_explained_variance_fraction": explained_fraction,
             "fitted_amplitude_used_in_prediction": False, "scored_pixels": int(score_mask.sum()),
             "median_dispersion_km_s": float(np.nanmedian(dispersion[score_mask])),
             "selected_twin_coefficients_per_component": int(twin_catalog.loc[galaxy].coefficients_per_component),
             "moment1_sha256": sha256(moment1_path), "moment2_sha256": sha256(moment2_path),
             "fitted_observation_nuisances": 2, "gravity_parameters_fitted": 0}
        )
        mond = per_model_maps["fixed_simple_mond"]
        atlas_rows.append(
            {"galaxy": galaxy, "observed": observed, "mask": score_mask,
             "image": mond["registered_baryons_photometric_axis"],
             "source": mond["registered_baryons_kinematic_axis"],
             "twin": mond["adaptive_twin_kinematic_axis"]}
        )
        print(
            f"{galaxy}: pixels={score_mask.sum()}, axis offset={axis_offset_deg:.2f} deg, "
            f"harmonic fraction={explained_fraction:.3f}"
        )

    scores = pd.DataFrame(score_rows)
    radial = pd.DataFrame(radial_rows)
    nuisance = pd.DataFrame(nuisance_rows)
    scores.to_csv(args.output / "holdout_velocity_field_scores.csv", index=False)
    radial.to_csv(args.output / "holdout_radial_speed_points.csv", index=False)
    nuisance.to_csv(args.output / "holdout_observation_nuisance_audit.csv", index=False)

    figure, axes = plt.subplots(len(atlas_rows), 5, figsize=(17, 3.2 * len(atlas_rows)), constrained_layout=True)
    for row_index, item in enumerate(atlas_rows):
        finite = item["mask"] & np.isfinite(item["observed"])
        limit = float(np.quantile(np.abs(item["observed"][finite]), 0.99))
        residual = item["source"] - item["observed"]
        residual_limit = max(float(np.quantile(np.abs(residual[finite]), 0.99)), 1.0)
        panels = [
            (item["observed"], "observed H I velocity", -limit, limit),
            (item["image"], "MOND · image axis", -limit, limit),
            (item["source"], "MOND · kinematic axis", -limit, limit),
            (item["twin"], "MOND · adaptive twin", -limit, limit),
            (residual, "source - observed", -residual_limit, residual_limit),
        ]
        for column, (values, title, vmin, vmax) in enumerate(panels):
            shown = np.where(item["mask"], values, np.nan)
            image = axes[row_index, column].imshow(
                shown, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
            )
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            figure.colorbar(image, ax=axes[row_index, column], fraction=0.046, label="km/s")
        axes[row_index, 0].set_ylabel(item["galaxy"])
    figure.suptitle("P0752 final holdout: observed fields versus fixed MOND")
    figure.savefig(args.output / "final_holdout_velocity_field_atlas.png", dpi=170)
    plt.close(figure)

    curve_figure, curve_axes = plt.subplots(
        len(config["systems"]), len(config["models"]), figsize=(12, 4.2 * len(config["systems"])),
        constrained_layout=True, squeeze=False
    )
    for row_index, galaxy in enumerate(config["systems"]):
        for column, model in enumerate(config["models"]):
            subset = radial[(radial.galaxy == galaxy) & (radial.model == model["id"])].sort_values("radius_kpc")
            ax = curve_axes[row_index, column]
            ax.plot(subset.radius_kpc, subset.observed_rotation_km_s, "o", label="observed", ms=4)
            ax.plot(subset.radius_kpc, subset.source_prediction_km_s, "-", label="registered baryons")
            ax.plot(subset.radius_kpc, subset.twin_prediction_km_s, "--", label="adaptive twin")
            ax.set_title(f"{galaxy} · {model['id']}")
            ax.set_xlabel("radius (kpc)")
            ax.set_ylabel("circular speed (km/s)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    curve_figure.suptitle("P0752 deprojected holdout speed curves")
    curve_figure.savefig(args.output / "final_holdout_radial_speed_curves.png", dpi=170)
    plt.close(curve_figure)

    source_scores = scores[scores.prediction_kind == "registered_baryons_kinematic_axis"]
    model_summary = (
        source_scores.groupby("model", as_index=False)
        .agg(median_field_error_ratio=("field_error_ratio", "median"),
             maximum_field_error_ratio=("field_error_ratio", "max"),
             median_gas_weighted_rmse_km_s=("gas_weighted_rmse_km_s", "median"),
             maximum_gas_weighted_rmse_km_s=("gas_weighted_rmse_km_s", "max"),
             galaxies=("galaxy", "nunique"))
    )
    verdicts: list[dict[str, Any]] = []
    for row in model_summary.itertuples(index=False):
        subset = source_scores[source_scores.model == row.model]
        strict = bool((subset.field_error_ratio <= 1.0).all())
        competitive = bool(float(row.median_field_error_ratio) <= 2.0 and float(row.maximum_field_error_ratio) <= 2.0)
        verdicts.append(
            {"model": row.model, "strictFormulaHoldoutSuccess": strict,
             "competitiveButIncomplete": (not strict) and competitive,
             "formulaHoldoutFailure": bool(float(row.maximum_field_error_ratio) > 2.0)}
        )
    model_summary["aggregate_error_band"] = model_summary.median_field_error_ratio.map(error_band)
    model_summary.to_csv(args.output / "holdout_model_summary.csv", index=False)

    gates = config["gates"]
    checks = {
        "requiredSystems": nuisance.galaxy.nunique() == int(gates["requiredSystems"]),
        "requiredTargetArraysOpened": target_arrays_opened == int(gates["requiredTargetArraysOpened"]),
        "requiredValidationArraysOpened": validation_arrays_opened == int(gates["requiredValidationArraysOpened"]),
        "requiredHoldoutArraysOpened": holdout_arrays_opened == int(gates["requiredHoldoutArraysOpened"]),
        "minimumScoredPixelsPerGalaxy": int(nuisance.scored_pixels.min())
        >= int(gates["minimumScoredPixelsPerGalaxy"]),
        "maximumAbsoluteAxisOffsetDeg": float(
            nuisance.kinematic_phase_offset_deg_in_registered_plane.max()
        ) <= float(gates["maximumAbsoluteAxisOffsetDeg"]),
        "maximumTwinSourcePredictionTransportRmseKmS": float(scores.twin_source_transport_rmse_km_s.max())
        <= float(gates["maximumTwinSourcePredictionTransportRmseKmS"]),
        "maximumGravityParametersFitted": int(scores.gravity_parameters_fitted.max())
        <= int(gates["maximumGravityParametersFitted"]),
        "maximumDarkMatterParameters": int(scores.dark_matter_parameters.max())
        <= int(gates["maximumDarkMatterParameters"]),
        "allTargetHashesMatchAcquisitionManifest": hashes_match,
        "allScoresFinite": bool(np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    status = "pass" if all(checks.values()) else "fail"
    report_core = {
        "schemaVersion": config["resultSchemaVersion"], "stage": config["stage"], "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "parents": {"resolvedAcquisition": acquisition["manifestSha256"],
                    "registeredBaryons": registered_report["reportSha256"],
                    "adaptiveTwins": twin_report["reportSha256"],
                    "observationPolicyDevelopment": policy_report["reportSha256"]},
        "systems": len(config["systems"]), "targetArraysOpened": target_arrays_opened,
        "validationArraysOpened": validation_arrays_opened, "holdoutArraysOpened": holdout_arrays_opened,
        "fittedObservationNuisances": 2 * len(config["systems"]), "fittedSpeedAmplitudesUsed": 0,
        "gravityParametersFitted": 0, "darkMatterParameters": 0, "checks": checks,
        "modelSummary": model_summary.to_dict(orient="records"), "formulaVerdicts": verdicts,
        "geometry": nuisance.to_dict(orient="records"), "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines: list[str] = []
    for galaxy in config["systems"]:
        for model in ("newtonian_thin_sheet", "fixed_simple_mond"):
            item = source_scores[(source_scores.galaxy == galaxy) & (source_scores.model == model)].iloc[0]
            lines.append(
                f"- {galaxy} · {model}: {item.gas_weighted_rmse_km_s:.2f} km/s RMSE, "
                f"error ratio {item.field_error_ratio:.2f} ({item.error_band})"
            )
    summary = f"""# P0752 final resolved-galaxy holdout

Protocol status: **{status.upper()}**

{chr(10).join(lines)}

- Holdout target arrays opened: {holdout_arrays_opened}
- Validation arrays opened: 0
- Fitted speed amplitudes used: 0
- Gravity or dark-matter parameters fitted: 0
- Report SHA-256: `{report['reportSha256']}`

The protocol status measures whether the preregistered test executed faithfully. Formula verdicts are reported separately and are never used to rewrite the protocol.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "formulaVerdicts": verdicts,
                      "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
