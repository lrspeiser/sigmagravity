"""Develop one kinematic-axis policy across all open adaptive twins."""

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
)
from run_p0747_post_reveal_kinematic_axis_diagnostic import fit_unit_phase  # noqa: E402
from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0750_adaptive_twin_kinematic_axis_development.json"
DEFAULT_OUTPUT = ROOT / "results/p0750_adaptive_twin_kinematic_axis_development"
P0738_RAW = ROOT / "data/raw/p0738_things_sings_resolved"
SPARC_TABLE = ROOT / "data/raw/sparc/table1.dat"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)

    acquisition_path = ROOT / config["parents"]["resolvedAcquisition"]
    acquisition = read_json(acquisition_path)
    if acquisition["manifestSha256"] != config["parents"]["resolvedAcquisitionManifestSha256"]:
        raise ValueError("resolved acquisition manifest hash mismatch")
    expected_hashes = {(row["galaxy"], row["kind"]): row["sha256"] for row in acquisition["files"]}
    twin_result = ROOT / config["parents"]["adaptiveTwinsResultPath"]
    twin_report = read_json(twin_result / "report.json")
    if twin_report["reportSha256"] != config["parents"]["adaptiveTwinsResultSha256"]:
        raise ValueError("adaptive-twin result hash mismatch")
    twin_catalog = pd.read_csv(twin_result / "selected_parameter_catalog.csv").set_index("galaxy")

    group_by_galaxy: dict[str, dict[str, Any]] = {}
    group_state: dict[str, dict[str, Any]] = {}
    for group in config["parents"]["groups"]:
        registered_dir = ROOT / group["registeredBaryonsResultPath"]
        reveal_dir = ROOT / group["velocityRevealResultPath"]
        registered_report = read_json(registered_dir / "report.json")
        reveal_report = read_json(reveal_dir / "report.json")
        if registered_report["reportSha256"] != group["registeredBaryonsResultSha256"]:
            raise ValueError(f"{group['id']} registered-baryon parent hash mismatch")
        if reveal_report["reportSha256"] != group["velocityRevealResultSha256"]:
            raise ValueError(f"{group['id']} reveal parent hash mismatch")
        state = {
            "registeredDir": registered_dir,
            "registeredReport": registered_report,
            "audit": pd.read_csv(registered_dir / "map_audit.csv").set_index("galaxy"),
            "nuisance": pd.read_csv(reveal_dir / "observation_nuisance_audit.csv").set_index("galaxy"),
        }
        group_state[group["id"]] = state
        for galaxy in group["systems"]:
            if galaxy in group_by_galaxy:
                raise ValueError(f"duplicate galaxy in observation groups: {galaxy}")
            group_by_galaxy[galaxy] = {"config": group, "state": state}
    if set(group_by_galaxy) != set(config["systems"]):
        raise ValueError("observation groups do not exactly cover configured systems")

    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    coordinates = registration.load_coordinates()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []
    target_arrays_opened = 0

    for galaxy in config["systems"]:
        group = group_by_galaxy[galaxy]["config"]
        state = group_by_galaxy[galaxy]["state"]
        row = state["audit"].loc[galaxy]
        nuisance_row = state["nuisance"].loc[galaxy]
        meta = metadata.loc[galaxy]
        source_path = state["registeredDir"] / "maps" / f"{galaxy}.npz"
        source_hash = next(
            item["sha256"] for item in state["registeredReport"]["mapFiles"] if item["galaxy"] == galaxy
        )
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
                raise ValueError(f"target hash mismatch for {galaxy} {kind}")
        with fits.open(moment1_path, memmap=True) as hdus:
            moment1 = np.asarray(hdus[0].data, dtype=float).squeeze()
            header1 = hdus[0].header.copy()
        with fits.open(moment2_path, memmap=True) as hdus:
            moment2 = np.asarray(hdus[0].data, dtype=float).squeeze()
            header2 = hdus[0].header.copy()
        target_arrays_opened += 2
        moment1 *= velocity_unit_scale(header1)
        moment2 *= velocity_unit_scale(header2)
        center = coordinates[galaxy]
        x1, y1 = registration.plane_world_pixels(
            axis, center, float(row.photometric_position_angle_deg), float(row.inclination_deg),
            float(row.distance_mpc), registration.celestial_wcs(header1)
        )
        x2, y2 = registration.plane_world_pixels(
            axis, center, float(row.photometric_position_angle_deg), float(row.inclination_deg),
            float(row.distance_mpc), registration.celestial_wcs(header2)
        )
        observed = registration.sample_plane(moment1, x1, y1) - float(nuisance_row.systemic_velocity_km_s)
        dispersion = registration.sample_plane(moment2, x2, y2)
        uncertainty = np.sqrt(
            np.square(np.where(np.isfinite(dispersion), np.maximum(dispersion, 0.0), np.nan))
            + (5.2 / 2.355) ** 2
        )
        xx, yy = np.meshgrid(axis, axis)
        radius = np.hypot(xx, yy)
        cos_azimuth = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0.0)
        sin_azimuth = np.divide(yy, radius, out=np.zeros_like(yy), where=radius > 0.0)
        score_mask = (
            np.isfinite(observed) & np.isfinite(dispersion) & (dispersion >= 0.0)
            & (source["gas"] > 0.0) & (radius >= float(row.things_beam_kpc))
            & (radius <= float(meta.HI_radius_kpc))
        )
        unit, fitted_amplitude, explained_fraction = fit_unit_phase(
            observed, cos_azimuth, sin_azimuth, source["gas"], uncertainty, score_mask
        )
        old_unit = np.asarray([int(nuisance_row.handedness), 0.0], dtype=float)
        axis_offset_deg = float(np.degrees(np.arccos(np.clip(np.dot(old_unit, unit), -1.0, 1.0))))
        projection_scale = math.sin(math.radians(float(row.inclination_deg)))
        old_projection = projection_scale * old_unit[0] * cos_azimuth
        new_projection = projection_scale * (unit[0] * cos_azimuth + unit[1] * sin_azimuth)
        geometry_rows.append(
            {"galaxy": galaxy, "split": group["id"], "scored_pixels": int(score_mask.sum()),
             "image_position_angle_deg": float(row.photometric_position_angle_deg),
             "kinematic_phase_offset_deg_in_registered_plane": axis_offset_deg,
             "kinematic_unit_x": float(unit[0]), "kinematic_unit_y": float(unit[1]),
             "fitted_first_harmonic_amplitude_km_s": fitted_amplitude,
             "first_harmonic_explained_variance_fraction": explained_fraction,
             "selected_twin_coefficients_per_component": int(twin_catalog.loc[galaxy].coefficients_per_component),
             "fitted_observation_nuisances": 1, "fitted_gravity_parameters": 0}
        )

        atlas_models: dict[str, np.ndarray] = {}
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
                    source_speed * old_projection, source["gas"], float(row.things_beam_kpc),
                    float(row.spacing_kpc)
                ),
                "registered_baryons_kinematic_axis": beam_convolve_velocity(
                    source_speed * new_projection, source["gas"], float(row.things_beam_kpc),
                    float(row.spacing_kpc)
                ),
                "adaptive_twin_kinematic_axis": beam_convolve_velocity(
                    twin_speed * new_projection, source["gas"], float(row.things_beam_kpc),
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
                    {"galaxy": galaxy, "split": group["id"], "model": model["id"],
                     "prediction_kind": kind, **metrics, "error_band": error_band(metrics["field_error_ratio"]),
                     "kinematic_twin_source_transport_rmse_km_s": transport,
                     "fitted_observation_nuisances": 1 if "kinematic_axis" in kind else 0,
                     "fitted_gravity_parameters": 0}
                )
            if model["id"] == "fixed_simple_mond":
                atlas_models = predictions
        atlas_rows.append({"galaxy": galaxy, "observed": observed, "mask": score_mask, **atlas_models})
        print(f"{galaxy}: offset={axis_offset_deg:.2f} deg; harmonic fraction={explained_fraction:.3f}")

    geometry = pd.DataFrame(geometry_rows)
    scores = pd.DataFrame(score_rows)
    geometry.to_csv(args.output / "kinematic_axis_policy_audit.csv", index=False)
    scores.to_csv(args.output / "adaptive_twin_velocity_field_scores.csv", index=False)
    fig, axes = plt.subplots(len(atlas_rows), 5, figsize=(17, 3.1 * len(atlas_rows)), constrained_layout=True)
    for row_index, item in enumerate(atlas_rows):
        finite = item["mask"] & np.isfinite(item["observed"])
        limit = float(np.quantile(np.abs(item["observed"][finite]), 0.99))
        old = item["registered_baryons_photometric_axis"]
        new = item["registered_baryons_kinematic_axis"]
        twin = item["adaptive_twin_kinematic_axis"]
        residual_limit = max(float(np.quantile(np.abs((new - item["observed"])[finite]), 0.99)), 1.0)
        panels = [
            (item["observed"], "observed", -limit, limit),
            (old, "MOND · image axis", -limit, limit),
            (new, "MOND · kinematic axis", -limit, limit),
            (twin, "MOND adaptive twin", -limit, limit),
            (new - item["observed"], "source residual", -residual_limit, residual_limit),
        ]
        for column, (values, title, vmin, vmax) in enumerate(panels):
            shown = np.where(item["mask"], values, np.nan)
            image = axes[row_index, column].imshow(
                shown, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
            )
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, label="km/s")
        axes[row_index, 0].set_ylabel(item["galaxy"])
    fig.suptitle("P0750 universal kinematic-axis policy on adaptive twins")
    fig.savefig(args.output / "adaptive_twin_kinematic_axis_atlas.png", dpi=170)
    plt.close(fig)

    gates = config["gates"]
    checks = {
        "requiredSystems": geometry.galaxy.nunique() == int(gates["requiredSystems"]),
        "requiredTargetArraysOpened": target_arrays_opened == int(gates["requiredTargetArraysOpened"]),
        "requiredHoldoutArraysOpened": 0 == int(gates["requiredHoldoutArraysOpened"]),
        "minimumScoredPixelsPerGalaxy": int(geometry.scored_pixels.min())
        >= int(gates["minimumScoredPixelsPerGalaxy"]),
        "maximumAbsoluteAxisOffsetDeg": float(
            geometry.kinematic_phase_offset_deg_in_registered_plane.max()
        ) <= float(gates["maximumAbsoluteAxisOffsetDeg"]),
        "requiredFiniteScores": bool(np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy()).all()),
        "maximumGravityParametersFitted": int(scores.fitted_gravity_parameters.max())
        <= int(gates["maximumGravityParametersFitted"]),
    }
    status = "pass" if all(checks.values()) else "fail"
    model_summary = (
        scores[scores.prediction_kind == "registered_baryons_kinematic_axis"]
        .groupby("model", as_index=False)
        .agg(median_rmse_km_s=("gas_weighted_rmse_km_s", "median"),
             maximum_rmse_km_s=("gas_weighted_rmse_km_s", "max"),
             median_field_error_ratio=("field_error_ratio", "median"),
             maximum_field_error_ratio=("field_error_ratio", "max"))
        .to_dict(orient="records")
    )
    report_core = {
        "schemaVersion": config["resultSchemaVersion"], "stage": config["stage"], "status": status,
        "developmentDisclosure": config["developmentDisclosure"],
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "adaptiveTwinsResultSha256": twin_report["reportSha256"], "systems": len(config["systems"]),
        "targetArraysOpened": target_arrays_opened, "holdoutArraysOpened": 0,
        "fittedObservationNuisances": len(config["systems"]), "fittedGravityParameters": 0,
        "checks": checks, "modelSummary": model_summary,
        "geometry": geometry.to_dict(orient="records"), "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines: list[str] = []
    for galaxy in config["systems"]:
        angle = float(geometry[geometry.galaxy == galaxy].iloc[0].kinematic_phase_offset_deg_in_registered_plane)
        mond = scores[(scores.galaxy == galaxy) & (scores.model == "fixed_simple_mond")]
        old = mond[mond.prediction_kind == "registered_baryons_photometric_axis"].iloc[0]
        new = mond[mond.prediction_kind == "registered_baryons_kinematic_axis"].iloc[0]
        twin = mond[mond.prediction_kind == "adaptive_twin_kinematic_axis"].iloc[0]
        lines.append(
            f"- {galaxy}: offset {angle:.2f}°; fixed-MOND source RMSE "
            f"{old.gas_weighted_rmse_km_s:.2f} → {new.gas_weighted_rmse_km_s:.2f} km/s; "
            f"adaptive twin {twin.gas_weighted_rmse_km_s:.2f} km/s"
        )
    summary = f"""# P0750 adaptive-twin kinematic-axis development

Status: **{status.upper()}**

{chr(10).join(lines)}

- Formula-independent observation nuisances fitted: {len(config['systems'])}
- Velocity amplitudes used in predictions: 0
- Gravity parameters fitted: 0
- Holdout arrays opened: 0
- Report SHA-256: `{report['reportSha256']}`

This is post-reveal observation-policy development, not validation or holdout evidence.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
