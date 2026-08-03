"""Reveal development THINGS velocity fields and score real/fake predictions."""

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
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_p0739_spiral_baryonic_registration as registration  # noqa: E402
from run_p0742_spiral_twin_roundtrip_development import (  # noqa: E402
    circular_speed_map,
    canonical_sha256,
    sha256,
)
from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0744_development_velocity_field_reveal.json"
DEFAULT_OUTPUT = ROOT / "results/p0744_development_velocity_field_reveal"
P0738_RAW = ROOT / "data/raw/p0738_things_sings_resolved"
P0738_RESULT = ROOT / "results/p0738_morphology_diverse_resolved_acquisition/manifest.json"
P0741_RESULT = ROOT / "results/p0741_fused_spiral_baryonic_registration_development"
P0743_RESULT = ROOT / "results/p0743_multiscale_spiral_twin_development"
SPARC_TABLE = ROOT / "data/raw/sparc/table1.dat"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(valid.sum()) < 8:
        raise ValueError("weighted median needs at least eight valid samples")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(values[order[min(index, len(order) - 1)]])


def velocity_unit_scale(header: fits.Header) -> float:
    unit = str(header.get("BUNIT", "")).strip().upper().replace(" ", "")
    if unit in {"METR/SEC", "M/S", "M*S-1", "M.S-1"}:
        return 0.001
    if "KM" in unit and ("SEC" in unit or "/S" in unit or "S-1" in unit):
        return 1.0
    raise ValueError(f"unsupported velocity unit {unit!r}")


def beam_convolve_velocity(
    velocity_km_s: np.ndarray,
    gas_weight: np.ndarray,
    beam_kpc: float,
    spacing_kpc: float,
) -> np.ndarray:
    sigma = beam_kpc / 2.355 / spacing_kpc
    numerator = gaussian_filter(velocity_km_s * gas_weight, sigma=sigma, mode="constant")
    denominator = gaussian_filter(gas_weight, sigma=sigma, mode="constant")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > np.max(denominator) * 1.0e-12,
    )


def field_scores(
    predicted: np.ndarray,
    observed: np.ndarray,
    uncertainty: np.ndarray,
    gas_weight: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = (
        mask
        & np.isfinite(predicted)
        & np.isfinite(observed)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
        & np.isfinite(gas_weight)
        & (gas_weight > 0.0)
    )
    gas = np.where(valid, gas_weight, 0.0)
    safe_uncertainty = np.where(valid, uncertainty, 1.0)
    inverse_variance = np.where(
        valid,
        gas_weight / np.square(safe_uncertainty),
        0.0,
    )
    # NaNs outside the score mask must not enter weighted sums: IEEE arithmetic
    # defines 0 * NaN as NaN, rather than zero.
    residual = np.where(valid, predicted - observed, 0.0)
    gas_rmse = float(np.sqrt(np.sum(gas * np.square(residual)) / np.sum(gas)))
    uncertainty_rms = float(
        np.sqrt(np.sum(gas * np.square(safe_uncertainty)) / np.sum(gas))
    )
    inverse_variance_rmse = float(
        np.sqrt(np.sum(inverse_variance * np.square(residual)) / np.sum(inverse_variance))
    )
    bias = float(np.sum(gas * residual) / np.sum(gas))
    ratio = gas_rmse / uncertainty_rms
    return {
        "scored_pixels": int(valid.sum()),
        "gas_weighted_rmse_km_s": gas_rmse,
        "inverse_variance_weighted_rmse_km_s": inverse_variance_rmse,
        "gas_weighted_bias_km_s": bias,
        "gas_weighted_uncertainty_rms_km_s": uncertainty_rms,
        "field_error_ratio": ratio,
        "standardized_mean_square": float(
            np.sum(gas * np.square(residual / safe_uncertainty)) / np.sum(gas)
        ),
    }


def error_band(ratio: float) -> str:
    if ratio <= 1.0:
        return "consistent"
    if ratio <= 2.0:
        return "close"
    return "miss"


def radial_points(
    observed: np.ndarray,
    source_speed: np.ndarray,
    twin_speed: np.ndarray,
    gas_weight: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    axis: np.ndarray,
    inclination_deg: float,
    handedness: int,
    beam_kpc: float,
    hi_radius_kpc: float,
    bins: int = 20,
) -> list[dict[str, float]]:
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)
    cos_azimuth = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0.0)
    projection = handedness * math.sin(math.radians(inclination_deg)) * cos_azimuth
    valid = mask & (np.abs(cos_azimuth) >= 0.5) & (np.abs(projection) > 0.0)
    observed_rotation = np.divide(
        observed,
        projection,
        out=np.full_like(observed, np.nan),
        where=np.abs(projection) > 0.0,
    )
    observed_rotation = np.where(observed_rotation > 0.0, observed_rotation, np.nan)
    edges = np.linspace(beam_kpc, min(hi_radius_kpc, float(axis[-1])), bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        selected = valid & (radius >= edges[index]) & (radius < edges[index + 1])
        if int(selected.sum()) < 8:
            continue
        weights = gas_weight[selected] / np.square(uncertainty[selected])
        rows.append(
            {
                "radius_kpc": float(np.sum(weights * radius[selected]) / np.sum(weights)),
                "observed_rotation_km_s": weighted_median(observed_rotation[selected], weights),
                "source_prediction_km_s": float(np.sum(weights * source_speed[selected]) / np.sum(weights)),
                "twin_prediction_km_s": float(np.sum(weights * twin_speed[selected]) / np.sum(weights)),
                "pixels": int(selected.sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    acquisition = read_json(P0738_RESULT)
    p0741 = read_json(P0741_RESULT / "report.json")
    p0743 = read_json(P0743_RESULT / "report.json")
    if acquisition["manifestSha256"] != config["parents"]["resolvedAcquisitionManifestSha256"]:
        raise ValueError("P0738 parent hash mismatch")
    if p0741["reportSha256"] != config["parents"]["registeredBaryonsResultSha256"]:
        raise ValueError("P0741 parent hash mismatch")
    if p0743["reportSha256"] != config["parents"]["fakeTwinsResultSha256"]:
        raise ValueError("P0743 parent hash mismatch")
    if p0743["selectedTier"] != config["parents"]["selectedTwinTier"]:
        raise ValueError("selected twin tier mismatch")
    expected_hashes = {
        (row["galaxy"], row["kind"]): row["sha256"] for row in acquisition["files"]
    }
    audit = pd.read_csv(P0741_RESULT / "map_audit.csv").set_index("galaxy")
    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    coordinates = registration.load_coordinates()
    args.output.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []
    target_arrays_opened = 0
    hashes_match = True
    for galaxy in config["systems"]:
        source_path = P0741_RESULT / "maps" / f"{galaxy}.npz"
        source_hash = next(row["sha256"] for row in p0741["mapFiles"] if row["galaxy"] == galaxy)
        twin_path = (
            P0743_RESULT
            / "tiers"
            / config["parents"]["selectedTwinTier"]
            / "generated_maps"
            / f"{galaxy}.npz"
        )
        selected_catalog = pd.read_csv(P0743_RESULT / "parameter_catalog.csv")
        expected_twin_hash = selected_catalog[
            (selected_catalog.tier == config["parents"]["selectedTwinTier"])
            & (selected_catalog.galaxy == galaxy)
        ].iloc[0].generated_map_sha256
        if sha256(source_path) != source_hash or sha256(twin_path) != expected_twin_hash:
            raise ValueError(f"baryonic map hash mismatch for {galaxy}")
        with np.load(source_path) as payload:
            axis = np.asarray(payload["axis_kpc"], dtype=float)
            source = {key: np.asarray(payload[key], dtype=float) for key in ("gas", "stars", "total")}
        with np.load(twin_path) as payload:
            twin = {key: np.asarray(payload[key], dtype=float) for key in ("gas", "stars", "total")}

        raw = P0738_RAW / galaxy
        moment1_path = next(raw.glob("*MOM1_THINGS.FITS"))
        moment2_path = next(raw.glob("*MOM2_THINGS.FITS"))
        for kind, path in (("moment1", moment1_path), ("moment2", moment2_path)):
            if sha256(path) != expected_hashes[(galaxy, kind)]:
                hashes_match = False
                raise ValueError(f"target hash mismatch for {galaxy} {kind}")
        with fits.open(moment1_path, memmap=True) as hdus:
            moment1 = np.asarray(hdus[0].data, dtype=float).squeeze()
            moment1_header = hdus[0].header.copy()
        with fits.open(moment2_path, memmap=True) as hdus:
            moment2 = np.asarray(hdus[0].data, dtype=float).squeeze()
            moment2_header = hdus[0].header.copy()
        target_arrays_opened += 2
        moment1 *= velocity_unit_scale(moment1_header)
        moment2 *= velocity_unit_scale(moment2_header)
        wcs1 = registration.celestial_wcs(moment1_header)
        wcs2 = registration.celestial_wcs(moment2_header)
        row = audit.loc[galaxy]
        meta = metadata.loc[galaxy]
        center = coordinates[galaxy]
        x1, y1 = registration.plane_world_pixels(
            axis,
            center,
            float(row.photometric_position_angle_deg),
            float(row.inclination_deg),
            float(row.distance_mpc),
            wcs1,
        )
        x2, y2 = registration.plane_world_pixels(
            axis,
            center,
            float(row.photometric_position_angle_deg),
            float(row.inclination_deg),
            float(row.distance_mpc),
            wcs2,
        )
        observed_absolute = registration.sample_plane(moment1, x1, y1)
        dispersion = registration.sample_plane(moment2, x2, y2)
        xx, yy = np.meshgrid(axis, axis)
        radius = np.hypot(xx, yy)
        core = (
            np.isfinite(observed_absolute)
            & np.isfinite(source["gas"])
            & (source["gas"] > 0.0)
            & (radius <= 0.5 * float(meta.effective_radius_kpc))
        )
        systemic = weighted_median(observed_absolute[core], source["gas"][core])
        observed = observed_absolute - systemic
        cos_azimuth = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0.0)
        convention_mask = (
            np.isfinite(observed)
            & (source["gas"] > 0.0)
            & (radius >= float(row.things_beam_kpc))
            & (radius <= float(meta.HI_radius_kpc))
        )
        covariance = float(np.sum(source["gas"][convention_mask] * observed[convention_mask] * cos_azimuth[convention_mask]))
        handedness = 1 if covariance >= 0.0 else -1
        uncertainty = np.sqrt(
            np.square(np.where(np.isfinite(dispersion), np.maximum(dispersion, 0.0), np.nan))
            + (5.2 / 2.355) ** 2
        )
        score_mask = (
            np.isfinite(observed)
            & np.isfinite(dispersion)
            & (dispersion >= 0.0)
            & (source["gas"] > 0.0)
            & (radius >= float(row.things_beam_kpc))
            & (radius <= float(meta.HI_radius_kpc))
        )
        projection = handedness * math.sin(math.radians(float(row.inclination_deg))) * cos_azimuth
        per_galaxy_models: dict[str, dict[str, np.ndarray]] = {}
        for model in config["models"]:
            kwargs = {
                "model": model["id"],
                "gravitational_constant": float(model["gravitationalConstantM3KgS2"]),
                "a0": float(model["a0MPerS2"]) if "a0MPerS2" in model else None,
                "padding_factor": float(config["fieldSettings"]["paddingFactor"]),
            }
            source_speed = circular_speed_map(source["total"], axis, **kwargs)
            twin_speed = circular_speed_map(twin["total"], axis, **kwargs)
            source_los = beam_convolve_velocity(
                source_speed * projection,
                source["gas"],
                float(row.things_beam_kpc),
                float(row.spacing_kpc),
            )
            twin_los = beam_convolve_velocity(
                twin_speed * projection,
                source["gas"],
                float(row.things_beam_kpc),
                float(row.spacing_kpc),
            )
            transport_score = field_scores(twin_los, source_los, uncertainty, source["gas"], score_mask)
            for map_kind, prediction in (("registered_baryons", source_los), ("fake_twin", twin_los)):
                metrics = field_scores(prediction, observed, uncertainty, source["gas"], score_mask)
                score_rows.append(
                    {
                        "galaxy": galaxy,
                        "model": model["id"],
                        "map_kind": map_kind,
                        **metrics,
                        "error_band": error_band(metrics["field_error_ratio"]),
                        "twin_source_transport_rmse_km_s": transport_score["gas_weighted_rmse_km_s"],
                        "gravity_parameters_fitted": 0,
                        "dark_matter_parameters": 0,
                    }
                )
            for point in radial_points(
                observed,
                source_speed,
                twin_speed,
                source["gas"],
                uncertainty,
                score_mask,
                axis,
                float(row.inclination_deg),
                handedness,
                float(row.things_beam_kpc),
                float(meta.HI_radius_kpc),
            ):
                radial_rows.append({"galaxy": galaxy, "model": model["id"], **point})
            per_galaxy_models[model["id"]] = {
                "source_los": source_los,
                "twin_los": twin_los,
            }
        nuisance_rows.append(
            {
                "galaxy": galaxy,
                "systemic_velocity_km_s": systemic,
                "handedness": handedness,
                "inclination_deg": float(row.inclination_deg),
                "position_angle_deg": float(row.photometric_position_angle_deg),
                "scored_pixels": int(score_mask.sum()),
                "median_dispersion_km_s": float(np.nanmedian(dispersion[score_mask])),
                "moment1_sha256": sha256(moment1_path),
                "moment2_sha256": sha256(moment2_path),
                "gravity_parameters_fitted": 0,
            }
        )
        atlas_rows.append(
            {
                "galaxy": galaxy,
                "observed": observed,
                "source": per_galaxy_models["fixed_simple_mond"]["source_los"],
                "twin": per_galaxy_models["fixed_simple_mond"]["twin_los"],
                "mask": score_mask,
            }
        )
        print(f"{galaxy}: pixels={score_mask.sum()}, systemic={systemic:.2f}, handedness={handedness:+d}")

    scores = pd.DataFrame(score_rows)
    radial = pd.DataFrame(radial_rows)
    nuisance = pd.DataFrame(nuisance_rows)
    scores.to_csv(args.output / "velocity_field_scores.csv", index=False)
    radial.to_csv(args.output / "radial_curve_points.csv", index=False)
    nuisance.to_csv(args.output / "observation_nuisance_audit.csv", index=False)

    figure, axes = plt.subplots(len(atlas_rows), 4, figsize=(14, 3.1 * len(atlas_rows)), constrained_layout=True)
    for row_index, item in enumerate(atlas_rows):
        finite = item["mask"] & np.isfinite(item["observed"])
        limit = float(np.quantile(np.abs(item["observed"][finite]), 0.99))
        residual = item["twin"] - item["observed"]
        residual_limit = max(float(np.quantile(np.abs(residual[finite]), 0.99)), 1.0)
        panels = [
            (item["observed"], "observed H I velocity", -limit, limit),
            (item["source"], "MOND on registered baryons", -limit, limit),
            (item["twin"], "MOND on fake twin", -limit, limit),
            (residual, "fake twin - observed", -residual_limit, residual_limit),
        ]
        for column, (values, title, vmin, vmax) in enumerate(panels):
            shown = np.where(item["mask"], values, np.nan)
            image = axes[row_index, column].imshow(
                shown,
                origin="lower",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            figure.colorbar(image, ax=axes[row_index, column], fraction=0.046, label="km/s")
        axes[row_index, 0].set_ylabel(item["galaxy"])
    figure.suptitle("P0744 real velocity fields versus fixed MOND predictions")
    figure.savefig(args.output / "velocity_field_comparison_atlas.png", dpi=170)
    plt.close(figure)

    source_scores = scores[scores.map_kind == "registered_baryons"]
    model_summary = (
        source_scores.groupby("model", as_index=False)
        .agg(
            median_field_error_ratio=("field_error_ratio", "median"),
            maximum_field_error_ratio=("field_error_ratio", "max"),
            median_gas_weighted_rmse_km_s=("gas_weighted_rmse_km_s", "median"),
            maximum_gas_weighted_rmse_km_s=("gas_weighted_rmse_km_s", "max"),
            galaxies=("galaxy", "nunique"),
        )
    )
    model_summary["aggregate_error_band"] = model_summary.median_field_error_ratio.map(error_band)
    model_summary.to_csv(args.output / "model_summary.csv", index=False)

    gates = config["gates"]
    checks = {
        "requiredSystems": nuisance.galaxy.nunique() == int(gates["requiredSystems"]),
        "requiredTargetArraysOpened": target_arrays_opened == int(gates["requiredTargetArraysOpened"]),
        "requiredValidationArraysOpened": 0 == int(gates["requiredValidationArraysOpened"]),
        "requiredHoldoutArraysOpened": 0 == int(gates["requiredHoldoutArraysOpened"]),
        "minimumScoredPixelsPerGalaxy": int(nuisance.scored_pixels.min())
        >= int(gates["minimumScoredPixelsPerGalaxy"]),
        "maximumTwinSourcePredictionTransportRmseKmS": float(
            scores.twin_source_transport_rmse_km_s.max()
        )
        <= float(gates["maximumTwinSourcePredictionTransportRmseKmS"]),
        "maximumGravityParametersFitted": int(scores.gravity_parameters_fitted.max())
        <= int(gates["maximumGravityParametersFitted"]),
        "maximumDarkMatterParameters": int(scores.dark_matter_parameters.max())
        <= int(gates["maximumDarkMatterParameters"]),
        "allTargetHashesMatchAcquisitionManifest": hashes_match,
        "allScoresFinite": bool(
            np.isfinite(
                scores[
                    [
                        "gas_weighted_rmse_km_s",
                        "inverse_variance_weighted_rmse_km_s",
                        "field_error_ratio",
                        "twin_source_transport_rmse_km_s",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    report_core = {
        "schemaVersion": "sigma-p0744-development-velocity-field-reveal-result/1",
        "stage": "P0744",
        "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "parents": {
            "p0738": acquisition["manifestSha256"],
            "p0741": p0741["reportSha256"],
            "p0743": p0743["reportSha256"],
        },
        "systems": len(config["systems"]),
        "targetArraysOpened": target_arrays_opened,
        "validationArraysOpened": 0,
        "holdoutArraysOpened": 0,
        "gravityParametersFitted": 0,
        "darkMatterParameters": 0,
        "checks": checks,
        "modelSummary": model_summary.to_dict(orient="records"),
        "maximumTwinSourcePredictionTransportRmseKmS": float(
            scores.twin_source_transport_rmse_km_s.max()
        ),
        "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = []
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"- {row.model}: median/worst RMSE {row.median_gas_weighted_rmse_km_s:.2f}/"
            f"{row.maximum_gas_weighted_rmse_km_s:.2f} km/s; median error ratio "
            f"{row.median_field_error_ratio:.2f} ({row.aggregate_error_band})"
        )
    summary = f"""# P0744 real development velocity-field reveal

Protocol/execution status: **{status.upper()}**

This status means the frozen comparison ran with finite, leakage-audited scores;
it is not a claim that either gravity formula passed every galaxy.

{chr(10).join(lines)}

- Real galaxies: {len(config['systems'])}
- Scored velocity pixels: {int(nuisance.scored_pixels.sum()):,}
- Maximum fake-twin/source prediction transport RMSE: {scores.twin_source_transport_rmse_km_s.max():.2f} km/s
- Validation arrays opened: 0
- Holdout arrays opened: 0
- Gravity parameters fitted: 0
- Dark-matter parameters: 0
- Report SHA-256: `{report['reportSha256']}`

This is a raw circular-equilibrium development comparison. It does not fit pressure support, bars, warps, streaming motions, M/L, distance, inclination, MOND parameters, or dark halos.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
