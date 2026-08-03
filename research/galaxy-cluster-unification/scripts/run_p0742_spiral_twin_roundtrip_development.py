"""Build formula-independent spiral twins and test prediction transport."""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.field_solvers import simple_mond_acceleration  # noqa: E402
from voidscreen.galaxy_maps import resolved_map_morphology  # noqa: E402
from voidscreen.geometric_transport import KPC_M, thin_sheet_newtonian_field  # noqa: E402
from voidscreen.resolved_galaxy_generator import (  # noqa: E402
    extract_galaxy_parameters,
    package_content_hash,
    render_galaxy,
    roundtrip_metrics,
    sample_vertical_realization,
)
from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0742_spiral_twin_roundtrip_development.json"
DEFAULT_OUTPUT = ROOT / "results/p0742_spiral_twin_roundtrip_development"
P0741_RESULT = ROOT / "results/p0741_fused_spiral_baryonic_registration_development"
SPARC_TABLE = ROOT / "data/raw/sparc/table1.dat"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(seed: int, *labels: str) -> int:
    payload = ":".join((str(seed), *labels)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def numeric_parameter_count(value: Any) -> int:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return 0
    if isinstance(value, (int, float)):
        return 1
    if isinstance(value, dict):
        return sum(numeric_parameter_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(numeric_parameter_count(item) for item in value)
    return 0


def aggregate_component(scores: pd.DataFrame, tier: str, component: str) -> dict[str, float]:
    selected = scores[(scores.tier == tier) & (scores.component == component)]
    return {
        "medianNormalizedL2": float(selected.normalized_l2.median()),
        "maximumNormalizedL2": float(selected.normalized_l2.max()),
        "medianPixelCorrelation": float(selected.pixel_correlation.median()),
        "minimumPixelCorrelation": float(selected.pixel_correlation.min()),
        "medianRadialProfileLog10Rmse": float(selected.radial_profile_log10_rmse.median()),
        "maximumMassRelativeError": float(selected.mass_relative_error.max()),
        "maximumConcentrationAbsoluteError": float(selected.concentration_absolute_error.max()),
        "maximumLopsidednessAbsoluteError": float(selected.lopsidedness_absolute_error.max()),
        "maximumClumpinessAbsoluteError": float(selected.clumpiness_absolute_error.max()),
    }


def circular_speed_map(
    surface: np.ndarray,
    axis: np.ndarray,
    *,
    model: str,
    gravitational_constant: float,
    a0: float | None,
    padding_factor: float,
) -> np.ndarray:
    spacing = float(axis[1] - axis[0])
    field = thin_sheet_newtonian_field(
        surface,
        spacing,
        gravitational_constant=gravitational_constant,
        padding_factor=padding_factor,
    )
    xx, yy = np.meshgrid(axis, axis)
    radius_kpc = np.hypot(xx, yy)
    radial_x = np.divide(xx, radius_kpc, out=np.zeros_like(xx), where=radius_kpc > 0.0)
    radial_y = np.divide(yy, radius_kpc, out=np.zeros_like(yy), where=radius_kpc > 0.0)
    inward = -(field.acceleration_x_m_s2 * radial_x + field.acceleration_y_m_s2 * radial_y)
    inward = np.maximum(inward, 0.0)
    if model == "fixed_simple_mond":
        if a0 is None:
            raise ValueError("MOND transport requires fixed a0")
        inward = simple_mond_acceleration(inward, a0)
    elif model != "newtonian_thin_sheet":
        raise ValueError(f"unknown transport model {model}")
    return np.sqrt(radius_kpc * KPC_M * inward) / 1000.0


def radial_curve(speed: np.ndarray, radius: np.ndarray, lower: float, upper: float, bins: int) -> np.ndarray:
    edges = np.linspace(lower, upper, bins + 1)
    result = np.full(bins, np.nan)
    for index in range(bins):
        selected = (
            np.isfinite(speed)
            & (radius >= edges[index])
            & (radius < edges[index + 1] if index < bins - 1 else radius <= edges[index + 1])
        )
        if int(selected.sum()) >= 8:
            result[index] = float(np.median(speed[selected]))
    return result


def transport_metrics(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    axis: np.ndarray,
    *,
    model: dict[str, Any],
    inclination_deg: float,
    beam_kpc: float,
    hi_radius_kpc: float,
    padding_factor: float,
    radial_bins: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    kwargs = {
        "model": model["id"],
        "gravitational_constant": float(model["gravitationalConstantM3KgS2"]),
        "a0": float(model["a0MPerS2"]) if "a0MPerS2" in model else None,
        "padding_factor": padding_factor,
    }
    source_speed = circular_speed_map(reference["total"], axis, **kwargs)
    twin_speed = circular_speed_map(generated["total"], axis, **kwargs)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)
    lower = max(beam_kpc, float(axis[1] - axis[0]))
    upper = min(hi_radius_kpc, float(axis[-1]))
    source_curve = radial_curve(source_speed, radius, lower, upper, radial_bins)
    twin_curve = radial_curve(twin_speed, radius, lower, upper, radial_bins)
    valid_curve = np.isfinite(source_curve) & np.isfinite(twin_curve)
    radial_rmse = float(np.sqrt(np.mean(np.square(twin_curve[valid_curve] - source_curve[valid_curve]))))

    cos_azimuth = np.divide(xx, radius, out=np.zeros_like(xx), where=radius > 0.0)
    projection = math.sin(math.radians(inclination_deg)) * cos_azimuth
    source_los = source_speed * projection
    twin_los = twin_speed * projection
    valid_map = (
        (radius >= lower)
        & (radius <= upper)
        & np.isfinite(source_los)
        & np.isfinite(twin_los)
        & np.isfinite(reference["gas"])
        & (reference["gas"] > 0.0)
    )
    weights = np.where(valid_map, reference["gas"], 0.0)
    los_rmse = float(
        np.sqrt(np.sum(weights * np.square(twin_los - source_los)) / np.sum(weights))
    )
    speed_scale = float(np.sqrt(np.sum(weights * np.square(source_los)) / np.sum(weights)))
    return (
        {
            "radial_speed_transport_rmse_km_s": radial_rmse,
            "los_speed_transport_rmse_km_s": los_rmse,
            "los_speed_transport_normalized_rmse": los_rmse / max(speed_scale, 1.0e-12),
            "radial_bins_scored": int(valid_curve.sum()),
            "map_pixels_scored": int(valid_map.sum()),
        },
        {
            "source_speed": source_speed,
            "twin_speed": twin_speed,
            "source_los": source_los,
            "twin_los": twin_los,
            "radius": radius,
        },
    )


def render_atlas(records: list[dict[str, Any]], output: Path, tier: str) -> None:
    figure, axes = plt.subplots(len(records), 4, figsize=(14.0, 3.1 * len(records)), constrained_layout=True)
    for row, record in enumerate(records):
        reference = record["reference"]
        generated = record["generated"]
        scale = max(float(np.max(reference)), np.finfo(float).tiny)
        residual = (generated - reference) / scale
        transport = record["transport"]
        panels = [
            (np.log10(reference / scale + 1e-5), "observed baryons", "magma", -5.0, 0.0),
            (np.log10(generated / scale + 1e-5), "fake twin baryons", "magma", -5.0, 0.0),
            (
                residual,
                "twin - observed / peak",
                "coolwarm",
                -max(float(np.quantile(np.abs(residual), 0.995)), 1e-4),
                max(float(np.quantile(np.abs(residual), 0.995)), 1e-4),
            ),
            (
                transport["twin_los"] - transport["source_los"],
                "MOND prediction difference (km/s)",
                "coolwarm",
                -max(float(np.quantile(np.abs(transport["twin_los"] - transport["source_los"]), 0.995)), 0.1),
                max(float(np.quantile(np.abs(transport["twin_los"] - transport["source_los"]), 0.995)), 0.1),
            ),
        ]
        for column, (values, title, cmap, vmin, vmax) in enumerate(panels):
            axes[row, column].imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(record["galaxy"])
    figure.suptitle(f"Resolved spiral twins ({tier})")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    parent = read_json(P0741_RESULT / "report.json")
    if parent["reportSha256"] != config["parent"]["resultSha256"]:
        raise ValueError("P0741 parent hash mismatch")
    audit = pd.read_csv(P0741_RESULT / "map_audit.csv").set_index("galaxy")
    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    args.output.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    vertical_rows: list[dict[str, Any]] = []
    atlas_by_tier: dict[str, list[dict[str, Any]]] = {}
    package_by_tier_galaxy: dict[tuple[str, str], dict[str, Any]] = {}
    generated_by_tier_galaxy: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for tier in config["representationTiers"]:
        tier_id = tier["id"]
        parameter_directory = args.output / "tiers" / tier_id / "parameters"
        generated_directory = args.output / "tiers" / tier_id / "generated_maps"
        parameter_directory.mkdir(parents=True, exist_ok=True)
        generated_directory.mkdir(parents=True, exist_ok=True)
        atlas_by_tier[tier_id] = []
        for galaxy in config["systems"]:
            source_path = P0741_RESULT / "maps" / f"{galaxy}.npz"
            expected_hash = next(
                row["sha256"] for row in parent["mapFiles"] if row["galaxy"] == galaxy
            )
            if sha256(source_path) != expected_hash:
                raise ValueError(f"P0741 map hash mismatch for {galaxy}")
            with np.load(source_path) as payload:
                axis = np.asarray(payload["axis_kpc"], dtype=float)
                reference = {
                    "gas": np.asarray(payload["gas"], dtype=float),
                    "stars": np.asarray(payload["stars"], dtype=float),
                    "total": np.asarray(payload["total"], dtype=float),
                }
            row = audit.loc[galaxy]
            meta = metadata.loc[galaxy]
            source_observables = {
                "dataset": "THINGS H I plus SINGS/AllWISE fused stellar maps (P0741)",
                "sourceMapSha256": expected_hash,
                "distanceMpc": float(row.distance_mpc),
                "inclinationDeg": float(row.inclination_deg),
                "positionAngleDeg": float(row.photometric_position_angle_deg),
                "gasMassSolar": float(row.gas_mass_solar),
                "stellarMassSolar": float(row.stellar_mass_solar),
                "stellarMassToLightAssumption": {
                    "band": "3.6 micron anchored with W1 coverage supplement",
                    "solarUnits": 0.5,
                    "status": "fixed_baryonic_conversion_assumption",
                },
            }
            package = extract_galaxy_parameters(
                galaxy,
                axis,
                reference["gas"],
                reference["stars"],
                source_observables=source_observables,
                radial_bins=int(tier["radialBins"]),
                maximum_fourier_mode=int(tier["maximumFourierMode"]),
                residual_feature_count=int(tier["residualFeaturesPerComponent"]),
            )
            if package["contentSha256"] != package_content_hash(package):
                raise RuntimeError(f"package hash replay failed for {tier_id} {galaxy}")
            parameter_path = parameter_directory / f"{galaxy}.json"
            parameter_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
            generated = render_galaxy(package, axis)
            generated_path = generated_directory / f"{galaxy}.npz"
            np.savez_compressed(
                generated_path,
                axis_kpc=axis,
                gas=generated["gas"],
                stars=generated["stars"],
                total=generated["total"],
                parameter_content_sha256=np.asarray(package["contentSha256"]),
            )
            package_by_tier_galaxy[(tier_id, galaxy)] = package
            generated_by_tier_galaxy[(tier_id, galaxy)] = generated
            for component in ("gas", "stars", "total"):
                score_rows.append(
                    {
                        "tier": tier_id,
                        "galaxy": galaxy,
                        "component": component,
                        **roundtrip_metrics(reference[component], generated[component], axis),
                    }
                )
            numeric_parameters = numeric_parameter_count(package["components"])
            catalog_rows.append(
                {
                    "tier": tier_id,
                    "galaxy": galaxy,
                    "cells_per_axis": len(axis),
                    "input_surface_cells": int(2 * axis.size**2),
                    "numeric_representation_values": numeric_parameters,
                    "cell_to_parameter_ratio": int(2 * axis.size**2) / numeric_parameters,
                    "parameter_json_bytes": parameter_path.stat().st_size,
                    "source_npz_bytes": source_path.stat().st_size,
                    "generated_npz_bytes": generated_path.stat().st_size,
                    "parameter_content_sha256": package["contentSha256"],
                    "generated_map_sha256": sha256(generated_path),
                    "gravity_parameter_count": len(package["gravityParameters"]),
                    "observed_velocity_arrays_opened": 0,
                }
            )
            fixed_mond_maps: dict[str, np.ndarray] | None = None
            for model in config["formulaTransportFixtures"]["models"]:
                metrics, maps = transport_metrics(
                    reference,
                    generated,
                    axis,
                    model=model,
                    inclination_deg=float(row.inclination_deg),
                    beam_kpc=float(row.things_beam_kpc),
                    hi_radius_kpc=float(meta.HI_radius_kpc),
                    padding_factor=float(config["formulaTransportFixtures"]["fieldPaddingFactor"]),
                    radial_bins=int(config["formulaTransportFixtures"]["radialScoreBins"]),
                )
                transport_rows.append(
                    {"tier": tier_id, "galaxy": galaxy, "model": model["id"], **metrics}
                )
                if model["id"] == "fixed_simple_mond":
                    fixed_mond_maps = maps
            assert fixed_mond_maps is not None
            atlas_by_tier[tier_id].append(
                {
                    "galaxy": galaxy,
                    "reference": reference["total"],
                    "generated": generated["total"],
                    "transport": fixed_mond_maps,
                }
            )

            for component in ("gas", "stars"):
                morphology = resolved_map_morphology(
                    generated[component], disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
                )
                r80 = float(morphology["r80_kpc"])
                z_limit = max(8.0 * (axis[1] - axis[0]), 0.8 * r80)
                z_axis = np.linspace(-z_limit, z_limit, 33)
                for realization in range(int(config["verticalPriors"]["realizationsPerComponent"])):
                    rng = np.random.default_rng(
                        stable_seed(
                            int(config["verticalPriors"]["seed"]),
                            tier_id,
                            galaxy,
                            component,
                            str(realization),
                        )
                    )
                    density, vertical_metadata = sample_vertical_realization(
                        generated[component],
                        axis,
                        z_axis,
                        r80_kpc=r80,
                        component=component,
                        rng=rng,
                    )
                    dz = float(z_axis[1] - z_axis[0])
                    projected = np.sum(density, axis=2) * dz
                    projection_error = float(
                        np.max(np.abs(projected - generated[component]))
                        / max(float(np.max(generated[component])), np.finfo(float).tiny)
                    )
                    vertical_rows.append(
                        {
                            "tier": tier_id,
                            "galaxy": galaxy,
                            "component": component,
                            "realization": realization,
                            **vertical_metadata,
                            "projection_relative_error": projection_error,
                        }
                    )
            print(f"{tier_id} {galaxy}: package={package['contentSha256'][:12]}")

    scores = pd.DataFrame(score_rows)
    transport = pd.DataFrame(transport_rows)
    catalog = pd.DataFrame(catalog_rows)
    vertical = pd.DataFrame(vertical_rows)
    scores.to_csv(args.output / "roundtrip_scores.csv", index=False)
    transport.to_csv(args.output / "formula_transport_scores.csv", index=False)
    catalog.to_csv(args.output / "parameter_catalog.csv", index=False)
    vertical.to_csv(args.output / "vertical_prior_ensemble.csv", index=False)

    for tier in config["representationTiers"]:
        tier_id = tier["id"]
        render_atlas(
            atlas_by_tier[tier_id],
            args.output / "tiers" / tier_id / "roundtrip_atlas.png",
            tier_id,
        )

    gates = config["gates"]
    tier_reports: list[dict[str, Any]] = []
    selected_tier: str | None = None
    for tier in config["representationTiers"]:
        tier_id = tier["id"]
        aggregate = {
            component: aggregate_component(scores, tier_id, component)
            for component in ("gas", "stars", "total")
        }
        tier_transport = transport[transport.tier == tier_id]
        tier_catalog = catalog[catalog.tier == tier_id]
        tier_vertical = vertical[vertical.tier == tier_id]
        checks = {
            "requiredSystems": tier_catalog.galaxy.nunique() == int(gates["requiredSystems"]),
            "requiredComponentsPerTier": len(scores[scores.tier == tier_id])
            == int(gates["requiredComponentsPerTier"]),
            "maximumMassRelativeError": float(
                scores[scores.tier == tier_id].mass_relative_error.max()
            )
            <= float(gates["maximumMassRelativeError"]),
            "maximum3dProjectionRelativeError": float(tier_vertical.projection_relative_error.max())
            <= float(gates["maximum3dProjectionRelativeError"]),
            "totalMedianNormalizedL2": aggregate["total"]["medianNormalizedL2"]
            <= float(gates["totalMedianNormalizedL2"]),
            "totalMaximumNormalizedL2": aggregate["total"]["maximumNormalizedL2"]
            <= float(gates["totalMaximumNormalizedL2"]),
            "totalMinimumPixelCorrelation": aggregate["total"]["minimumPixelCorrelation"]
            >= float(gates["totalMinimumPixelCorrelation"]),
            "gasMaximumNormalizedL2": aggregate["gas"]["maximumNormalizedL2"]
            <= float(gates["gasMaximumNormalizedL2"]),
            "starsMaximumNormalizedL2": aggregate["stars"]["maximumNormalizedL2"]
            <= float(gates["starsMaximumNormalizedL2"]),
            "minimumCellToNumericParameterRatio": float(tier_catalog.cell_to_parameter_ratio.min())
            >= float(gates["minimumCellToNumericParameterRatio"]),
            "maximumMedianFormulaRadialSpeedTransportRmseKmS": float(
                tier_transport.radial_speed_transport_rmse_km_s.median()
            )
            <= float(gates["maximumMedianFormulaRadialSpeedTransportRmseKmS"]),
            "maximumWorstFormulaRadialSpeedTransportRmseKmS": float(
                tier_transport.radial_speed_transport_rmse_km_s.max()
            )
            <= float(gates["maximumWorstFormulaRadialSpeedTransportRmseKmS"]),
            "maximumMedianFormulaLosSpeedTransportRmseKmS": float(
                tier_transport.los_speed_transport_rmse_km_s.median()
            )
            <= float(gates["maximumMedianFormulaLosSpeedTransportRmseKmS"]),
            "maximumWorstFormulaLosSpeedTransportRmseKmS": float(
                tier_transport.los_speed_transport_rmse_km_s.max()
            )
            <= float(gates["maximumWorstFormulaLosSpeedTransportRmseKmS"]),
            "requiredObservedVelocityArraysOpened": int(
                tier_catalog.observed_velocity_arrays_opened.sum()
            )
            == int(gates["requiredObservedVelocityArraysOpened"]),
            "maximumFittedGravityParameters": int(tier_catalog.gravity_parameter_count.sum())
            <= int(gates["maximumFittedGravityParameters"]),
        }
        passed = all(checks.values())
        if passed and selected_tier is None:
            selected_tier = tier_id
        tier_reports.append(
            {
                "id": tier_id,
                "controls": tier,
                "status": "pass" if passed else "fail",
                "checks": checks,
                "aggregate": aggregate,
                "minimumCellToNumericParameterRatio": float(
                    tier_catalog.cell_to_parameter_ratio.min()
                ),
                "formulaTransport": {
                    "medianRadialSpeedRmseKmS": float(
                        tier_transport.radial_speed_transport_rmse_km_s.median()
                    ),
                    "worstRadialSpeedRmseKmS": float(
                        tier_transport.radial_speed_transport_rmse_km_s.max()
                    ),
                    "medianLosSpeedRmseKmS": float(
                        tier_transport.los_speed_transport_rmse_km_s.median()
                    ),
                    "worstLosSpeedRmseKmS": float(
                        tier_transport.los_speed_transport_rmse_km_s.max()
                    ),
                },
            }
        )

    status = "pass" if selected_tier is not None else "fail"
    if selected_tier is not None:
        render_atlas(
            atlas_by_tier[selected_tier],
            args.output / "selected_spiral_twin_atlas.png",
            selected_tier,
        )
    report_core = {
        "schemaVersion": "sigma-p0742-spiral-twin-roundtrip-development-result/1",
        "stage": "P0742",
        "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "parentResultSha256": parent["reportSha256"],
        "selectionRule": config["selectionRule"],
        "selectedTier": selected_tier,
        "tiers": tier_reports,
        "systems": len(config["systems"]),
        "validationArraysOpened": 0,
        "holdoutArraysOpened": 0,
        "observedVelocityArraysOpened": 0,
        "fittedGravityParameters": 0,
        "formulaFixtureParametersFitted": 0,
        "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected_tier is None:
        selected_summary = "No frozen tier passed every gate."
    else:
        selected = next(item for item in tier_reports if item["id"] == selected_tier)
        selected_summary = (
            f"Selected the lowest passing tier: **{selected_tier}**. Its total-map median/worst "
            f"normalized errors are {selected['aggregate']['total']['medianNormalizedL2']:.3f}/"
            f"{selected['aggregate']['total']['maximumNormalizedL2']:.3f}; fixed-formula radial "
            f"transport median/worst RMSE are {selected['formulaTransport']['medianRadialSpeedRmseKmS']:.2f}/"
            f"{selected['formulaTransport']['worstRadialSpeedRmseKmS']:.2f} km/s."
        )
    summary = f"""# P0742 spiral fake-twin round trip

Status: **{status.upper()}**

{selected_summary}

- Real development baryonic maps: 4
- Frozen representation tiers tested: 3
- Fixed formula transport fixtures: Newtonian thin sheet and simple MOND
- Observed velocity arrays opened: 0
- Validation arrays opened: 0
- Holdout arrays opened: 0
- Fitted gravity parameters: 0
- Report SHA-256: `{report['reportSha256']}`

This stage tests whether a formula predicts nearly the same speeds on a real baryonic map and its fake twin. It does **not** yet test either prediction against observed speeds.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "selectedTier": selected_tier, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
