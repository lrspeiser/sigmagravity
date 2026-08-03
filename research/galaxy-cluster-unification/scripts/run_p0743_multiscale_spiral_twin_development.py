"""Run the frozen multiscale rescue of the P0742 spiral twins."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0742_spiral_twin_roundtrip_development import (  # noqa: E402
    aggregate_component,
    canonical_sha256,
    numeric_parameter_count,
    render_atlas,
    sha256,
    stable_seed,
    transport_metrics,
)
from voidscreen.galaxy_maps import resolved_map_morphology  # noqa: E402
from voidscreen.multiscale_galaxy_generator import (  # noqa: E402
    extract_galaxy_parameters,
    package_content_hash,
    render_galaxy,
)
from voidscreen.resolved_galaxy_generator import roundtrip_metrics, sample_vertical_realization  # noqa: E402
from voidscreen.sparc_morphology import parse_sparc_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/p0743_multiscale_spiral_twin_development.json"
DEFAULT_OUTPUT = ROOT / "results/p0743_multiscale_spiral_twin_development"
P0741_RESULT = ROOT / "results/p0741_fused_spiral_baryonic_registration_development"
P0742_RESULT = ROOT / "results/p0742_spiral_twin_roundtrip_development"
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
    p0741 = read_json(P0741_RESULT / "report.json")
    p0742 = read_json(P0742_RESULT / "report.json")
    if p0741["reportSha256"] != config["parents"]["baryonicMapsResultSha256"]:
        raise ValueError("P0741 parent hash mismatch")
    if p0742["reportSha256"] != config["parents"]["failedRadialTwinResultSha256"]:
        raise ValueError("P0742 parent hash mismatch")
    audit = pd.read_csv(P0741_RESULT / "map_audit.csv").set_index("galaxy")
    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    args.output.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    vertical_rows: list[dict[str, Any]] = []
    atlas_by_tier: dict[str, list[dict[str, Any]]] = {}

    for tier in config["representation"]["tiers"]:
        tier_id = tier["id"]
        coefficient_count = int(tier["coefficientsPerComponent"])
        parameter_directory = args.output / "tiers" / tier_id / "parameters"
        generated_directory = args.output / "tiers" / tier_id / "generated_maps"
        parameter_directory.mkdir(parents=True, exist_ok=True)
        generated_directory.mkdir(parents=True, exist_ok=True)
        atlas_by_tier[tier_id] = []
        for galaxy in config["systems"]:
            source_path = P0741_RESULT / "maps" / f"{galaxy}.npz"
            expected_hash = next(
                row["sha256"] for row in p0741["mapFiles"] if row["galaxy"] == galaxy
            )
            if sha256(source_path) != expected_hash:
                raise ValueError(f"P0741 source hash mismatch for {galaxy}")
            with np.load(source_path) as payload:
                axis = np.asarray(payload["axis_kpc"], dtype=float)
                reference = {
                    "gas": np.asarray(payload["gas"], dtype=float),
                    "stars": np.asarray(payload["stars"], dtype=float),
                    "total": np.asarray(payload["total"], dtype=float),
                }
            row = audit.loc[galaxy]
            meta = metadata.loc[galaxy]
            package = extract_galaxy_parameters(
                galaxy,
                axis,
                reference["gas"],
                reference["stars"],
                coefficient_count_per_component=coefficient_count,
                source_observables={
                    "dataset": "THINGS H I plus SINGS/AllWISE fused stellar maps (P0741)",
                    "sourceMapSha256": expected_hash,
                    "distanceMpc": float(row.distance_mpc),
                    "inclinationDeg": float(row.inclination_deg),
                    "positionAngleDeg": float(row.photometric_position_angle_deg),
                    "gasMassSolar": float(row.gas_mass_solar),
                    "stellarMassSolar": float(row.stellar_mass_solar),
                    "stellarMassToLightAssumption": {
                        "solarUnits": 0.5,
                        "status": "fixed_baryonic_conversion_assumption",
                    },
                },
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
                    "coefficient_count_per_component": coefficient_count,
                    "parameter_json_bytes": parameter_path.stat().st_size,
                    "source_npz_bytes": source_path.stat().st_size,
                    "generated_npz_bytes": generated_path.stat().st_size,
                    "parameter_content_sha256": package["contentSha256"],
                    "generated_map_sha256": sha256(generated_path),
                    "gravity_parameter_count": 0,
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
            print(f"{tier_id} {galaxy}: {package['contentSha256'][:12]}")

    scores = pd.DataFrame(score_rows)
    transport = pd.DataFrame(transport_rows)
    catalog = pd.DataFrame(catalog_rows)
    vertical = pd.DataFrame(vertical_rows)
    scores.to_csv(args.output / "roundtrip_scores.csv", index=False)
    transport.to_csv(args.output / "formula_transport_scores.csv", index=False)
    catalog.to_csv(args.output / "parameter_catalog.csv", index=False)
    vertical.to_csv(args.output / "vertical_prior_ensemble.csv", index=False)
    for tier in config["representation"]["tiers"]:
        render_atlas(
            atlas_by_tier[tier["id"]],
            args.output / "tiers" / tier["id"] / "roundtrip_atlas.png",
            tier["id"],
        )

    gates = config["gates"]
    tier_reports: list[dict[str, Any]] = []
    selected_tier: str | None = None
    for tier in config["representation"]["tiers"]:
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
            "maximumMassRelativeError": float(scores[scores.tier == tier_id].mass_relative_error.max())
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
                "coefficientsPerComponent": int(tier["coefficientsPerComponent"]),
                "status": "pass" if passed else "fail",
                "checks": checks,
                "aggregate": aggregate,
                "minimumCellToNumericParameterRatio": float(tier_catalog.cell_to_parameter_ratio.min()),
                "formulaTransport": {
                    "medianRadialSpeedRmseKmS": float(
                        tier_transport.radial_speed_transport_rmse_km_s.median()
                    ),
                    "worstRadialSpeedRmseKmS": float(
                        tier_transport.radial_speed_transport_rmse_km_s.max()
                    ),
                    "medianLosSpeedRmseKmS": float(tier_transport.los_speed_transport_rmse_km_s.median()),
                    "worstLosSpeedRmseKmS": float(tier_transport.los_speed_transport_rmse_km_s.max()),
                },
            }
        )

    status = "pass" if selected_tier is not None else "fail"
    if selected_tier is not None:
        render_atlas(
            atlas_by_tier[selected_tier],
            args.output / "selected_multiscale_spiral_twin_atlas.png",
            selected_tier,
        )
    report_core = {
        "schemaVersion": "sigma-p0743-multiscale-spiral-twin-development-result/1",
        "stage": "P0743",
        "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "p0741ResultSha256": p0741["reportSha256"],
        "p0742FailureSha256": p0742["reportSha256"],
        "developmentDisclosure": config["developmentDisclosure"],
        "selectionRule": config["representation"]["selectionRule"],
        "selectedTier": selected_tier,
        "tiers": tier_reports,
        "systems": len(config["systems"]),
        "validationArraysOpened": 0,
        "holdoutArraysOpened": 0,
        "observedVelocityArraysOpened": 0,
        "fittedGravityParameters": 0,
        "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if selected_tier is None:
        selection = "No frozen multiscale tier passed every gate."
    else:
        selected = next(item for item in tier_reports if item["id"] == selected_tier)
        selection = (
            f"The smallest passing representation is **{selected_tier}**. Total-map median/worst "
            f"normalized error is {selected['aggregate']['total']['medianNormalizedL2']:.3f}/"
            f"{selected['aggregate']['total']['maximumNormalizedL2']:.3f}; fixed-formula radial "
            f"transport median/worst RMSE is {selected['formulaTransport']['medianRadialSpeedRmseKmS']:.2f}/"
            f"{selected['formulaTransport']['worstRadialSpeedRmseKmS']:.2f} km/s."
        )
    summary = f"""# P0743 multiscale spiral fake twins

Status: **{status.upper()}**

{selection}

- Development galaxies: 4
- Multiscale coefficient budgets tested: 128, 256, 512 per baryonic component
- Observed velocity arrays opened: 0
- Validation arrays opened: 0
- Holdout arrays opened: 0
- Fitted gravity parameters: 0
- Report SHA-256: `{report['reportSha256']}`

The selected budget is a development result. It must now be frozen unchanged before a validation or holdout speed target is opened.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "selectedTier": selected_tier, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
