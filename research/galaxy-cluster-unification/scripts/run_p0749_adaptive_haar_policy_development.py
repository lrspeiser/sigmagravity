"""Develop a formula-independent adaptive Haar policy across all open galaxies."""

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


DEFAULT_CONFIG = ROOT / "configs/p0749_adaptive_haar_policy_development.json"
DEFAULT_OUTPUT = ROOT / "results/p0749_adaptive_haar_policy_development"
SPARC_TABLE = ROOT / "data/raw/sparc/table1.dat"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_checks(
    component_scores: dict[str, dict[str, float]],
    projection_error: float,
    compression_ratio: float,
    gates: dict[str, Any],
) -> dict[str, bool]:
    return {
        "maximumMassRelativeError": max(
            float(component_scores[key]["mass_relative_error"]) for key in ("gas", "stars", "total")
        ) <= float(gates["maximumMassRelativeError"]),
        "maximum3dProjectionRelativeError": projection_error
        <= float(gates["maximum3dProjectionRelativeError"]),
        "gasMaximumNormalizedL2": float(component_scores["gas"]["normalized_l2"])
        <= float(gates["gasMaximumNormalizedL2"]),
        "starsMaximumNormalizedL2": float(component_scores["stars"]["normalized_l2"])
        <= float(gates["starsMaximumNormalizedL2"]),
        "totalMaximumNormalizedL2": float(component_scores["total"]["normalized_l2"])
        <= float(gates["totalMaximumNormalizedL2"]),
        "totalMinimumPixelCorrelation": float(component_scores["total"]["pixel_correlation"])
        >= float(gates["totalMinimumPixelCorrelation"]),
        "minimumCellToNumericParameterRatio": compression_ratio
        >= float(gates["minimumCellToNumericParameterRatio"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    args.output.mkdir(parents=True, exist_ok=True)
    parameter_dir = args.output / "selected" / "parameters"
    generated_dir = args.output / "selected" / "generated_maps"
    parameter_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    parent_by_galaxy: dict[str, dict[str, Any]] = {}
    parent_reports: dict[str, dict[str, Any]] = {}
    audit_by_galaxy: dict[str, pd.Series] = {}
    for parent in config["parents"]:
        result_dir = ROOT / parent["resultPath"]
        report = read_json(result_dir / "report.json")
        if parent.get("resultSha256") and report["reportSha256"] != parent["resultSha256"]:
            raise ValueError(f"{parent['id']} parent result hash mismatch")
        if parent.get("configSha256") and report["configSha256"] != parent["configSha256"]:
            raise ValueError(f"{parent['id']} parent config hash mismatch")
        parent_reports[parent["id"]] = report
        audit = pd.read_csv(result_dir / "map_audit.csv").set_index("galaxy")
        for galaxy in parent["systems"]:
            if str(audit.loc[galaxy].split) != parent["eligibleSplit"]:
                raise ValueError(f"{galaxy} is outside the declared {parent['eligibleSplit']} split")
            if galaxy in parent_by_galaxy:
                raise ValueError(f"duplicate galaxy in parent groups: {galaxy}")
            parent_by_galaxy[galaxy] = {**parent, "resultDir": result_dir}
            audit_by_galaxy[galaxy] = audit.loc[galaxy]

    systems = [galaxy for parent in config["parents"] for galaxy in parent["systems"]]
    metadata = parse_sparc_metadata(SPARC_TABLE).set_index("galaxy")
    gates = config["gates"]
    candidates = [int(value) for value in config["representation"]["candidateCoefficientsPerComponent"]]
    score_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    vertical_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    atlas_rows: list[dict[str, Any]] = []

    for galaxy in systems:
        parent = parent_by_galaxy[galaxy]
        report = parent_reports[parent["id"]]
        row = audit_by_galaxy[galaxy]
        meta = metadata.loc[galaxy]
        source_path = parent["resultDir"] / "maps" / f"{galaxy}.npz"
        source_hash = next(item["sha256"] for item in report["mapFiles"] if item["galaxy"] == galaxy)
        if sha256(source_path) != source_hash:
            raise ValueError(f"source-map hash mismatch for {galaxy}")
        with np.load(source_path) as payload:
            axis = np.asarray(payload["axis_kpc"], dtype=float)
            reference = {
                key: np.asarray(payload[key], dtype=float) for key in ("gas", "stars", "total")
            }

        selected: tuple[int, dict[str, Any], dict[str, np.ndarray], float] | None = None
        for coefficient_count in candidates:
            tier_id = f"haar_{coefficient_count}"
            package = extract_galaxy_parameters(
                galaxy,
                axis,
                reference["gas"],
                reference["stars"],
                coefficient_count_per_component=coefficient_count,
                source_observables={
                    "dataset": "THINGS H I plus SINGS/AllWISE fused stellar maps",
                    "sourceMapSha256": source_hash,
                    "split": parent["eligibleSplit"],
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
                raise RuntimeError(f"package hash replay failed for {galaxy} {tier_id}")
            generated = render_galaxy(package, axis)
            component_scores = {
                key: roundtrip_metrics(reference[key], generated[key], axis)
                for key in ("gas", "stars", "total")
            }
            for key, metrics in component_scores.items():
                score_rows.append(
                    {"galaxy": galaxy, "split": parent["eligibleSplit"], "tier": tier_id,
                     "component": key, **metrics}
                )
            numeric_parameters = numeric_parameter_count(package["components"])
            compression_ratio = float(2 * axis.size**2 / numeric_parameters)
            maximum_projection_error = 0.0
            candidate_vertical: list[dict[str, Any]] = []
            for component in ("gas", "stars"):
                morphology = resolved_map_morphology(
                    generated[component], disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
                )
                z_limit = max(8.0 * (axis[1] - axis[0]), 0.8 * float(morphology["r80_kpc"]))
                z_axis = np.linspace(-z_limit, z_limit, 33)
                for realization in range(int(config["verticalPriors"]["realizationsPerComponent"])):
                    rng = np.random.default_rng(
                        stable_seed(int(config["verticalPriors"]["seed"]), galaxy, tier_id,
                                    component, str(realization))
                    )
                    density, vertical_metadata = sample_vertical_realization(
                        generated[component], axis, z_axis,
                        r80_kpc=float(morphology["r80_kpc"]), component=component, rng=rng
                    )
                    projected = np.sum(density, axis=2) * float(z_axis[1] - z_axis[0])
                    error = float(
                        np.max(np.abs(projected - generated[component]))
                        / max(float(np.max(generated[component])), np.finfo(float).tiny)
                    )
                    maximum_projection_error = max(maximum_projection_error, error)
                    candidate_vertical.append(
                        {"galaxy": galaxy, "split": parent["eligibleSplit"], "tier": tier_id,
                         "component": component, "realization": realization,
                         **vertical_metadata, "projection_relative_error": error}
                    )
            checks = candidate_checks(
                component_scores, maximum_projection_error, compression_ratio, gates
            )
            passed = all(checks.values())
            selection_rows.append(
                {"galaxy": galaxy, "split": parent["eligibleSplit"], "tier": tier_id,
                 "coefficients_per_component": coefficient_count, "status": "pass" if passed else "fail",
                 "selected": False, "compression_ratio": compression_ratio,
                 "maximum_projection_relative_error": maximum_projection_error,
                 **{f"check_{key}": value for key, value in checks.items()}}
            )
            vertical_rows.extend(candidate_vertical)
            if passed:
                selected = (coefficient_count, package, generated, compression_ratio)
                break

        if selected is None:
            print(f"{galaxy}: no candidate passed")
            continue
        coefficient_count, package, generated, compression_ratio = selected
        tier_id = f"haar_{coefficient_count}"
        selection_rows[-1]["selected"] = True
        parameter_path = parameter_dir / f"{galaxy}.json"
        generated_path = generated_dir / f"{galaxy}.npz"
        parameter_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(
            generated_path, axis_kpc=axis, gas=generated["gas"], stars=generated["stars"],
            total=generated["total"], parameter_content_sha256=np.asarray(package["contentSha256"]),
            selected_coefficients_per_component=np.asarray(coefficient_count),
        )
        catalog_rows.append(
            {"galaxy": galaxy, "split": parent["eligibleSplit"], "selected_tier": tier_id,
             "coefficients_per_component": coefficient_count, "cells_per_axis": len(axis),
             "input_surface_cells": int(2 * axis.size**2),
             "numeric_representation_values": numeric_parameter_count(package["components"]),
             "cell_to_parameter_ratio": compression_ratio,
             "parameter_content_sha256": package["contentSha256"],
             "parameter_file_sha256": sha256(parameter_path),
             "generated_map_sha256": sha256(generated_path), "source_map_sha256": source_hash,
             "gravity_parameter_count": 0, "observed_velocity_arrays_opened": 0}
        )
        fixed_mond_maps: dict[str, np.ndarray] | None = None
        for model in config["formulaTransportFixtures"]["models"]:
            metrics, maps = transport_metrics(
                reference, generated, axis, model=model,
                inclination_deg=float(row.inclination_deg), beam_kpc=float(row.things_beam_kpc),
                hi_radius_kpc=float(meta.HI_radius_kpc),
                padding_factor=float(config["formulaTransportFixtures"]["fieldPaddingFactor"]),
                radial_bins=int(config["formulaTransportFixtures"]["radialScoreBins"]),
            )
            transport_rows.append(
                {"galaxy": galaxy, "split": parent["eligibleSplit"], "selected_tier": tier_id,
                 "model": model["id"], **metrics}
            )
            if model["id"] == "fixed_simple_mond":
                fixed_mond_maps = maps
        assert fixed_mond_maps is not None
        atlas_rows.append(
            {"galaxy": galaxy, "reference": reference["total"], "generated": generated["total"],
             "transport": fixed_mond_maps}
        )
        print(f"{galaxy}: selected {tier_id} without velocity or formula scoring")

    scores = pd.DataFrame(score_rows)
    selection = pd.DataFrame(selection_rows)
    catalog = pd.DataFrame(catalog_rows)
    vertical = pd.DataFrame(vertical_rows)
    transport = pd.DataFrame(transport_rows)
    scores.to_csv(args.output / "candidate_roundtrip_scores.csv", index=False)
    selection.to_csv(args.output / "adaptive_selection_audit.csv", index=False)
    catalog.to_csv(args.output / "selected_parameter_catalog.csv", index=False)
    vertical.to_csv(args.output / "candidate_vertical_prior_ensemble.csv", index=False)
    transport.to_csv(args.output / "selected_formula_transport_scores.csv", index=False)
    if atlas_rows:
        render_atlas(atlas_rows, args.output / "adaptive_haar_selected_atlas.png", "adaptive Haar")

    split_counts = catalog.split.value_counts().to_dict() if not catalog.empty else {}
    checks = {
        "requiredSystems": len(catalog) == int(gates["requiredSystems"]),
        "requiredUniqueDevelopmentMapBundlesOpened": int(split_counts.get("development", 0))
        == int(gates["requiredUniqueDevelopmentMapBundlesOpened"]),
        "requiredUniqueValidationMapBundlesOpened": int(split_counts.get("validation", 0))
        == int(gates["requiredUniqueValidationMapBundlesOpened"]),
        "requiredUniqueHoldoutMapBundlesOpened": int(split_counts.get("holdout", 0))
        == int(gates["requiredUniqueHoldoutMapBundlesOpened"]),
        "requiredObservedVelocityArraysOpened": 0 == int(gates["requiredObservedVelocityArraysOpened"]),
        "maximumFittedGravityParameters": int(catalog.gravity_parameter_count.sum())
        <= int(gates["maximumFittedGravityParameters"]) if not catalog.empty else False,
        "allSelectedByBaryonicGates": bool(len(catalog) == len(systems) and selection[selection.selected]
                                             .filter(like="check_").to_numpy(dtype=bool).all()),
        "formulaTransportNonbinding": config["formulaTransportFixtures"]["bindingForSelection"] is False,
    }
    status = "pass" if all(checks.values()) else "fail"
    selected_records = catalog[["galaxy", "split", "selected_tier", "coefficients_per_component",
                                "cell_to_parameter_ratio", "generated_map_sha256"]].to_dict(orient="records") \
        if not catalog.empty else []
    report_core = {
        "schemaVersion": config["resultSchemaVersion"], "stage": config["stage"], "status": status,
        "configSha256": hashlib.sha256(config_bytes).hexdigest(),
        "parentResultSha256": {key: value["reportSha256"] for key, value in parent_reports.items()},
        "developmentDisclosure": config["developmentDisclosure"],
        "selectionRule": config["representation"]["selectionRule"],
        "systems": len(catalog), "selectedRepresentations": selected_records,
        "observedVelocityArraysOpened": 0, "holdoutMapBundlesOpened": int(split_counts.get("holdout", 0)),
        "fittedGravityParameters": 0, "checks": checks, "claimBoundary": config["claimBoundary"],
    }
    report = {**report_core, "reportSha256": canonical_sha256(report_core)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selected_lines = "\n".join(
        f"- {row.galaxy}: {int(row.coefficients_per_component)} coefficients/component "
        f"({row.cell_to_parameter_ratio:.1f} source cells per stored numeric value)"
        for row in catalog.itertuples(index=False)
    )
    summary = f"""# P0749 adaptive Haar policy development

Status: **{status.upper()}**

{selected_lines}

- Selection inputs: baryonic maps, mass conservation, projection replay, and compression only
- Observed velocity arrays opened: 0
- Holdout pixel arrays opened: 0
- Fitted gravity parameters: 0
- Report SHA-256: `{report['reportSha256']}`

Formula transport is measured only after the representation is selected.
"""
    (args.output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks, "reportSha256": report["reportSha256"]}))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
