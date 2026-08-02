#!/usr/bin/env python3
"""Extract and round-trip real resolved galaxies without gravity parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.galaxy_maps import resolved_map_morphology
from voidscreen.resolved_galaxy_generator import (
    extract_galaxy_parameters,
    package_content_hash,
    render_galaxy,
    roundtrip_metrics,
    sample_vertical_realization,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0720_resolved_galaxy_parameter_roundtrip.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0720_resolved_galaxy_parameter_roundtrip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def render_atlas(records: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(len(records), 3, figsize=(11.0, 3.0 * len(records)))
    for row, record in enumerate(records):
        reference = record["reference"]
        generated = record["generated"]
        scale = max(float(np.max(reference)), np.finfo(float).tiny)
        reference_log = np.log10(reference / scale + 1e-5)
        generated_log = np.log10(generated / scale + 1e-5)
        residual = (generated - reference) / scale
        axes[row, 0].imshow(reference_log.T, origin="lower", cmap="magma", vmin=-5.0, vmax=0.0)
        axes[row, 1].imshow(generated_log.T, origin="lower", cmap="magma", vmin=-5.0, vmax=0.0)
        limit = max(float(np.quantile(np.abs(residual), 0.995)), 1e-4)
        axes[row, 2].imshow(
            residual.T, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        axes[row, 0].set_ylabel(record["galaxy"])
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for axis, title in zip(
        axes[0], ["Observed baryons (log)", "Generated baryons (log)", "Residual / peak"]
    ):
        axis.set_title(title)
    figure.suptitle("P0720 known-galaxy extraction → generation round trips", y=0.999)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_controlled_family(
    package: dict[str, Any], axis: np.ndarray, output: Path
) -> list[dict[str, Any]]:
    variants: list[tuple[str, dict[str, dict[str, Any]]]] = [
        ("replay", {}),
        ("compact", {"gas": {"radial_scale": 0.72}, "stars": {"radial_scale": 0.72}}),
        ("diffuse", {"gas": {"radial_scale": 1.28}, "stars": {"radial_scale": 1.28}}),
        ("smooth", {"gas": {"fourier_scale": 0.0, "residual_scale": 0.0}, "stars": {"fourier_scale": 0.0, "residual_scale": 0.0}}),
        ("asymmetric", {"gas": {"fourier_scale": 1.7, "residual_scale": 1.3}, "stars": {"fourier_scale": 1.7, "residual_scale": 1.3}}),
        ("gas-rich", {"gas": {"mass_scale": 2.0}, "stars": {"mass_scale": 0.7}}),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    catalog: list[dict[str, Any]] = []
    for axis_plot, (name, controls) in zip(axes.ravel(), variants):
        rendered = render_galaxy(package, axis, component_controls=controls)
        total = rendered["total"]
        peak = max(float(np.max(total)), np.finfo(float).tiny)
        axis_plot.imshow(np.log10(total.T / peak + 1e-5), origin="lower", cmap="magma")
        axis_plot.set_title(name)
        axis_plot.set_xticks([])
        axis_plot.set_yticks([])
        catalog.append({"name": name, "componentControls": controls})
    figure.suptitle(f"Parameter-controlled family generated from {package['galaxy']}")
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return catalog


def aggregate_component(scores: pd.DataFrame, component: str) -> dict[str, float]:
    selected = scores[scores.component == component]
    return {
        "median_normalized_l2": float(selected.normalized_l2.median()),
        "maximum_normalized_l2": float(selected.normalized_l2.max()),
        "median_pixel_correlation": float(selected.pixel_correlation.median()),
        "minimum_pixel_correlation": float(selected.pixel_correlation.min()),
        "median_radial_profile_log10_rmse": float(
            selected.radial_profile_log10_rmse.median()
        ),
        "maximum_mass_relative_error": float(selected.mass_relative_error.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    output = arguments.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    maps_directory = ROOT / config["parent_map_directory"]
    audit_path = ROOT / config["parent_audit_csv"]
    audit = pd.read_csv(audit_path).sort_values("galaxy").reset_index(drop=True)

    parameter_directory = output / "parameters"
    generated_directory = output / "generated_maps"
    parameter_directory.mkdir(parents=True, exist_ok=True)
    generated_directory.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    vertical_rows: list[dict[str, Any]] = []
    atlas_records: list[dict[str, Any]] = []
    first_package: dict[str, Any] | None = None
    first_axis: np.ndarray | None = None

    for _, metadata in audit.iterrows():
        galaxy = str(metadata.galaxy)
        source_path = maps_directory / f"{galaxy}.npz"
        with np.load(source_path) as payload:
            axis = np.asarray(payload["axis_kpc"], dtype=float)
            reference = {
                "gas": np.asarray(payload["gas"], dtype=float),
                "stars": np.asarray(payload["stars"], dtype=float),
                "total": np.asarray(payload["total"], dtype=float),
            }
        source_observables = {
            "dataset": "LITTLE THINGS registered baryonic maps (P0639)",
            "sourceMapSha256": sha256(source_path),
            "distanceMpc": float(metadata.distance_mpc),
            "inclinationDeg": float(metadata.inclination_deg),
            "positionAngleDeg": float(metadata.position_angle_deg),
            "gasMassSolar": float(metadata.gas_mass_solar),
            "stellarMassSolar": float(metadata.stellar_mass_solar),
            "stellarMassToLightAssumption": {
                "band": "V",
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
            radial_bins=int(config["radial_bins"]),
            maximum_fourier_mode=int(config["maximum_fourier_mode"]),
            residual_feature_count=int(config["residual_feature_count_per_component"]),
        )
        parameter_path = parameter_directory / f"{galaxy}.json"
        parameter_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        if package["contentSha256"] != package_content_hash(package):
            raise RuntimeError(f"parameter hash replay failed for {galaxy}")
        generated = render_galaxy(package, axis)
        np.savez_compressed(
            generated_directory / f"{galaxy}.npz",
            axis_kpc=axis,
            gas=generated["gas"],
            stars=generated["stars"],
            total=generated["total"],
            parameter_content_sha256=np.asarray(package["contentSha256"]),
        )
        for component in ("gas", "stars", "total"):
            score_rows.append(
                {
                    "galaxy": galaxy,
                    "component": component,
                    **roundtrip_metrics(reference[component], generated[component], axis),
                }
            )
        parameter_count = numeric_parameter_count(package["components"])
        input_cells = int(2 * axis.size**2)
        catalog_rows.append(
            {
                "galaxy": galaxy,
                "cells_per_axis": int(axis.size),
                "input_surface_cells": input_cells,
                "numeric_representation_values": parameter_count,
                "cell_to_parameter_ratio": input_cells / parameter_count,
                "parameter_json_bytes": parameter_path.stat().st_size,
                "source_npz_bytes": source_path.stat().st_size,
                "parameter_content_sha256": package["contentSha256"],
                "gravity_parameter_count": len(package["gravityParameters"]),
                "velocity_targets_used": package["velocityTargetsUsed"],
            }
        )
        atlas_records.append(
            {"galaxy": galaxy, "reference": reference["total"], "generated": generated["total"]}
        )

        for component in ("gas", "stars"):
            morphology = resolved_map_morphology(
                generated[component], disk_axis_kpc=axis, smoothing_sigma_pixel=2.0
            )
            r80 = float(morphology["r80_kpc"])
            z_limit = max(8.0 * (axis[1] - axis[0]), 0.8 * r80)
            z_axis = np.linspace(-z_limit, z_limit, 33)
            for realization in range(int(config["vertical_realizations_per_component"])):
                rng = np.random.default_rng(
                    stable_seed(int(config["vertical_seed"]), galaxy, component, str(realization))
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
                z_second_moment = float(
                    np.sum(density * z_axis[None, None, :] ** 2)
                    / np.sum(density)
                )
                vertical_rows.append(
                    {
                        "galaxy": galaxy,
                        "component": component,
                        "realization": realization,
                        **vertical_metadata,
                        "zAxisLimitKpc": z_limit,
                        "projectionRelativeError": projection_error,
                        "massWeightedZ2Kpc2": z_second_moment,
                    }
                )
        if first_package is None:
            first_package = package
            first_axis = axis

    scores = pd.DataFrame(score_rows)
    catalog = pd.DataFrame(catalog_rows)
    vertical = pd.DataFrame(vertical_rows)
    scores.to_csv(output / "roundtrip_scores.csv", index=False)
    catalog.to_csv(output / "parameter_catalog.csv", index=False)
    vertical.to_csv(output / "vertical_prior_ensemble.csv", index=False)
    render_atlas(atlas_records, output / "known_galaxy_roundtrip_atlas.png")
    assert first_package is not None and first_axis is not None
    family = render_controlled_family(
        first_package, first_axis, output / "parameter_controlled_family.png"
    )
    (output / "parameter_controlled_family.json").write_text(
        json.dumps(
            {
                "seedGalaxy": first_package["galaxy"],
                "note": "These are controlled counterfactual morphologies, not newly observed galaxies.",
                "variants": family,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate = {
        component: aggregate_component(scores, component)
        for component in ("gas", "stars", "total")
    }
    gates = config["gates"]
    checks = {
        "galaxy_count": len(catalog) == int(gates["required_galaxies"]),
        "mass_closure": float(scores.mass_relative_error.max())
        <= float(gates["maximum_mass_relative_error"]),
        "three_dimensional_projection_closure": float(vertical.projectionRelativeError.max())
        <= float(gates["maximum_3d_projection_relative_error"]),
        "total_median_normalized_l2": aggregate["total"]["median_normalized_l2"]
        <= float(gates["total_median_normalized_l2"]),
        "total_maximum_normalized_l2": aggregate["total"]["maximum_normalized_l2"]
        <= float(gates["total_maximum_normalized_l2"]),
        "total_median_pixel_correlation": aggregate["total"]["median_pixel_correlation"]
        >= float(gates["total_median_pixel_correlation"]),
        "total_minimum_pixel_correlation": aggregate["total"]["minimum_pixel_correlation"]
        >= float(gates["total_minimum_pixel_correlation"]),
        "gas_median_normalized_l2": aggregate["gas"]["median_normalized_l2"]
        <= float(gates["gas_median_normalized_l2"]),
        "gas_maximum_normalized_l2": aggregate["gas"]["maximum_normalized_l2"]
        <= float(gates["gas_maximum_normalized_l2"]),
        "stars_median_normalized_l2": aggregate["stars"]["median_normalized_l2"]
        <= float(gates["stars_median_normalized_l2"]),
        "stars_maximum_normalized_l2": aggregate["stars"]["maximum_normalized_l2"]
        <= float(gates["stars_maximum_normalized_l2"]),
        "no_gravity_parameters": int(catalog.gravity_parameter_count.sum()) == 0,
        "no_velocity_targets": not bool(catalog.velocity_targets_used.any()),
        "deterministic_hashes": catalog.parameter_content_sha256.nunique() == len(catalog),
        "vertical_ambiguity_demonstrated": bool(
            (vertical.groupby(["galaxy", "component"]).massWeightedZ2Kpc2.nunique() > 1).all()
        ),
    }
    report = {
        "stage": config["stage"],
        "status": "pass" if all(checks.values()) else "needs_improvement",
        "purpose": config["purpose"],
        "galaxies": len(catalog),
        "components_scored": len(scores),
        "aggregate": aggregate,
        "maximum_3d_projection_relative_error": float(vertical.projectionRelativeError.max()),
        "vertical_realizations": len(vertical),
        "median_cell_to_parameter_ratio": float(catalog.cell_to_parameter_ratio.median()),
        "maximum_cell_to_parameter_ratio": float(catalog.cell_to_parameter_ratio.max()),
        "gravity_parameters": 0,
        "velocity_targets_used": False,
        "checks": checks,
        "commissioning_note": config["commissioning_note"],
        "scientific_scope": {
            "proved": "The compact baryonic representation can replay these registered 2D maps to the reported fidelity and can produce multiple projection-equivalent 3D priors.",
            "not_proved": [
                "unique recovery of true three-dimensional matter density",
                "recovery directly from raw sky images without P0639 preprocessing assumptions",
                "correct rotation speeds or lensing under any gravity theory",
                "generalization beyond the 13 gas-rich dwarf galaxies in this commissioning sample"
            ],
        },
        "source_audit_sha256": sha256(audit_path),
        "config_sha256": sha256(config_path),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = f"""# P0720 resolved-galaxy parameter round trip

- Status: **{report['status'].upper()}**
- Real resolved galaxies: **{report['galaxies']}**
- Gravity parameters used during extraction: **0**
- Observed velocity targets used during extraction: **no**
- Total-map median normalized error: **{aggregate['total']['median_normalized_l2']:.3f}**
- Total-map worst normalized error: **{aggregate['total']['maximum_normalized_l2']:.3f}**
- Total-map median pixel correlation: **{aggregate['total']['median_pixel_correlation']:.3f}**
- Maximum 2D mass-closure error: **{scores.mass_relative_error.max():.3e}**
- Maximum 3D-to-2D projection error: **{report['maximum_3d_projection_relative_error']:.3e}**
- Median input-cell / numeric-representation ratio: **{report['median_cell_to_parameter_ratio']:.1f}×**

This is a representation and generation result, not a gravity result.  Gas and
stellar maps were reduced to radial/Fourier structure plus signed local
features, then regenerated without reading a rotation curve or fitting a
gravity parameter.  The 3D products are ensembles of declared vertical priors:
different thickness and flaring choices project to the same 2D mass map, which
is the physically honest treatment of the missing depth information.

The stellar maps inherit P0639's fixed V-band mass-to-light assumption of 0.5.
The commissioning gates were informed by exploratory work on these same maps;
they must not be described as blind validation.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
