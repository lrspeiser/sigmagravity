#!/usr/bin/env python3
"""Audit the exact tensor-AQUAL activation while outcomes remain sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.geometric_transport import (
    aperture_weighted_statistics,
    resample_surface_density,
)
from voidscreen.tensor_activation import (
    constitutive_tensor_components,
    exact_tensor_activation,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0660_exact_tensor_activation_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def gaussian(axis, center_x, center_y, scale, mass):
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values = np.exp(-0.5 * ((xx - center_x) ** 2 + (yy - center_y) ** 2) / scale**2)
    cell = float(axis[1] - axis[0])
    return values * float(mass) / (float(np.sum(values)) * cell**2)


def activation_row(case, domain, scenario, resolution, stars, gas, cell_kpc, protocol):
    definitions = protocol["definitions"]
    result = exact_tensor_activation(
        stars,
        gas,
        cell_kpc,
        a0_m_s2=float(definitions["a0_m_s2"]),
        coherence_length_kpc=float(definitions["coherence_length_kpc"]),
        coherence_power=float(definitions["coherence_power"]),
        mu_floor=float(definitions["mu_floor"]),
    )
    total = np.asarray(stars, dtype=float) + np.asarray(gas, dtype=float)
    quantities = {
        "sigma": result.sigma,
        "transverse_mismatch": result.transverse_mismatch,
        "survival": result.survival,
        "screen": result.high_acceleration_screen,
        "trace_length_kpc": result.path.trace_length_kpc,
    }
    statistics = {
        name: aperture_weighted_statistics(
            values,
            total,
            result.total_field.magnitude_m_s2,
            cell_kpc,
        )
        for name, values in quantities.items()
    }
    direction_norm = np.hypot(
        result.transport_direction_x,
        result.transport_direction_y,
    )
    return {
        "case": case,
        "domain": domain,
        "scenario": scenario,
        "resolution": resolution,
        "cells": int(np.asarray(stars).shape[0]),
        "cell_kpc": float(cell_kpc),
        **{
            f"{name}_{statistic}": value
            for name, values in statistics.items()
            for statistic, value in values.items()
        },
        "sigma_global_minimum": float(np.min(result.sigma)),
        "sigma_global_maximum": float(np.max(result.sigma)),
        "minimum_eigenvalue_proxy": float(np.min(result.minimum_eigenvalue_proxy)),
        "maximum_direction_norm_error": float(np.max(np.abs(direction_norm - 1.0))),
        "all_coefficients_finite": bool(
            np.all(np.isfinite(result.sigma))
            and np.all(np.isfinite(result.minimum_eigenvalue_proxy))
        ),
    }


def synthetic_audits(protocol):
    axis = np.linspace(-12.0, 12.0, 257)
    cell = float(axis[1] - axis[0])
    radial_stars = gaussian(axis, 0.0, 0.0, 1.0, 3.0e10)
    radial_gas = gaussian(axis, 0.0, 0.0, 1.8, 7.0e10)
    offset_stars = gaussian(axis, -1.5, 0.0, 1.0, 3.0e10)
    offset_gas = gaussian(axis, 1.5, 0.0, 1.8, 7.0e10)
    radial = activation_row(
        "radial_cocentered",
        "synthetic",
        "nominal",
        "primary",
        radial_stars,
        radial_gas,
        cell,
        protocol,
    )
    offset = activation_row(
        "offset",
        "synthetic",
        "nominal",
        "primary",
        offset_stars,
        offset_gas,
        cell,
        protocol,
    )
    rotated = activation_row(
        "offset_rotated",
        "synthetic",
        "nominal",
        "primary",
        np.rot90(offset_stars),
        np.rot90(offset_gas),
        cell,
        protocol,
    )
    rotation_error = abs(
        rotated["sigma_weighted_mean"] / max(offset["sigma_weighted_mean"], 1e-30) - 1.0
    )
    activation = exact_tensor_activation(offset_stars, offset_gas, cell)
    direct = constitutive_tensor_components(
        activation.sigma,
        activation.transport_direction_x,
        activation.transport_direction_y,
    )
    reversed_tensor = constitutive_tensor_components(
        activation.sigma,
        -activation.transport_direction_x,
        -activation.transport_direction_y,
    )
    numerator = np.sqrt(
        sum(float(np.mean((first - second) ** 2)) for first, second in zip(direct, reversed_tensor, strict=True))
    )
    denominator = np.sqrt(sum(float(np.mean(first**2)) for first in direct))
    reversal_error = float(numerator / max(denominator, np.finfo(float).tiny))
    return pd.DataFrame([radial, offset, rotated]), rotation_error, reversal_error


def registered_map_audits(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    map_paths = []
    galaxy_folder = ROOT / inputs["galaxies"]
    for path in sorted(galaxy_folder.glob("*.npz")):
        map_paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            nominal_stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        for scenario, scale in inputs["galaxy_stellar_scale_sensitivity"].items():
            stars = nominal_stars * float(scale)
            for resolution, cells in (
                ("primary", int(inputs["galaxy_native_cells"])),
                ("check", int(inputs["galaxy_resolution_check_cells"])),
            ):
                target_stars = stars if cells == len(axis) else resample_surface_density(stars, cells)
                target_gas = gas if cells == len(axis) else resample_surface_density(gas, cells)
                cell = float((axis[-1] - axis[0]) / (cells - 1))
                rows.append(
                    activation_row(
                        path.stem,
                        "registered_galaxy_baryons_only",
                        scenario,
                        resolution,
                        target_stars,
                        target_gas,
                        cell,
                        protocol,
                    )
                )

    cluster_folder = ROOT / inputs["clusters"]
    for path in sorted(cluster_folder.glob("*.npz")):
        map_paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            scenario_maps = {
                scenario: (data[keys[0]].astype(float), data[keys[1]].astype(float))
                for scenario, keys in inputs["cluster_sensitivity_maps"].items()
            }
        for scenario, (stars, gas) in scenario_maps.items():
            for resolution, cells in (
                ("primary", int(inputs["cluster_primary_cells"])),
                ("check", int(inputs["cluster_resolution_check_cells"])),
            ):
                cell = float((axis[-1] - axis[0]) / (cells - 1))
                rows.append(
                    activation_row(
                        path.stem.replace("_baryons", ""),
                        "registered_cluster_baryons_only",
                        scenario,
                        resolution,
                        resample_surface_density(stars, cells),
                        resample_surface_density(gas, cells),
                        cell,
                        protocol,
                    )
                )
    return pd.DataFrame(rows), map_paths


def evaluate(protocol, parent, synthetic, observed, rotation_error, reversal_error):
    gates = protocol["predeclared_progression_gates"]
    inputs = protocol["map_inputs"]
    primary = observed[observed.resolution.eq("primary")].copy()
    summary = (
        primary.groupby(["domain", "scenario"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_sigma=("sigma_weighted_mean", "median"),
            minimum_sigma=("sigma_weighted_mean", "min"),
            maximum_sigma=("sigma_weighted_mean", "max"),
            median_transverse_mismatch=("transverse_mismatch_weighted_mean", "median"),
            median_survival=("survival_weighted_mean", "median"),
            median_trace_length_kpc=("trace_length_kpc_weighted_mean", "median"),
        )
    )
    nominal = summary[summary.scenario.eq("nominal")].set_index("domain")
    galaxy_sigma = float(nominal.loc["registered_galaxy_baryons_only", "median_sigma"])
    cluster_sigma = float(nominal.loc["registered_cluster_baryons_only", "median_sigma"])
    ratio = cluster_sigma / max(galaxy_sigma, np.finfo(float).tiny)
    sensitivity_ratios = {}
    for scenario in inputs["galaxy_stellar_scale_sensitivity"]:
        block = summary[summary.scenario.eq(scenario)].set_index("domain")
        sensitivity_ratios[scenario] = float(
            block.loc["registered_cluster_baryons_only", "median_sigma"]
            / max(
                float(block.loc["registered_galaxy_baryons_only", "median_sigma"]),
                np.finfo(float).tiny,
            )
        )

    resolution = observed.pivot_table(
        index=["case", "domain", "scenario"],
        columns="resolution",
        values="sigma_weighted_mean",
    ).reset_index()
    resolution["absolute_fractional_change"] = np.abs(
        resolution["check"] / np.maximum(resolution["primary"], np.finfo(float).tiny) - 1.0
    )
    resolution_summary = (
        resolution.groupby("domain", as_index=False)
        .absolute_fractional_change.median()
        .rename(columns={"absolute_fractional_change": "median_absolute_fractional_change"})
    )
    maximum_domain_median_resolution_change = float(
        resolution_summary.median_absolute_fractional_change.max()
    )
    galaxy_count = int(
        primary[primary.domain.eq("registered_galaxy_baryons_only")].case.nunique()
    )
    cluster_count = int(
        primary[primary.domain.eq("registered_cluster_baryons_only")].case.nunique()
    )
    bounded = bool(
        observed.all_coefficients_finite.all()
        and observed.sigma_global_minimum.min() >= 0.0
        and observed.sigma_global_maximum.max() <= 1.0
    )
    minimum_eigenvalue = float(observed.minimum_eigenvalue_proxy.min())
    solar_anisotropy = float(parent["metrics"]["solar_1au_constitutive_anisotropy"])
    radial_sigma = float(
        synthetic.set_index("case").loc["radial_cocentered", "sigma_weighted_mean"]
    )
    definitions = protocol["definitions"]
    gate_results = {
        "P0659_parent": bool(parent["all_progression_gates_pass"])
        is bool(gates["P0659_all_progression_gates_pass"]),
        "galaxy_coverage": galaxy_count == int(gates["registered_galaxy_count"]),
        "cluster_coverage": cluster_count == int(gates["registered_cluster_count"]),
        "bounded_sigma": bounded is bool(gates["sigma_finite_and_in_closed_unit_interval"]),
        "positive_eigenvalue": bool(minimum_eigenvalue > 0.0)
        is bool(gates["minimum_active_constitutive_eigenvalue_strictly_positive"]),
        "radial_null": radial_sigma <= gates["radial_cocentered_sigma_weighted_mean_max"],
        "rotation_covariance": rotation_error <= gates["rotation_covariance_relative_error_max"],
        "direction_reversal": reversal_error
        <= gates["direction_reversal_tensor_relative_error_max"],
        "registered_domain_separation": ratio
        >= gates["registered_cluster_to_galaxy_nominal_median_sigma_ratio_min"],
        "galaxy_channel_small": galaxy_sigma
        <= gates["registered_galaxy_nominal_median_sigma_max"],
        "cluster_channel_present": cluster_sigma
        >= gates["registered_cluster_nominal_median_sigma_min"],
        "mass_sensitivity_sign_stable": bool(min(sensitivity_ratios.values()) > 1.0)
        is bool(gates["cluster_to_galaxy_sigma_ratio_above_one_in_all_mass_sensitivities"]),
        "resolution_stability": maximum_domain_median_resolution_change
        <= gates["median_resolution_change_fraction_max"],
        "solar_proxy": solar_anisotropy
        <= gates["solar_1au_constitutive_anisotropy_max"],
        "no_new_constants": int(definitions["new_universal_constants_after_P0659"])
        == int(gates["new_universal_constants_after_P0659"]),
        "no_per_object_parameters": int(definitions["per_object_gravity_parameters"])
        == int(gates["per_object_gravity_parameters"]),
        "sealed_targets_untouched": not bool(gates["sealed_target_outcomes_opened"]),
    }
    metrics = {
        "radial_cocentered_sigma_weighted_mean": radial_sigma,
        "rotation_covariance_relative_error": float(rotation_error),
        "direction_reversal_tensor_relative_error": float(reversal_error),
        "registered_galaxy_nominal_median_sigma": galaxy_sigma,
        "registered_cluster_nominal_median_sigma": cluster_sigma,
        "registered_cluster_to_galaxy_nominal_median_sigma_ratio": float(ratio),
        "mass_sensitivity_cluster_to_galaxy_ratios": sensitivity_ratios,
        "minimum_constitutive_eigenvalue_proxy": minimum_eigenvalue,
        "maximum_domain_median_resolution_change_fraction": maximum_domain_median_resolution_change,
        "solar_1au_constitutive_anisotropy": solar_anisotropy,
    }
    return gate_results, metrics, summary, resolution, resolution_summary


def make_figure(observed, metrics, resolution_summary, output):
    primary = observed[
        observed.resolution.eq("primary") & observed.scenario.eq("nominal")
    ].copy()
    primary["label"] = primary.domain.str.contains("cluster").map(
        {False: "galaxy", True: "cluster"}
    )
    primary = primary.sort_values(["label", "sigma_weighted_mean"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = primary.label.map({"galaxy": "#3274a1", "cluster": "#d95f02"})
    axes[0].bar(primary.case, primary.sigma_weighted_mean, color=colors)
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_ylabel("weighted exact tensor sigma")
    axes[0].set_title("Registered nominal maps")
    ratios = metrics["mass_sensitivity_cluster_to_galaxy_ratios"]
    axes[1].bar(ratios.keys(), ratios.values(), color="#55a868")
    axes[1].axhline(10.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("cluster / galaxy median sigma")
    axes[1].set_title("Mass-map sensitivities")
    resolution_plot = resolution_summary.copy()
    resolution_plot["label"] = resolution_plot.domain.str.replace(
        "registered_", "", regex=False
    ).str.replace("_baryons_only", "", regex=False)
    axes[2].bar(
        resolution_plot.label,
        resolution_plot.median_absolute_fractional_change,
        color="#c44e52",
    )
    axes[2].axhline(0.35, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("median fractional resolution change")
    axes[2].set_title("Resolution audit")
    figure.suptitle("P0660 exact tensor activation audit")
    figure.tight_layout()
    figure.savefig(output / "p0660_exact_tensor_activation.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0660_activation_score":
        raise RuntimeError("P0660 protocol is not frozen")
    parent_path = ROOT / protocol["parent_result"]
    parent = read_json(parent_path)
    if not parent["all_progression_gates_pass"]:
        raise RuntimeError("P0659 parent no longer passes")

    synthetic, rotation_error, reversal_error = synthetic_audits(protocol)
    observed, map_paths = registered_map_audits(protocol)
    gate_results, metrics, summary, resolution, resolution_summary = evaluate(
        protocol,
        parent,
        synthetic,
        observed,
        rotation_error,
        reversal_error,
    )
    all_pass = bool(all(gate_results.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0660-EXACT-TENSOR-ACTIVATION-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_real_map_field_solves": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(ROOT / "src/voidscreen/tensor_activation.py"),
        "registered_map_manifest_sha256": manifest_sha256(map_paths),
        "coverage": {
            "registered_galaxies": int(
                observed[observed.domain.eq("registered_galaxy_baryons_only")].case.nunique()
            ),
            "registered_clusters": int(
                observed[observed.domain.eq("registered_cluster_baryons_only")].case.nunique()
            ),
            "mass_scenarios": int(observed.scenario.nunique()),
            "resolutions": int(observed.resolution.nunique()),
            "new_universal_constants_after_P0659": int(
                protocol["definitions"]["new_universal_constants_after_P0659"]
            ),
            "per_object_gravity_parameters": int(
                protocol["definitions"]["per_object_gravity_parameters"]
            ),
        },
        "metrics": metrics,
        "gate_results": gate_results,
        "domain_summary": summary.to_dict(orient="records"),
        "resolution_summary": resolution_summary.to_dict(orient="records"),
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    synthetic.to_csv(output / "synthetic_activation_audits.csv", index=False)
    observed.to_csv(output / "registered_map_activation_scores.csv", index=False)
    summary.to_csv(output / "domain_summary.csv", index=False)
    resolution.to_csv(output / "resolution_audit.csv", index=False)
    make_figure(observed, metrics, resolution_summary, output)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0660 exact tensor activation audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Exact nominal weighted sigma: galaxies **{metrics['registered_galaxy_nominal_median_sigma']:.6g}**, clusters **{metrics['registered_cluster_nominal_median_sigma']:.6g}**.
- Exact cluster/galaxy ratio: **{metrics['registered_cluster_to_galaxy_nominal_median_sigma_ratio']:.4g}x**.
- Maximum domain-median resolution change: **{metrics['maximum_domain_median_resolution_change_fraction']:.3%}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633 velocities and P0640 lensing constraints opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
