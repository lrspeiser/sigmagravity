#!/usr/bin/env python3
"""Audit a physical, rather than pixel-bounded, tensor coherence length."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0660_exact_tensor_activation_audit import (
    evaluate as evaluate_registered,
)
from run_p0660_exact_tensor_activation_audit import (
    gaussian,
    manifest_sha256,
    sha256,
)

from voidscreen.geometric_transport import (
    aperture_weighted_statistics,
    resample_surface_density,
)
from voidscreen.physical_tensor_activation import exact_physical_tensor_activation
from voidscreen.tensor_activation import constitutive_tensor_components

DEFAULT_CONFIG = ROOT / "configs" / "p0662_physical_tidal_length_tensor_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def failed_gates(report: dict) -> list[str]:
    return [name for name, passed in report["gate_results"].items() if not passed]


def activation_row(case, domain, scenario, resolution, stars, gas, cell_kpc, protocol):
    definitions = protocol["definitions"]
    result = exact_physical_tensor_activation(
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
        "trace_length_kpc": result.trace_length_kpc,
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
    activation = exact_physical_tensor_activation(offset_stars, offset_gas, cell)
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
        sum(
            float(np.mean((first - second) ** 2))
            for first, second in zip(direct, reversed_tensor, strict=True)
        )
    )
    denominator = np.sqrt(sum(float(np.mean(first**2)) for first in direct))
    return (
        pd.DataFrame([radial, offset, rotated]),
        float(rotation_error),
        float(numerator / max(denominator, np.finfo(float).tiny)),
    )


def registered_map_audits(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    map_paths = []
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
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
                rows.append(
                    activation_row(
                        path.stem,
                        "registered_galaxy_baryons_only",
                        scenario,
                        resolution,
                        stars if cells == len(axis) else resample_surface_density(stars, cells),
                        gas if cells == len(axis) else resample_surface_density(gas, cells),
                        float((axis[-1] - axis[0]) / (cells - 1)),
                        protocol,
                    )
                )
    for path in sorted((ROOT / inputs["clusters"]).glob("*.npz")):
        map_paths.append(path)
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            maps = {
                scenario: (data[keys[0]].astype(float), data[keys[1]].astype(float))
                for scenario, keys in inputs["cluster_sensitivity_maps"].items()
            }
        for scenario, (stars, gas) in maps.items():
            for resolution, cells in (
                ("primary", int(inputs["cluster_primary_cells"])),
                ("check", int(inputs["cluster_resolution_check_cells"])),
            ):
                rows.append(
                    activation_row(
                        path.stem.replace("_baryons", ""),
                        "registered_cluster_baryons_only",
                        scenario,
                        resolution,
                        resample_surface_density(stars, cells),
                        resample_surface_density(gas, cells),
                        float((axis[-1] - axis[0]) / (cells - 1)),
                        protocol,
                    )
                )
    return pd.DataFrame(rows), map_paths


def physical_scale_covariance(protocol):
    cells = 129
    axis = np.linspace(-12.0, 12.0, cells)
    cell = float(axis[1] - axis[0])
    stars = gaussian(axis, -1.5, 0.0, 1.0, 3.0e10)
    gas = gaussian(axis, 1.5, 0.0, 1.8, 7.0e10)
    factor = float(protocol["analytic_tests"]["scale_covariance_length_factor"])
    first = exact_physical_tensor_activation(stars, gas, cell)
    second = exact_physical_tensor_activation(stars, gas, factor * cell)
    active = first.trace_length_kpc > 1e-8
    return float(
        np.median(
            np.abs(
                second.trace_length_kpc[active]
                / (factor * first.trace_length_kpc[active])
                - 1.0
            )
        )
    )


def synthetic_resolution_audit(protocol):
    cells_grid = protocol["analytic_tests"]["synthetic_resolution_cells"]
    maximum = int(max(cells_grid))
    axis = np.linspace(-12.0, 12.0, maximum)
    stars = gaussian(axis, -1.5, 0.0, 1.0, 3.0e10)
    gas = gaussian(axis, 1.5, 0.0, 1.8, 7.0e10)
    rows = []
    for cells in cells_grid:
        cells = int(cells)
        row = activation_row(
            "offset_resolution",
            "synthetic",
            "nominal",
            str(cells),
            stars if cells == maximum else resample_surface_density(stars, cells),
            gas if cells == maximum else resample_surface_density(gas, cells),
            float((axis[-1] - axis[0]) / (cells - 1)),
            protocol,
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("cells")
    reference = float(frame.iloc[-1].sigma_weighted_mean)
    frame["fractional_change_from_finest"] = np.abs(
        frame.sigma_weighted_mean / max(reference, np.finfo(float).tiny) - 1.0
    )
    nonreference = frame.iloc[:-1].fractional_change_from_finest.to_numpy(float)
    return frame, float(np.median(nonreference))


def make_figure(observed, metrics, resolution_summary, output):
    primary = observed[
        observed.resolution.eq("primary") & observed.scenario.eq("nominal")
    ].copy()
    primary["map_type"] = primary.domain.str.contains("cluster").map(
        {False: "galaxy", True: "cluster"}
    )
    primary = primary.sort_values(["map_type", "sigma_weighted_mean"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = primary.map_type.map({"galaxy": "#3274a1", "cluster": "#d95f02"})
    axes[0].bar(primary.case, primary.sigma_weighted_mean, color=colors)
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_ylabel("weighted physical tensor sigma")
    axes[0].set_title("Registered nominal maps")
    sensitivity = metrics["mass_sensitivity_cluster_to_galaxy_ratios"]
    axes[1].bar(sensitivity.keys(), sensitivity.values(), color="#55a868")
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
    axes[2].set_title("Physical-length resolution audit")
    figure.suptitle("P0662 physical tidal-length tensor audit")
    figure.tight_layout()
    figure.savefig(output / "p0662_physical_tidal_length_tensor.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = read_json(config_path)
    if protocol.get("status") != "frozen_before_any_P0662_activation_score":
        raise RuntimeError("P0662 protocol is not frozen")
    parent = read_json(ROOT / protocol["parent_result"])
    predecessor_paths = [ROOT / path for path in protocol["diagnostic_predecessors"]]
    predecessors = [read_json(path) for path in predecessor_paths]
    predecessor_states = (
        failed_gates(predecessors[0]) == ["registered_domain_separation"]
        and failed_gates(predecessors[1]) == ["resolution_stability"]
    )
    if not parent["all_progression_gates_pass"] or not predecessor_states:
        raise RuntimeError("P0662 prerequisite state changed")

    scale_error = physical_scale_covariance(protocol)
    synthetic_resolution, synthetic_resolution_change = synthetic_resolution_audit(protocol)
    synthetic, rotation_error, reversal_error = synthetic_audits(protocol)
    observed, map_paths = registered_map_audits(protocol)
    compatible_protocol = json.loads(json.dumps(protocol))
    compatible_gates = compatible_protocol["predeclared_progression_gates"]
    compatible_gates["cluster_to_galaxy_sigma_ratio_above_one_in_all_mass_sensitivities"] = True
    base_gates, metrics, domain_summary, resolution, resolution_summary = evaluate_registered(
        compatible_protocol,
        parent,
        synthetic,
        observed,
        rotation_error,
        reversal_error,
    )
    gates = protocol["predeclared_progression_gates"]
    sensitivity_minimum = float(
        min(metrics["mass_sensitivity_cluster_to_galaxy_ratios"].values())
    )
    gate_results = {
        "P0659_parent": base_gates.pop("P0659_parent"),
        "P0660_diagnostic_state": failed_gates(predecessors[0])
        == ["registered_domain_separation"],
        "P0661_diagnostic_state": failed_gates(predecessors[1])
        == ["resolution_stability"],
        "no_pixel_bounds": protocol["definitions"]["pixel_floor_or_cap"] == "none",
        "physical_scale_covariance": scale_error
        <= gates["physical_length_scale_covariance_relative_error_max"],
        "synthetic_resolution_stability": synthetic_resolution_change
        <= gates["synthetic_resolution_median_change_max"],
        **base_gates,
    }
    gate_results["mass_sensitivity_sign_stable"] = sensitivity_minimum >= float(
        gates["cluster_to_galaxy_sigma_ratio_min_in_all_mass_sensitivities"]
    )
    all_pass = bool(all(gate_results.values()))
    metrics.update(
        {
            "physical_length_scale_covariance_relative_error": scale_error,
            "synthetic_resolution_median_change_fraction": synthetic_resolution_change,
            "minimum_mass_sensitivity_cluster_to_galaxy_ratio": sensitivity_minimum,
        }
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "P0662-PHYSICAL-TIDAL-LENGTH-TENSOR-AUDIT-RESULTS-1.0.0",
        "status": "pass" if all_pass else "fail",
        "all_progression_gates_pass": all_pass,
        "candidate_advanced_to_real_map_field_solves": all_pass,
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__).resolve()),
        "activation_source_sha256": sha256(
            ROOT / "src/voidscreen/physical_tensor_activation.py"
        ),
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
        "domain_summary": domain_summary.to_dict(orient="records"),
        "resolution_summary": resolution_summary.to_dict(orient="records"),
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    synthetic.to_csv(output / "synthetic_activation_audits.csv", index=False)
    synthetic_resolution.to_csv(output / "synthetic_resolution_audit.csv", index=False)
    observed.to_csv(output / "registered_map_activation_scores.csv", index=False)
    domain_summary.to_csv(output / "domain_summary.csv", index=False)
    resolution.to_csv(output / "resolution_audit.csv", index=False)
    make_figure(observed, metrics, resolution_summary, output)
    failed = [name for name, passed in gate_results.items() if not passed]
    summary_text = f"""# P0662 physical tidal-length tensor audit

- Status: **{'PASS' if all_pass else 'FAIL'}**.
- Physical length scale-covariance error: **{scale_error:.3e}**.
- Synthetic median resolution change: **{synthetic_resolution_change:.3%}**.
- Exact nominal weighted sigma: galaxies **{metrics['registered_galaxy_nominal_median_sigma']:.6g}**, clusters **{metrics['registered_cluster_nominal_median_sigma']:.6g}**.
- Exact cluster/galaxy ratio: **{metrics['registered_cluster_to_galaxy_nominal_median_sigma_ratio']:.4g}x**.
- Registered maximum domain-median resolution change: **{metrics['maximum_domain_median_resolution_change_fraction']:.3%}**.
- Failed frozen gates: **{', '.join(failed) if failed else 'none'}**.
- Sealed P0633 velocities and P0640 lensing constraints opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
