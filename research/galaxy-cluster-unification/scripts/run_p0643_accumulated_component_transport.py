#!/usr/bin/env python3
"""Test a finite path-accumulation gate without opening sealed outcomes."""

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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.geometric_transport import (
    G_SI,
    M_SUN_KG,
    aperture_weighted_statistics,
    component_cancellation,
    high_acceleration_screen,
    resample_surface_density,
    streamline_incoherence,
    thin_sheet_newtonian_field,
)

DEFAULT_PROTOCOL = ROOT / "configs" / "p0643_accumulated_component_transport.json"
DEFAULT_OUTPUT = ROOT / "results" / "p0643_accumulated_component_transport"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian(axis, center_x, center_y, scale, mass):
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values = np.exp(-0.5 * ((xx - center_x) ** 2 + (yy - center_y) ** 2) / scale**2)
    cell = float(axis[1] - axis[0])
    return values * float(mass) / (float(np.sum(values)) * cell**2)


def base_fields(stars, gas, cell_kpc, a0):
    star_field = thin_sheet_newtonian_field(stars, cell_kpc)
    gas_field = thin_sheet_newtonian_field(gas, cell_kpc)
    total = stars + gas
    total_field = thin_sheet_newtonian_field(total, cell_kpc)
    path = streamline_incoherence(total_field, cell_kpc)
    cancellation = component_cancellation(star_field, gas_field)
    screen = high_acceleration_screen(total_field.magnitude_m_s2, a0)
    return total, total_field, path, cancellation, screen


def survival(trace_length_kpc, length_kpc, exponent):
    ratio = np.maximum(np.asarray(trace_length_kpc, dtype=float) / float(length_kpc), 0.0)
    return 1.0 - np.exp(-np.power(ratio, float(exponent)))


def score_grid(case, domain, scenario, stars, gas, cell_kpc, protocol):
    definitions = protocol["definitions"]
    grid = protocol["sensitivity_grid"]
    total, field, path, cancellation, screen = base_fields(
        stars, gas, cell_kpc, float(definitions["a0_m_s2"])
    )
    rows = []
    for length in grid["Lc_kpc"]:
        for exponent in grid["q"]:
            accumulated = cancellation * survival(path.trace_length_kpc, length, exponent)
            activation = screen * accumulated
            stats = aperture_weighted_statistics(
                activation, total, field.magnitude_m_s2, cell_kpc
            )
            rows.append(
                {
                    "case": case,
                    "domain": domain,
                    "scenario": scenario,
                    "Lc_kpc": float(length),
                    "q": float(exponent),
                    **{f"activation_{key}": value for key, value in stats.items()},
                    "geometry_weighted_mean": aperture_weighted_statistics(
                        accumulated, total, field.magnitude_m_s2, cell_kpc
                    )["weighted_mean"],
                    "cancellation_weighted_mean": aperture_weighted_statistics(
                        cancellation, total, field.magnitude_m_s2, cell_kpc
                    )["weighted_mean"],
                    "trace_length_weighted_mean_kpc": aperture_weighted_statistics(
                        path.trace_length_kpc, total, field.magnitude_m_s2, cell_kpc
                    )["weighted_mean"],
                }
            )
    return rows


def synthetic_suite(protocol):
    cells = 257
    small_axis = np.linspace(-12.0, 12.0, cells)
    small_stars = gaussian(small_axis, -1.5, 0.0, 1.0, 3.0e10)
    small_gas = gaussian(small_axis, 1.5, 0.0, 1.8, 7.0e10)
    radial_stars = gaussian(small_axis, 0.0, 0.0, 1.0, 3.0e10)
    radial_gas = gaussian(small_axis, 0.0, 0.0, 1.8, 7.0e10)
    large_axis = np.linspace(-1200.0, 1200.0, cells)
    large_stars = gaussian(large_axis, -150.0, 0.0, 100.0, 3.0e13)
    large_gas = gaussian(large_axis, 150.0, 0.0, 180.0, 7.0e13)
    cases = [
        ("radial_cocentered", radial_stars, radial_gas, small_axis),
        ("small_offset", small_stars, small_gas, small_axis),
        ("large_offset", large_stars, large_gas, large_axis),
        ("large_offset_rotated", np.rot90(large_stars), np.rot90(large_gas), large_axis),
        (
            "large_offset_translated",
            ndimage.shift(large_stars, (17, -13), order=1, mode="constant", cval=0.0),
            ndimage.shift(large_gas, (17, -13), order=1, mode="constant", cval=0.0),
            large_axis,
        ),
    ]
    rows = []
    for name, stars, gas, axis in cases:
        rows.extend(
            score_grid(
                name,
                "synthetic",
                "nominal",
                stars,
                gas,
                float(axis[1] - axis[0]),
                protocol,
            )
        )
    return pd.DataFrame(rows)


def observed_suite(protocol):
    inputs = protocol["map_inputs"]
    rows = []
    galaxy_scales = list(inputs["galaxy_stellar_scale_sensitivity"])
    galaxy_names = {0.5: "low", 1.0: "nominal", 2.0: "high"}
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            nominal_stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        for scale in galaxy_scales:
            rows.extend(
                score_grid(
                    path.stem,
                    "sealed_galaxy_baryons_only",
                    galaxy_names[float(scale)],
                    nominal_stars * float(scale),
                    gas,
                    float(axis[1] - axis[0]),
                    protocol,
                )
            )
    target = int(inputs["cluster_downsample_cells"])
    for path in sorted((ROOT / inputs["clusters"]).glob("*.npz")):
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            maps = {
                scenario: (
                    data[keys[0]].astype(float),
                    data[keys[1]].astype(float),
                )
                for scenario, keys in inputs["cluster_sensitivity_maps"].items()
            }
        cell = float((axis[-1] - axis[0]) / (target - 1))
        for scenario, (stars, gas) in maps.items():
            rows.extend(
                score_grid(
                    path.stem.replace("_baryons", ""),
                    "sealed_cluster_baryons_only",
                    scenario,
                    resample_surface_density(stars, target),
                    resample_surface_density(gas, target),
                    cell,
                    protocol,
                )
            )
    return pd.DataFrame(rows)


def primary_rows(frame, protocol):
    definitions = protocol["definitions"]
    return frame[
        np.isclose(frame.Lc_kpc, float(definitions["primary_Lc_kpc"]))
        & np.isclose(frame.q, float(definitions["primary_q"]))
    ].copy()


def evaluate(protocol, synthetic, observed):
    definitions = protocol["definitions"]
    limits = protocol["predeclared_primary_gates"]
    syn = primary_rows(synthetic, protocol).set_index("case")
    radial = float(syn.loc["radial_cocentered", "activation_weighted_mean"])
    small = float(syn.loc["small_offset", "activation_weighted_mean"])
    large = float(syn.loc["large_offset", "activation_weighted_mean"])
    rotation_error = abs(
        float(syn.loc["large_offset_rotated", "activation_weighted_mean"]) / large - 1.0
    )
    translation_error = abs(
        float(syn.loc["large_offset_translated", "activation_weighted_mean"]) / large - 1.0
    )
    primary = primary_rows(observed, protocol)
    grouped = (
        primary.groupby(["domain", "scenario"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_activation=("activation_weighted_mean", "median"),
            minimum_activation=("activation_weighted_mean", "min"),
            maximum_activation=("activation_weighted_mean", "max"),
            median_trace_length_kpc=("trace_length_weighted_mean_kpc", "median"),
        )
    )
    nominal = grouped[grouped.scenario.eq("nominal")].set_index("domain")
    galaxy = float(nominal.loc["sealed_galaxy_baryons_only", "median_activation"])
    cluster = float(nominal.loc["sealed_cluster_baryons_only", "median_activation"])
    sensitivity_ratios = {}
    for scenario in ("low", "nominal", "high"):
        block = grouped[grouped.scenario.eq(scenario)].set_index("domain")
        sensitivity_ratios[scenario] = float(
            block.loc["sealed_cluster_baryons_only", "median_activation"]
            / max(float(block.loc["sealed_galaxy_baryons_only", "median_activation"]), 1e-15)
        )
    solar_g = G_SI * M_SUN_KG / 149_597_870_700.0**2
    solar_coefficient = (
        float(definitions["maximum_future_lambda_for_solar_proxy"])
        * float(definitions["a0_m_s2"])
        / (float(definitions["a0_m_s2"]) + solar_g)
    )
    gate_results = {
        "radial_null": radial <= limits["radial_cocentered_activation_max"],
        "physical_path_accumulation": large / max(small, 1e-15)
        >= limits["large_offset_over_small_offset_activation_ratio_min"],
        "registered_domain_separation": cluster / max(galaxy, 1e-15)
        >= limits["registered_cluster_to_galaxy_median_activation_ratio_min"],
        "galaxy_channel_small": galaxy <= limits["registered_galaxy_median_activation_max"],
        "cluster_channel_present": cluster >= limits["registered_cluster_median_activation_min"],
        "mass_sensitivity_sign_stable": all(value > 1.0 for value in sensitivity_ratios.values())
        is bool(limits["cluster_ratio_above_one_in_all_mass_sensitivities"]),
        "solar_proxy": solar_coefficient <= limits["solar_1au_coefficient_max"],
        "rotation_covariance": rotation_error <= limits["rotation_covariance_relative_error_max"],
        "translation_covariance": translation_error
        <= limits["translation_covariance_relative_error_max"],
        "sealed_targets_untouched": not bool(protocol["map_inputs"]["sealed_target_outcomes_opened"]),
    }
    sensitivity = (
        observed[observed.scenario.eq("nominal")]
        .groupby(["domain", "Lc_kpc", "q"], as_index=False)
        .activation_weighted_mean.median()
    )
    pivot = sensitivity.pivot_table(
        index=["Lc_kpc", "q"], columns="domain", values="activation_weighted_mean"
    ).reset_index()
    pivot["cluster_to_galaxy_ratio"] = (
        pivot["sealed_cluster_baryons_only"] / pivot["sealed_galaxy_baryons_only"]
    )
    return {
        "status": "pass" if all(gate_results.values()) else "fail",
        "all_primary_gates_pass": bool(all(gate_results.values())),
        "gate_results": gate_results,
        "primary_metrics": {
            "radial_activation": radial,
            "small_offset_activation": small,
            "large_offset_activation": large,
            "large_over_small_ratio": large / max(small, 1e-15),
            "registered_galaxy_median_activation": galaxy,
            "registered_cluster_median_activation": cluster,
            "registered_cluster_to_galaxy_ratio": cluster / max(galaxy, 1e-15),
            "mass_sensitivity_cluster_to_galaxy_ratios": sensitivity_ratios,
            "solar_1au_max_future_lambda_coefficient": solar_coefficient,
            "rotation_covariance_relative_error": rotation_error,
            "translation_covariance_relative_error": translation_error,
        },
        "candidate_advanced": bool(all(gate_results.values())),
        "candidate": {
            "base_geometry": "component_cancellation",
            "Lc_kpc": float(definitions["primary_Lc_kpc"]),
            "q": float(definitions["primary_q"]),
            "spatial_gravity_parameters": 0,
            "universal_new_constants": 1,
        },
        "primary_domain_summary": grouped.to_dict(orient="records"),
        "sensitivity_summary": pivot.to_dict(orient="records"),
    }, grouped, pivot


def make_figure(synthetic, observed, pivot, protocol, output):
    primary_syn = primary_rows(synthetic, protocol)
    primary_obs = primary_rows(observed, protocol)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    primary_syn.set_index("case").activation_weighted_mean.plot.bar(ax=axes[0], logy=True)
    axes[0].set_title("Primary synthetic scale test")
    axes[0].set_ylabel("screened accumulated activation")
    nominal = primary_obs[primary_obs.scenario.eq("nominal")]
    nominal.pivot(index="case", columns="domain", values="activation_weighted_mean").plot.bar(
        ax=axes[1], logy=True
    )
    axes[1].set_title("Primary registered maps")
    for exponent, block in pivot.groupby("q"):
        axes[2].plot(block.Lc_kpc, block.cluster_to_galaxy_ratio, "o-", label=f"q={exponent:g}")
    axes[2].axhline(4.0, color="black", linestyle="--", linewidth=1)
    axes[2].axvline(10.0, color="tab:red", linestyle=":", linewidth=1, label="primary Lc")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("universal coherence length Lc (kpc)")
    axes[2].set_ylabel("cluster / galaxy median activation")
    axes[2].set_title("Sensitivity only")
    axes[2].legend(fontsize=8)
    for axis in axes[:2]:
        axis.tick_params(axis="x", labelrotation=40)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("P0643 finite path accumulation")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_any_P0643_score":
        raise RuntimeError("P0643 protocol is not frozen")
    parent = json.loads((ROOT / protocol["parent_result"]).read_text(encoding="utf-8"))
    if parent["selection_used_sealed_target_outcomes"] or parent["selected_operator"] is not None:
        raise RuntimeError("P0642 blindness or rejection state changed")
    synthetic = synthetic_suite(protocol)
    observed = observed_suite(protocol)
    report, grouped, pivot = evaluate(protocol, synthetic, observed)
    report.update(
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256(protocol_path),
            "source_sha256": sha256(Path(__file__)),
            "coverage": {
                "registered_galaxies": int(observed[observed.domain.str.contains("galaxy")].case.nunique()),
                "registered_clusters": int(observed[observed.domain.str.contains("cluster")].case.nunique()),
                "mass_scenarios_per_system": 3,
                "sensitivity_rows_per_scenario": int(
                    len(protocol["sensitivity_grid"]["Lc_kpc"])
                    * len(protocol["sensitivity_grid"]["q"])
                ),
                "per_object_gravity_parameters": 0,
            },
            "sealed_P0633_kinematics_opened": False,
            "sealed_P0640_lensing_constraints_opened": False,
            "claim_boundary": protocol["claim_boundary"],
        }
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output / "synthetic_accumulation_scores.csv", index=False)
    observed.to_csv(output / "registered_map_accumulation_scores.csv", index=False)
    grouped.to_csv(output / "primary_domain_summary.csv", index=False)
    pivot.to_csv(output / "sensitivity_summary.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_figure(synthetic, observed, pivot, protocol, output / "accumulated_transport_screen.png")
    metrics = report["primary_metrics"]
    summary = f"""# P0643 finite path-accumulation screen

- Primary candidate: component cancellation times `1-exp(-ell/10 kpc)`.
- Status: **{report['status'].upper()}** ({sum(report['gate_results'].values())}/{len(report['gate_results'])} gates).
- Synthetic large/small offset activation ratio: **{metrics['large_over_small_ratio']:.4g}x**.
- Registered median activation: galaxies **{metrics['registered_galaxy_median_activation']:.6g}**, clusters **{metrics['registered_cluster_median_activation']:.6g}**.
- Registered cluster/galaxy ratio: **{metrics['registered_cluster_to_galaxy_ratio']:.4g}x**.
- Candidate advanced to spent-data lensing ablation: **{report['candidate_advanced']}**.
- Sealed velocities and raw lensing opened: **no**.

This result tests whether a universal accumulation scale supplies the domain
lever missing in P0642. It does not yet test raw image positions or establish a
relativistic photon law.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
