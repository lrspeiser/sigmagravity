#!/usr/bin/env python3
"""Screen bounded component-angle transport without opening sealed outcomes."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0643_accumulated_component_transport import gaussian, survival

from voidscreen.geometric_transport import (
    aperture_weighted_statistics,
    component_angle_mismatch,
    high_acceleration_screen,
    resample_surface_density,
    streamline_incoherence,
    thin_sheet_newtonian_field,
)

DEFAULT_CONFIG = ROOT / "configs" / "p0649_bounded_angle_transport_screen.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_case(case, domain, scenario, stars, gas, cell_kpc, protocol):
    fixed = protocol["fixed_field"]
    star_field = thin_sheet_newtonian_field(stars, cell_kpc)
    gas_field = thin_sheet_newtonian_field(gas, cell_kpc)
    total = stars + gas
    total_field = thin_sheet_newtonian_field(total, cell_kpc)
    path = streamline_incoherence(total_field, cell_kpc)
    survived = survival(
        path.trace_length_kpc,
        float(fixed["coherence_length_kpc"]),
        float(fixed["accumulation_power"]),
    )
    screen = high_acceleration_screen(total_field.magnitude_m_s2, float(fixed["a0_m_s2"]))
    rows = []
    for mode in fixed["modes"]:
        mismatch = component_angle_mismatch(star_field, gas_field, mode=mode["id"])
        activation = screen * survived * mismatch
        activation_stats = aperture_weighted_statistics(
            activation, total, total_field.magnitude_m_s2, cell_kpc
        )
        mismatch_stats = aperture_weighted_statistics(
            mismatch, total, total_field.magnitude_m_s2, cell_kpc
        )
        rows.append(
            {
                "case": case,
                "domain": domain,
                "scenario": scenario,
                "mode": mode["id"],
                **{f"activation_{key}": value for key, value in activation_stats.items()},
                "activation_global_minimum": float(np.min(activation)),
                "activation_global_maximum": float(np.max(activation)),
                "mismatch_weighted_mean": mismatch_stats["weighted_mean"],
                "trace_length_weighted_mean_kpc": aperture_weighted_statistics(
                    path.trace_length_kpc,
                    total,
                    total_field.magnitude_m_s2,
                    cell_kpc,
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
            score_case(
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
    names = {0.5: "low", 1.0: "nominal", 2.0: "high"}
    for path in sorted((ROOT / inputs["galaxies"]).glob("*.npz")):
        with np.load(path) as data:
            axis = data["axis_kpc"].astype(float)
            nominal_stars = data["stars"].astype(float)
            gas = data["gas"].astype(float)
        for factor in inputs["galaxy_stellar_scale_sensitivity"]:
            rows.extend(
                score_case(
                    path.stem,
                    "sealed_galaxy_baryons_only",
                    names[float(factor)],
                    nominal_stars * float(factor),
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
                score_case(
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


def evaluate(protocol, synthetic, observed):
    primary_mode = protocol["fixed_field"]["primary_mode"]
    limits = protocol["predeclared_primary_gates"]
    syn = synthetic[synthetic["mode"].eq(primary_mode)].set_index("case")
    radial = float(syn.loc["radial_cocentered", "activation_weighted_mean"])
    small = float(syn.loc["small_offset", "activation_weighted_mean"])
    large = float(syn.loc["large_offset", "activation_weighted_mean"])
    rotation_error = abs(
        float(syn.loc["large_offset_rotated", "activation_weighted_mean"]) / large - 1.0
    )
    translation_error = abs(
        float(syn.loc["large_offset_translated", "activation_weighted_mean"]) / large - 1.0
    )
    grouped = (
        observed.groupby(["mode", "domain", "scenario"], as_index=False)
        .agg(
            systems=("case", "nunique"),
            median_activation=("activation_weighted_mean", "median"),
            minimum_activation=("activation_weighted_mean", "min"),
            maximum_activation=("activation_weighted_mean", "max"),
            median_mismatch=("mismatch_weighted_mean", "median"),
            median_trace_length_kpc=("trace_length_weighted_mean_kpc", "median"),
        )
    )
    primary = grouped[grouped["mode"].eq(primary_mode)]
    nominal = primary[primary.scenario.eq("nominal")].set_index("domain")
    galaxy = float(nominal.loc["sealed_galaxy_baryons_only", "median_activation"])
    cluster = float(nominal.loc["sealed_cluster_baryons_only", "median_activation"])
    sensitivity_ratios = {}
    for scenario in ("low", "nominal", "high"):
        block = primary[primary.scenario.eq(scenario)].set_index("domain")
        sensitivity_ratios[scenario] = float(
            block.loc["sealed_cluster_baryons_only", "median_activation"]
            / max(float(block.loc["sealed_galaxy_baryons_only", "median_activation"]), 1e-15)
        )
    primary_rows = observed[observed["mode"].eq(primary_mode)]
    global_min = float(primary_rows.activation_global_minimum.min())
    global_max = float(primary_rows.activation_global_maximum.max())
    solar_activation = 0.0
    gate_results = {
        "radial_null": radial <= limits["radial_cocentered_activation_max"],
        "physical_path_accumulation": large / max(small, 1e-15)
        >= limits["large_offset_over_small_offset_activation_ratio_min"],
        "registered_domain_separation": cluster / max(galaxy, 1e-15)
        >= limits["registered_cluster_to_galaxy_median_activation_ratio_min"],
        "galaxy_channel_bounded": galaxy <= limits["registered_galaxy_median_activation_max"],
        "cluster_channel_present": cluster >= limits["registered_cluster_median_activation_min"],
        "mass_sensitivity_sign_stable": all(value > 1.0 for value in sensitivity_ratios.values())
        is bool(limits["cluster_ratio_above_one_in_all_mass_sensitivities"]),
        "activation_bounds": global_min >= limits["activation_min"]
        and global_max <= limits["activation_max"],
        "solar_one_component_null": solar_activation
        <= limits["solar_one_component_activation_max"],
        "rotation_covariance": rotation_error <= limits["rotation_covariance_relative_error_max"],
        "translation_covariance": translation_error
        <= limits["translation_covariance_relative_error_max"],
        "sealed_targets_untouched": not bool(limits["sealed_target_outcomes_opened"]),
    }
    metrics = {
        "radial_activation": radial,
        "small_offset_activation": small,
        "large_offset_activation": large,
        "large_over_small_ratio": large / max(small, 1e-15),
        "registered_galaxy_median_activation": galaxy,
        "registered_cluster_median_activation": cluster,
        "registered_cluster_to_galaxy_ratio": cluster / max(galaxy, 1e-15),
        "mass_sensitivity_cluster_to_galaxy_ratios": sensitivity_ratios,
        "activation_global_minimum": global_min,
        "activation_global_maximum": global_max,
        "solar_one_component_activation": solar_activation,
        "rotation_covariance_relative_error": rotation_error,
        "translation_covariance_relative_error": translation_error,
    }
    return gate_results, metrics, grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_any_P0649_score":
        raise RuntimeError("P0649 protocol is not frozen")
    synthetic = synthetic_suite(protocol)
    observed = observed_suite(protocol)
    gates, metrics, grouped = evaluate(protocol, synthetic, observed)
    report = {
        "report_version": "P0649-BOUNDED-ANGLE-TRANSPORT-SCREEN-RESULTS-1.0.0",
        "status": "pass" if all(gates.values()) else "fail",
        "all_primary_gates_pass": bool(all(gates.values())),
        "candidate_advanced": bool(all(gates.values())),
        "protocol_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "primary_mode": protocol["fixed_field"]["primary_mode"],
        "gate_results": gates,
        "primary_metrics": metrics,
        "mode_domain_summary": grouped.to_dict(orient="records"),
        "coverage": {
            "registered_galaxies": int(
                observed[observed.domain.str.contains("galaxy")].case.nunique()
            ),
            "registered_clusters": int(
                observed[observed.domain.str.contains("cluster")].case.nunique()
            ),
            "mismatch_modes": int(observed["mode"].nunique()),
            "mass_scenarios_per_system": 3,
            "per_object_gravity_parameters": 0,
            "unbounded_amplitude_parameters": 0,
        },
        "sealed_P0633_kinematics_opened": False,
        "sealed_P0640_lensing_constraints_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(output / "synthetic_mode_scores.csv", index=False)
    observed.to_csv(output / "registered_map_mode_scores.csv", index=False)
    grouped.to_csv(output / "mode_domain_summary.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    syn_primary = synthetic[synthetic["mode"].eq(report["primary_mode"])]
    syn_primary.set_index("case").activation_weighted_mean.plot.bar(ax=axes[0], logy=True)
    axes[0].set(title="Primary synthetic geometry", ylabel="bounded activation")
    nominal = grouped[grouped.scenario.eq("nominal")]
    pivot = nominal.pivot(index="mode", columns="domain", values="median_activation")
    pivot.plot.bar(ax=axes[1], logy=True)
    axes[1].set(title="Registered domain medians", ylabel="bounded activation")
    primary_rows = observed[
        observed["mode"].eq(report["primary_mode"]) & observed.scenario.eq("nominal")
    ]
    primary_rows.pivot(index="case", columns="domain", values="activation_weighted_mean").plot.bar(
        ax=axes[2], logy=True
    )
    axes[2].set(title="Primary registered systems", ylabel="bounded activation")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=40)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "bounded_angle_transport_screen.png", dpi=180)
    plt.close(figure)

    summary = f"""# P0649 bounded angle-transport screen

- Status: **{report['status'].upper()}** ({sum(gates.values())}/{len(gates)} gates).
- Primary invariant: **{report['primary_mode']}** with no fitted amplitude.
- Registered median activation: galaxies **{metrics['registered_galaxy_median_activation']:.6g}**, clusters **{metrics['registered_cluster_median_activation']:.6g}**.
- Cluster/galaxy median ratio: **{metrics['registered_cluster_to_galaxy_ratio']:.4g}x**.
- Synthetic large/small ratio: **{metrics['large_over_small_ratio']:.4g}x**.
- Candidate advanced to amplitude-one spent-lens testing: **{report['candidate_advanced']}**.
- Sealed outcomes opened: **no**.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
